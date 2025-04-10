import torch

class BregFunction(torch.nn.Module):
    def transform_forward(self, x, y):
        return x, y

    def forward(self, x, y):
        x, y = self.transform_forward(x, y)
        loss = self.psi(x) - self.psi(y) - self.prod(self.d(y), x-y)
        d_loss = self.grad(x, y)
        sqd_loss = self.hess(x, y)
        return loss, d_loss, sqd_loss

    def grad(self, x, y):
        return self.d(x) - self.d(y)

    def hess(self, x, y):
        return self.dsq(x)

    def prod(self, x, y):
        return torch.sum(x*y, dim=-1)
    
    def psi(self, x):
        raise NotImplementedError
    
    def d(self, x):
        raise NotImplementedError

    def dsq(self, x):
        raise NotImplementedError

class SquareBregFunction(BregFunction):
    def transform_forward(self, x, y):
        if x.ndim > 1:
            x = x.flatten(start_dim=1)
        if y.ndim > 1:
            y = y.flatten(start_dim=1)
        return x, y
    
    def prod(self, x, y):
        return torch.sum(x*y, dim=-1)
    
    def psi(self, x):
        return torch.sum(x**2, dim=-1)

    def d(self, x):
        return 2*x

    def dsq(self, x):
        return torch.full_like(x, 2).unsqueeze(-1)
    
class KLDivBregman(BregFunction):
    def __init__(self, w=1.0):
        """
        Parameters:
          w : scaling factor that is applied to the logits.
        """
        super().__init__()
        self.w = w

    def forward(self, x, y):
        """
        Parameters:
          x : logits (un-normalized values)
          y : target probability vector
        Returns:
            loss : KL divergence
            grad : gradient of the loss w.r.t. x
            hessian : Hessian of the loss w.r.t. x
        """
        # x are the logits (un-normalized values); y is a target probability vector.
        z = self.w * x

        log_s = torch.log_softmax(z, dim=-1)
        s = log_s.exp()

        phi_x = torch.sum(s * log_s, dim=-1, keepdim=True)
        phi_y = torch.sum(torch.where(y > 0, y * torch.log(y), torch.zeros_like(y)), dim=-1, keepdim=True)

        d_phi = 1 + log_s
        diff = y - s
        
        inner = torch.einsum('...i,...i->...', d_phi, diff).unsqueeze(-1)
        loss = phi_y - phi_x - inner

        grad = self.w * (s - y)

        log_outer = log_s.unsqueeze(-1) + log_s.unsqueeze(-2)  # shape [..., n, n]
        outer = log_outer.exp() # s_i * s_j in a stable way
        hessian = self.w**2 * (torch.diag_embed(s) - outer + 1e-12)

        return loss, grad, hessian
    
class AutogradBregman(BregFunction):
    def __init__(self, phi_func, forward_transform=None, d_phi_x_func=None):
        super().__init__()
        self.phi_func = phi_func
        self._transform_forward = forward_transform
        self._d_phi_x_func = d_phi_x_func

    def transform_forward(self, x, y):
        if self._transform_forward is not None:
            x, y = self._transform_forward(x, y)
        return super().transform_forward(x, y)

    def forward(self, x, y):
        x = x.requires_grad_(True)
        x, y = self.transform_forward(x, y)

        phi_x = self.phi_func(x)
        phi_y = self.phi_func(y)

        diff = y - x
        if self._d_phi_x_func is not None:
            d_phi_x = torch.autograd.grad(
                outputs=phi_x.sum(), inputs=x, create_graph=True, retain_graph=True
            )[0]
        else:
            d_phi_x = self._d_phi_x_func(x)

        inner = (d_phi_x * diff).sum(-1, keepdim=True)
        loss = phi_y - phi_x - inner

        # Gradient of loss w.r.t x
        d_loss = torch.autograd.grad(
            outputs=loss.sum(), inputs=x, create_graph=True, retain_graph=True
        )[0]

        # Hessian of loss w.r.t x
        B = []
        for i in range(d_loss.shape[-1]):
            grad2 = torch.autograd.grad(d_loss[..., i].sum(), x, retain_graph=True, create_graph=True)[0]
            B.append(grad2.unsqueeze(-2))
        dd_loss = torch.cat(B, dim=-2)

        return loss.detach(), d_loss.detach(), dd_loss.detach()