import torch
from torch.nn import functional as F
from torch import nn

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

class SquareComplexBregFunction(BregFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        y = y.to(dtype=x.dtype, device=x.device)
        x, y = self.transform_forward(x, y)

        loss = torch.norm(x - y)

        d_loss = (x - y)
        dd_loss = torch.full_like(x, 1.0, device=x.device, dtype=x.dtype).unsqueeze(-1)
        return loss, d_loss, dd_loss
    
class SoftmaxSquaredLoss(torch.nn.Module):
    def __init__(self, w=1.0):
        super().__init__()
        self.w = w

    def forward(self, x, y, only_loss=False):
        z = self.w * x
        log_s = torch.log_softmax(z, dim=-1)
        s = log_s.exp()  # softmax(w*x)
        diff = s - y

        # --------
        # Loss: 0.5 * ||s - y||^2
        loss = 0.5 * (diff ** 2).sum(dim=-1, keepdim=True)

        if only_loss:
            return loss

        # --------
        # Gradient: grad = w * J^T @ diff
        # Jacobian of softmax: J = diag(s) - s outer s
        J = torch.diag_embed(s) - torch.einsum('...i,...j->...ij', s, s)
        grad = self.w * torch.einsum('...ij,...j->...i', J, diff)

        # --------
        # Hessian: w^2 * [J^T @ J + correction term]
        # Approximate Hessian by just using J^T @ J (omit 3rd-order correction)
        JTJ = torch.einsum('...ik,...jk->...ij', J, J)
        hessian = self.w ** 2 * JTJ

        return loss, grad, hessian
    
class KLDivBregman(BregFunction):
    def __init__(self, w=1.0, grad_clip=1e3):
        """
        Parameters:
          w : scaling factor that is applied to the logits.
        """
        super().__init__()
        self.w = w
        self.grad_clip = grad_clip

    def forward(self, x, y, only_loss=False):
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
        z = self.w * x # shape [..., C]
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1) # append 0 to logits

        log_s = torch.log_softmax(z, dim=-1) #append 0 and softmax -> remove 0

        loss = F.cross_entropy(log_s, y.argmax(dim=-1), reduction='none') # shape [..., C]
        if only_loss:
            return loss
        s = log_s.exp()
        y = y # remove the last column (corresponding to the appended 0)

        # phi_x = torch.sum(s * log_s, dim=-1, keepdim=True)
        # phi_y = torch.sum(torch.where(y > 0, y * torch.log(y), torch.zeros_like(y)), dim=-1, keepdim=True)

        # d_phi = 1 + log_s.clamp(min=-self.grad_clip, max=self.grad_clip)
        # diff = y - s
        
        # inner = torch.einsum('...i,...i->...', d_phi, diff).unsqueeze(-1)
        # loss = phi_y - phi_x - inner
        
        log_outer = log_s.unsqueeze(-1) + log_s.unsqueeze(-2)  # shape [..., n, n]
        outer = log_outer.exp() # s_i * s_j in a stable way
        grad = self.w * (s - y)[..., :-1] # Exclude the last column (corresponding to the appended 0)

        hessian = self.w**2 * (torch.diag_embed(s) - outer)[..., :-1, :-1] # Exclude the last row and column (corresponding to the appended 0)
        return loss, grad, hessian
    
class BinaryKLDivBregman(BregFunction):
    def __init__(self, w=1.0):
        """
        Parameters:
          w : scaling factor that is applied to the logits.
        """
        super().__init__()
        self.w = w

    def forward(self, x, y, only_loss=False, eps=1e-12):
        """
        Parameters:
          x : logits (un-normalized values)
          y : target probability vector
        Returns:
            loss : KL divergence
            grad : gradient of the loss w.r.t. x
            hessian : Hessian of the loss w.r.t. x
        """
        z = self.w * x
        s = torch.sigmoid(z)

        # Prevent log(0) by clamping
        s = s.clamp(min=eps, max=1 - eps)
        y = y.clamp(min=eps, max=1 - eps)

        # Compute KL divergence
        kl = torch.where(
            y > 0, y * torch.log(y / s), torch.zeros_like(y)
        ) + torch.where(
            y < 1, (1 - y) * torch.log((1 - y) / (1 - s)), torch.zeros_like(y)
        )

        if only_loss:
            return kl

        grad = self.w * (s - y)
        hessian = (self.w ** 2 * s * (1 - s)).unsqueeze(-1)

        return kl, grad, hessian
    
class XEAutogradBregman(BregFunction):
    def __init__(self, w=1.0):
        super().__init__()
        self.w = w

    def forward(self, x, y, only_loss=False):
        x = x.requires_grad_(True)
        
        z = self.w * x # shape [..., C]
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1) # append 0 to logits

        log_s = torch.log_softmax(z, dim=-1) #append 0 and softmax -> remove 0

        loss = F.cross_entropy(log_s, y.argmax(dim=-1), reduction='none') # shape [..., C]
        if only_loss:
            return loss

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

    def forward(self, x, y, only_loss=False):
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

        if only_loss:
            return loss

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
    

class AutogradLoss(nn.Module):
    def __init__(self, loss_func=None):
        super().__init__()
        if loss_func is None:
            loss_func = nn.MSELoss(reduction='none')
        self.loss_func = loss_func
            
    def forward(self, model_out, y_true, only_loss=False):
        model_out = model_out.requires_grad_(True)
        
        loss = self.loss_func(model_out, y_true)
        if only_loss:
            return loss

        # Gradient of loss w.r.t model_out
        d_loss = torch.autograd.grad(
            outputs=loss.sum(), inputs=model_out, create_graph=True, retain_graph=True
        )[0]

        # Hessian of loss w.r.t model_out
        B = []
        for i in range(d_loss.shape[-1]):
            grad2 = torch.autograd.grad(d_loss[..., i].sum(), model_out, retain_graph=True, create_graph=True)[0]
            B.append(grad2.unsqueeze(-2))
        sqd_loss = torch.cat(B, dim=-2)

        return loss.detach(), d_loss.detach(), sqd_loss.detach()
    
from torch.distributions import Normal

class UncertaintyAutogradLoss(nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, y_pred, y_true, only_loss=False):
        y_pred = y_pred.requires_grad_(True)

        y_mean = y_pred[..., 0]
        y_std = torch.nn.functional.softplus(y_pred[..., 1])

        # Define normal distribution
        normal = Normal(loc=y_mean, scale=y_std)

        # NLL Loss
        loss = -normal.log_prob(y_true)
        if only_loss:
            return loss

        # Gradient of loss w.r.t x
        d_loss = torch.autograd.grad(
            outputs=loss.sum(), inputs=y_pred, create_graph=True, retain_graph=True
        )[0]

        # Hessian of loss w.r.t x
        B = []
        for i in range(d_loss.shape[-1]):
            grad2 = torch.autograd.grad(d_loss[..., i].sum(), y_pred, retain_graph=True, create_graph=True)[0]
            B.append(grad2.unsqueeze(-2))
        dd_loss = torch.cat(B, dim=-2)

        return loss.detach(), d_loss.detach(), dd_loss.detach()
    