import torch

class Cpdsymm:
    def __init__(self, degree, rank, input: torch.tensor, solving_alg = None):
        # take input of shape smaple by features
        self.degree = degree
        self.rank = rank
        self.input = input
        self.input_shape = input.shape[-1]
        CPDblock = torch.randn(rank, self.input_shape)
        self.CPDblock = CPDblock/torch.norm(CPDblock)
        self.c = torch.randn(rank)   
        if solving_alg is not None:
            self.solver = solving_alg

    def forward(self):
        Wx = torch.einsum('sf, rf -> sr', self.input, self.CPDblock)
        jac = Wx.pow(self.degree-1)
        self.halfjac = jac
        out = jac*Wx
        self.cgrad = torch.clone(out)
        out = torch.einsum('sr, r -> s', out, self.c)
        return out

    def block_grad(self):
        self.CPDblockgrad = self.degree*torch.einsum('sr, sf, r -> sfr', self.halfjac, self.input, self.c)
        return self.CPDblockgrad

    def Hessian(self, loss_grad):
        jac = self.block_grad()
        H = torch.einsum('srf, sab -> rfab', jac,jac)
        self.H = H.reshape(self.rank*self.input_shape,self.rank*self.input_shape)
        self.Hc = torch.einsum('sa, sb -> ab', self.cgrad, self.cgrad)
        return

    def Jacobian(self, loss_grad):
        jac = self.block_grad()
        J = torch.einsum('sab, s -> ab', jac, loss_grad)
        self.Jc = torch.einsum('sa, s -> a', self.cgrad, loss_grad)
        self.J = J.reshape(self.rank*self.input_shape)
        return
        
    def solver(self, loss_grad):
        self.Hessian(loss_grad)
        self.Jacobian(loss_grad)
        Hreg = torch.eye(self.H.shape[-1])*1e-2
        creg = torch.eye(self.Hc.shape[-1])*1e-2
        step = torch.linalg.solve(self.H + Hreg, -self.J)
        self.cstep = torch.linalg.solve(self.Hc + creg, -self.Jc)
        self.step = step.reshape(self.rank, self.input_shape)
        return

    def update_blocks(self,loss_grad):
        self.solver(loss_grad)
        self.c += 0.05*self.cstep
        self.CPDblock += 0.05*self.step
        return

sample = 1000
features = 4
degree = 2
rank = 100
input = torch.randn(sample, features)

y = torch.sum(input*input*3, dim=-1)*0.5

loss = lambda x, y: torch.sum((x-y)**2)
loss_grad = lambda x, y: 2*(x-y)
cpd = Cpdsymm(degree, rank, input)
out = cpd.forward()
print(loss(out, y))
for i in range(100):
    g = loss_grad(out, y)
    cpd.update_blocks(g)
    out = cpd.forward()
    print(loss(out, y))
