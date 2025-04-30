# %%
import torch

# Set parameters
N = 100         # number of samples
D = 20          # feature dimension (can vary!)
batch_size = 10 # batch size
max_iter = 50   # max Lanczos steps
tol = 1e-6      # tolerance for residual

# Random data
vecset = torch.rand(N, D)
vecsol = torch.rand(N, D)
ddlos = torch.rand(N)

# Split into batches
setbatch = [(vecset[i*batch_size:(i+1)*batch_size],
             ddlos[i*batch_size:(i+1)*batch_size])
            for i in range(N // batch_size)]

# Build A matrix: A = sum_s ddlos_s * (vec_s ⊗ vec_s)
A = torch.einsum("si,sj,s->ij", vecset, vecset, ddlos)

# Solve direct reference solution
rhs_full = (vecsol * ddlos.unsqueeze(1)).sum(0)
x_true = torch.linalg.solve(A, rhs_full)
print("Reference solution x_true computed.")

# %%
# Helper functions
def Hv(vecbatch, ddbatch, v):
    """Apply weighted matrix to vector."""
    coeff = torch.einsum("si, i -> s", vecbatch, v)   # size (batch,)
    return (vecbatch * (coeff * ddbatch).unsqueeze(1)).sum(0)

def V(veclist):
    """Stack a list of vectors as columns."""
    return torch.stack(veclist, dim=1)

def T(a, b):
    """Build tridiagonal matrix from a (diagonal) and b (off-diagonals)."""
    a = torch.tensor(a)
    b = torch.tensor(b)
    Tmat = torch.diag(a)
    Tmat += torch.diag(b[1:], diagonal=1)
    Tmat += torch.diag(b[1:], diagonal=-1)
    return Tmat

# %%
# Lanczos-Galerkin iteration
v = [torch.zeros(D)]   # v[0] is dummy zero vector
a = [0.0]              # a[0] unused dummy
b = [0.0]              # b[0] unused dummy

x0 = torch.rand(D)    # random initial guess

# Compute initial residual r0 = rhs_full - A @ x0
Hx0 = 0
for vecbatch, ddbatch in setbatch:
    Hx0 += Hv(vecbatch, ddbatch, x0)
r0 = rhs_full - Hx0

b1 = torch.norm(r0)
b.append(b1)
v1 = r0 / b1
v.append(v1)

for j in range(1, max_iter+1):
    # Apply H to v[j]
    Hv1 = 0
    for vecbatch, ddbatch in setbatch:
        Hv1 += Hv(vecbatch, ddbatch, v[j])

    v2 = Hv1 - b[j] * v[j-1]
    a1 = torch.dot(v2, v[j])
    a.append(a1)
    v2 = v2 - a1 * v[j]
    b2 = torch.norm(v2)
    b.append(b2)

    # Solve small system to estimate residual
    Vm = V(v[1:j+1])               # columns v1 to vj
    Tm = T(a[1:j+1], b[1:j+1])      # T_m
    rhs = torch.zeros(j)
    rhs[0] = b[1]                  # beta_1 e1

    y = torch.linalg.solve(Tm, rhs)

    residual_estimate = b[j+1] * torch.abs(y[-1])

    print(f"Iter {j}: Residual estimate = {residual_estimate:.2e}")

    if residual_estimate < tol:
        print(f"Converged at iteration {j} with residual {residual_estimate:.2e}")
        break

    if b2 < 1e-12:  # Handle lucky exact convergence
        print(f"Break: β[{j+1}] ≈ 0")
        break

    v.append(v2 / b2)

# Final solution
sol = x0 + Vm @ y

print("\nDone. Final approximate solution computed.")
print("Residual norm ||b - A sol|| =", torch.norm(rhs_full - A @ sol).item())
print((rhs_full - A @ sol).square().sum())
#%%