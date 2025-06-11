import torch
from sklearn.metrics import r2_score
torch.set_default_dtype(torch.float64)
from tensor.layers import TensorTrainLayer
from tensor.bregman import SquareBregFunction

#%%
r = 2
f = 1
N = 7
# Create input data
S = 5
num_swipes = 2
x = torch.randn(S, f)
xinp = torch.cat((torch.ones(S).unsqueeze(-1),x), dim=-1)
print(xinp)
# Define Bregman function
bf = SquareBregFunction()

y = (x.sum(-1, keepdim=True))**3

layer = TensorTrainLayer(num_carriages=N, bond_dim=r, input_features=f+1, perturb=True)
layer2 = TensorTrainLayer(num_carriages=N, bond_dim=r, input_features=f+1, perturb=False)
for i in range(N):
    print(layer.nodes[i].tensor.shape)
    print(layer2.nodes[i].tensor.shape)


def convergence_criterion(*args):
    y_pred = layer(xinp)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y.detach().cpu().numpy()

    r2 = r2_score(y_true, y_pred)
    print('R² score:', r2)
    return False

# Define Bregman function
print('Num params:', layer.num_parameters())
#%%
from tensor.utils import visualize_tensornetwork
# visualize_tensornetwork(layer.tensor_network)
#%%
layer.tensor_network.accumulating_swipe(xinp.unsqueeze(-1), y.unsqueeze(-1), bf, convergence_criterion=convergence_criterion, orthonormalize=False, method='ridge_exact', eps=0, verbose=2, num_swipes=num_swipes, skip_second=False, direction='l2r')
#%%
