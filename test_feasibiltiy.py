#%%
import torch

from tensor.layers import TensorTrainLayer

data = torch.randn((4096, 67)).cuda()
target = torch.randn((4096, 1)).cuda()

train = TensorTrainLayer(4, 10, 67, 1).cuda()

#%%
from tensor.bregman import SquareBregFunction
bf = SquareBregFunction()
#%%
import time
start = time.time()
train.tensor_network.accumulating_swipe(data, target, bf, batch_size=-1, lr=1.0, orthonormalize=False, method='exact', eps=1e-2, verbose=2, num_swipes=1)
end = time.time()
print('Time taken:', end-start)
# %%
