import torch
import numpy as np

def compress(block_left, block_right, rank=5, cut_off=None):
    shape_left, shape_right = block_left.shape, block_right.shape
    contract = torch.einsum('abcd,defg->abcefg', block_left, block_right)
    matrix = contract.flatten(0,2).flatten(1,-1)
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)

    # Truncate u and v to the given error
    s_cumsum = torch.flip(s,dims=[0]).cumsum(0)
    rank = min(rank, s_cumsum.shape[0])
    if cut_off is not None:
        rank = max(min(rank, (s_cumsum / s.sum() > cut_off).sum()), 1)
    split_err = s_cumsum[-rank] / s.sum()
    u = u[..., :rank]
    v = v[:rank]
    s = s[:rank]

    v = s.diag() @ v

    u = u.reshape(*shape_left[:-1], rank)
    v = v.reshape(rank, *shape_right[1:])
    return u, v, split_err

def train_compress(blocks, rank=5, cut_off=None):
    blocks = blocks.copy()
    errors = []
    N = len(blocks)
    for i in range(N - 1):
        u, v, error = compress(blocks[i], blocks[i+1], rank=rank, cut_off=cut_off)
        blocks[i] = u
        blocks[i+1] = v
        errors.append(error)
    return blocks, errors

def train_concat(blocks1, blocks2, device='cpu'):
    if len(blocks1) != len(blocks2):
        raise ValueError("blocks1 and blocks2 must have the same length")

    output = []
    for b1, b2 in zip(blocks1, blocks2):
        output.append(concat(b1, b2, device=device))
        #print("concat", 'b1 shape:', b1.shape, 'b2 shape:', b2.shape, 'output shape:', output[-1].shape)
    return output

def block_left_feature_compress(block, rank=5, cut_off=None):
    shape_block = block.shape
    matrix = block.flatten(0,1).flatten(1,-1)
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)

    # Truncate u and v to the given error
    s_cumsum = torch.flip(s,dims=[0]).cumsum(0)
    rank = min(rank, s_cumsum.shape[0])
    if cut_off is not None:
        rank = max(min(rank, (s_cumsum / s.sum() > cut_off).sum()), 1)
    split_err = s_cumsum[-rank] / s.sum()
    u = u[..., :rank]
    v = v[:rank]
    s = s[:rank]

    v = s.diag() @ v

    #print(u.shape, shape_block, rank)
    u = u.reshape(*shape_block[:2], shape_block[-2], rank)
    v = v.reshape(rank, *shape_block[2:])

    return u, v, split_err

def feature_split(block, feature_shape, rank=5, cut_off=None):
    block = block.reshape(block.shape[0], *feature_shape, *block.shape[-2:])
    split_blocks = []
    errors = []
    for f in range(len(feature_shape)-1):
        u, block, error = block_left_feature_compress(block, rank=rank, cut_off=cut_off)
        print('u shape', u.shape, 'block', block.shape)
        split_blocks.append(u)
        errors.append(error)
    return split_blocks + [block], np.mean(errors)

def concat(block1, block2, device='cpu'):
    if block1.shape[0] == 1 or block2.shape[0] == 1:
        rl = max(block1.shape[0], block2.shape[0])
    else:
        rl = block1.shape[0] + block2.shape[0]
    if block1.shape[3] == 1 or block2.shape[3] == 1:
        rr = max(block1.shape[3], block2.shape[3])
    else:
        rr = block1.shape[3] + block2.shape[3]
    output = torch.zeros((rl, block1.shape[1], block1.shape[2], rr), device=device)
    output[:block1.shape[0], ..., :block1.shape[3]] = block1
    output[-block2.shape[0]:, ..., -block2.shape[3]:] = block2
    return output

class DataCompression:
    def __init__(self, X, device='cpu'):
        self. X = X
        self.device = device
        self.blocks = None

    def sequential_compress(self, batch_size, degree, rank=5, cut_off=None):
        previous_blocks = []
        errors = []
        for i in range(self.X.shape[0] // batch_size):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch_size)
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            if previous_blocks:
                blocks = train_concat(previous_blocks, uncompressed_blocks, self.device)
            else:
                blocks = uncompressed_blocks
            previous_blocks, err = train_compress(blocks, rank=rank, cut_off=cut_off)
            errors.append(err)
            print(f"Batch {i+1}/{self.X.shape[0]//batch_size}, Error: {err[-1].item() if err else 'N/A'}", 'block_shapes:', [b.shape for b in previous_blocks])
        self.blocks = previous_blocks
        return self.blocks

    def parallel_compress(self, batch_size, degree, cuts=5, rank=5, cut_off=None):
        N = self.X.shape[0] // batch_size
        blocks = []
        for i in range(N):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch_size)
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            block, err = train_compress(uncompressed_blocks, rank=rank, cut_off=cut_off)
            blocks.append(block)

        new_blocks = []
        errors = []
        num_blocks_per_cut = (N + cuts - 1) // cuts
        for i in range(0, len(blocks), num_blocks_per_cut):
            cut_blocks = blocks[i:i + num_blocks_per_cut]
            cut_previous = cut_blocks[0]
            for j in range(1, len(cut_blocks)):
                cut_previous = train_concat(cut_previous, cut_blocks[j], device=self.device)
            compress_block, error = train_compress(cut_previous, rank=rank, cut_off=cut_off)
            new_blocks.append(compress_block)
            errors.append(error)
            print(f"Cut {i//num_blocks_per_cut + 1}/{cuts}, Error: {error[-1].item() if error else 'N/A'}", 'block_shapes:', [b.shape for b in compress_block])
        self.blocks = new_blocks
        return self.blocks
    
    def feature_compress(self, batch_size, degree, feature_dim, rank=5, cut_off=None):
        previous_blocks = []
        errors = []
        for i in range(self.X.shape[0] // batch_size):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch_size)            
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            new_blocks = []
            for i in range(len(uncompressed_blocks)):
                split_blocks, error = feature_split(uncompressed_blocks[i], feature_dim, rank=rank, cut_off=cut_off)
                new_blocks.extend(split_blocks)
            if previous_blocks:
                blocks = train_concat(previous_blocks, new_blocks, self.device)
            else:
                blocks = new_blocks
            previous_blocks, err = train_compress(blocks, rank=rank, cut_off=cut_off)
            errors.append(err)
            print(f"Batch {i+1}/{self.X.shape[0]//batch_size}, Error: {err[-1].item() if err else 'N/A'}", 'block_shapes:', [b.shape for b in previous_blocks])
        self.blocks = previous_blocks
        return self.blocks

