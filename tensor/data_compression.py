import torch
import numpy as np
from tqdm import tqdm

def compress(block_left, block_right, rank=5, cut_off=None, full=True):
    shape_left, shape_right = block_left.shape, block_right.shape
    contract = torch.einsum('abcd,defg->abcefg', block_left, block_right)
    matrix = contract.flatten(0,2).flatten(1,-1)
    rank = min(rank, min(matrix.shape))
    if full:
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)
        # Truncate u and v to the given error
        s_cumsum = torch.flip(s,dims=[0]).cumsum(0)
        if cut_off is not None:
            rank = max(min(rank, (s_cumsum / s.sum() > cut_off).sum()), 1)
        split_err = s_cumsum[-rank] / s.sum()
        u = u[..., :rank]
        v = v[:rank]
        s = s[:rank]
    else:
        u, s, v = torch.svd_lowrank(matrix, q=rank)
        v = v.T
        split_err = s[-1]

    v = s.diag() @ v
    u = u.reshape(*shape_left[:-1], rank)
    v = v.reshape(rank, *shape_right[1:])
    return u, v, split_err

def train_compress(blocks, rank=5, cut_off=None, full=True):
    errors = []
    N = len(blocks)
    for i in range(N - 1):
        u, v, error = compress(blocks[i], blocks[i+1], rank=rank, cut_off=cut_off, full=full)
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

def block_left_feature_compress(block, rank=5, cut_off=None, full=True):
    shape_block = block.shape
    matrix = block.flatten(0,1).flatten(1,-1)
    rank = min(rank, min(matrix.shape))
    if full:
        u, s, v = torch.linalg.svd(matrix, full_matrices=False)
        # Truncate u and v to the given error
        s_cumsum = torch.flip(s,dims=[0]).cumsum(0)
        if cut_off is not None:
            rank = max(min(rank, (s_cumsum / s.sum() > cut_off).sum()), 1)
        split_err = s_cumsum[-rank] / s.sum()
        u = u[..., :rank]
        v = v[:rank]
        s = s[:rank]
    else:
        u, s, v = torch.svd_lowrank(matrix, q=rank)
        v = v.T
        split_err = s[-1]

    v = s.diag() @ v
    u = u.reshape(*shape_block[:2], shape_block[-2], rank)
    v = v.reshape(rank, *shape_block[2:])

    return u, v, split_err

def feature_split(block, feature_shape, rank=5, cut_off=None, full=True):
    block = block.reshape(block.shape[0], *feature_shape, *block.shape[-2:])
    split_blocks = []
    errors = []
    for f in range(len(feature_shape)-1):
        u, block, error = block_left_feature_compress(block, rank=rank, cut_off=cut_off, full=full)
        split_blocks.append(u)
        errors.append(error)
    return split_blocks + [block], torch.mean(torch.stack(errors)).item()

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
    def __init__(self, X, device='cpu', full_svd=True):
        self. X = X
        self.device = device
        self.blocks = None
        self.full = full_svd

    def non_compressed(self, degree, batch_index=None, batch_size=None):
        if batch_index is None or batch_size is None:
            batch_index = 0
            batch_size = self.X.shape[0]
        batch = self.X[batch_index * batch_size:(batch_index + 1) * batch_size]
        left_block = batch.T.reshape(1, -1, 1, batch.shape[0])
        middle_blocks = []
        for j in range(degree-1):
            diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
            middle_blocks.append(diag_block)
        uncompressed_blocks = [left_block] + middle_blocks
        self.blocks = uncompressed_blocks
        return self.blocks

    def sequential_compress(self, batch_size, degree, rank=5, cut_off=None):
        previous_blocks = []
        errors = []
        batches = (self.X.shape[0] + batch_size - 1) // batch_size # round up division
        for i in (tbar:=tqdm(range(batches))):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch.shape[0])
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            if previous_blocks:
                blocks = train_concat(previous_blocks, uncompressed_blocks, self.device)
            else:
                blocks = uncompressed_blocks
            previous_blocks, err = train_compress(blocks, rank=rank, cut_off=cut_off, full=self.full)
            errors.append(err)
            del uncompressed_blocks
            tbar.set_postfix_str(f"Batch | Error: {err[-1].item() if err else 'N/A'}" + f" | block_shapes: {[b.shape for b in previous_blocks]}")
        self.blocks = previous_blocks
        return self.blocks

    def parallel_compress(self, batch_size, degree, iterations=None, cut_size=2, rank=5, cut_off=None, rank_factor=1.5):
        N = (self.X.shape[0] + batch_size - 1) // batch_size # round up division
        blocks = []
        for i in (tbar:=tqdm(range(N))):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch.shape[0])
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            block, err = train_compress(uncompressed_blocks, rank=int(rank_factor * rank / cut_size), cut_off=cut_off, full=self.full)
            tbar.set_postfix_str(f"Compress | Error: {err[-1].item() if err else 'N/A'}" + f" | block_shapes: {[b.shape for b in block]}")
            blocks.append(block)
            del uncompressed_blocks

        iterations = 1+int(np.log10(len(blocks)) / np.log10(cut_size)) if iterations is None else iterations
        print("Starting parallel compression with", iterations, "iterations and cut size", cut_size)
        for it in range(iterations):
            new_blocks = []
            B = (len(blocks) + cut_size - 1) // cut_size
            if it == iterations - 1:
                _rank = rank
            else:
                _rank = int(rank_factor * rank / cut_size)
            for i in (tbar:=tqdm(range(0, len(blocks), cut_size))):
                cut_blocks = blocks[i:i + cut_size]
                cut_previous = cut_blocks[0]
                for j in range(1, len(cut_blocks)):
                    cut_previous = train_concat(cut_previous, cut_blocks[j], device=self.device)
                compress_block, err = train_compress(cut_previous, rank=_rank, cut_off=cut_off, full=self.full)
                new_blocks.append(compress_block)
                tbar.set_postfix_str(f"Cut | Error: {err[-1].item() if err else 'N/A'}" + f" | block_shapes: {[b.shape for b in compress_block]}")
            blocks = new_blocks
        if len(blocks) > 1:
            compressed_block = blocks[0]
            for b in blocks[1:]:
                compressed_block = train_concat(compressed_block, b, device=self.device)
            compressed_blocks, final_err = train_compress(compressed_block, rank=rank, cut_off=cut_off)
        else:
            compressed_blocks = blocks[0]
        self.blocks = compressed_blocks
        return self.blocks
    
    def feature_compress(self, batch_size, degree, feature_dim, rank=5, cut_off=None):
        previous_blocks = []
        errors = []
        batches = (self.X.shape[0] + batch_size - 1) // batch_size # round up division
        for i in (tbar:=tqdm(range(batches))):
            batch = self.X[i * batch_size:(i + 1) * batch_size]
            left_block = batch.T.reshape(1, -1, 1, batch.shape[0])            
            middle_blocks = []
            for j in range(degree-1):
                diag_block = torch.diag_embed(batch.T, dim1=0, dim2=-1).unsqueeze(-2)
                middle_blocks.append(diag_block)
            uncompressed_blocks = [left_block] + middle_blocks
            new_blocks = []
            for j in range(len(uncompressed_blocks)):
                split_blocks, err = feature_split(uncompressed_blocks[j], feature_dim, rank=rank, cut_off=cut_off, full=self.full)
                new_blocks.extend(split_blocks)
            if previous_blocks:
                blocks = train_concat(previous_blocks, new_blocks, self.device)
            else:
                blocks = new_blocks
            previous_blocks, err = train_compress(blocks, rank=rank, cut_off=cut_off, full=self.full)
            errors.append(err)
            tbar.set_postfix_str(f"Batch | Error: {err[-1].item() if err else 'N/A'}" + f" | block_shapes: {[b.shape for b in previous_blocks]}")
        self.blocks = previous_blocks
        return self.blocks