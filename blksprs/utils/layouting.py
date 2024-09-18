import torch

from blksprs.ops.distribution import scatter


def create_gather_sparsity_layout(src, idx, sparsity_layout_idx, sparsity_block_size, triton_block_size=None):
    idx_o = torch.ones_like(idx)
    idx_t = idx // sparsity_block_size
    sparsity_layout_tgt = torch.ones(
        size=(src.size(0), src.size(1) // sparsity_block_size, src.size(2) // sparsity_block_size), device=src.device)

    return scatter(idx_o, sparsity_layout_idx, idx_t, sparsity_layout_tgt, sparsity_block_size, triton_block_size)
