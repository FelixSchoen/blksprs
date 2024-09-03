import torch
from torch import Tensor

from blksprs.ops.tools import BaseBlocksparse


class BlocksparseTranspose(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor, shuffle_blocks: bool = True) -> (Tensor, Tensor):
        self.validate_tensors(x)

        x_t = x.transpose(1, 2).contiguous()
        sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()

        if shuffle_blocks:
            sparsity_layout_t_flat = sparsity_layout.reshape(-1)
            shuffle_layout = ((torch.cumsum(sparsity_layout_t_flat, dim=-1) - 1) *
                              (sparsity_layout_t_flat == 1) -
                              (1 * (sparsity_layout_t_flat == 0)))
            shuffle_layout = (shuffle_layout.reshape(sparsity_layout.size()).transpose(-1, -2).contiguous()
                              .reshape(-1).to(torch.int))
            shuffle_layout = shuffle_layout[shuffle_layout >= 0]
            x_t = x_t[shuffle_layout, :, :]

        return x_t, sparsity_layout_t
