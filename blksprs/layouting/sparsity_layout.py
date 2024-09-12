import torch
from torch import Tensor

from blksprs.utils.tools import get_triton_block_size


def create_sparsity_layout(x: Tensor, sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    output = torch.zeros(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size,
                         dtype=torch.int32)

    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = output.stride()

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)
