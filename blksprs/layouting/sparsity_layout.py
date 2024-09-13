import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_triton_block_size, validate_dimensions, validate_device, \
    validate_dtype_float, validate_contiguous


def create_sparsity_layout(x: Tensor, sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float(x)
    validate_device(x)

    output = torch.zeros(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size,
                         device=x.device, dtype=torch.int32)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = x.stride()
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = output.stride()

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    validate_triton_block_size(triton_block_size, sparsity_block_size)

    triton_grid = lambda meta: [x_b,
                                triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_sparsity_layout[triton_grid]
     (x,
      x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
      output,
      o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
      sparsity_block_size,
      triton_block_size))

    return output


@triton.jit
def kernel_sparsity_layout(x,
                           x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                           o,
                           o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                           sparsity_block_size,
                           TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_bat = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    blk_x_idx = (pid_bat * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    if tl.min(blk_x) != 0 or tl.max(blk_x) != 0:
        blk_o_idx = (pid_bat * o_b_s +
                     (((pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_r_s +
                      ((pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_c_s))
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, 1, mask=blk_o_msk)
