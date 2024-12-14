import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_triton_block_size, validate_dimensions, validate_device, \
    validate_contiguous


def build_distribution_layout(indices: BlksprsTensor, sparsity_layout_indices: Tensor,
                              dim: int, size_target: torch.Size,
                              sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    """Builds the sparsity layout of either the source of a gather or the target of a scatter operation.

    Args:
        indices (BlksprsTensor): The block-sparse indices tensor in compressed form used for the gather or scatter operation.
        sparsity_layout_indices (Tensor): The sparsity layout of the indices block-sparse tensor.
        dim (int): The dimension along which the operation is conducted.
        size_target (torch.Size): The size of the block-sparse target tensor in regular form.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int, optional): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: The sparsity layout of the source or target tensor.

    """
    validate_dimensions(indices)
    validate_contiguous(indices)
    validate_device(indices)

    sparsity_lut_i = torch.nonzero(sparsity_layout_indices).contiguous()

    adjusted_dim = dim % 3

    output = torch.zeros(size_target[0], size_target[1] // sparsity_block_size, size_target[2] // sparsity_block_size,
                         dtype=torch.bool, device=indices.device)

    i_b, i_r, i_c = indices.size()
    i_b_s, i_r_s, i_c_s = stride(indices)
    s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
    s_lut_i_r_s, s_lut_i_c_s = stride(sparsity_lut_i)
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = stride(output)

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    validate_triton_block_size(triton_block_size, sparsity_block_size)

    triton_grid = lambda meta: [i_b,
                                triton.cdiv(i_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(i_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_distribution_layout[triton_grid]
     (indices,
      i_b, i_b_s, i_r_s, i_c_s,
      sparsity_lut_i,
      s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
      adjusted_dim,
      output,
      o_b, o_b_s, o_r_s, o_c_s,
      sparsity_block_size,
      triton_block_size))

    return output


@triton.jit
def kernel_distribution_layout(i,
                               i_b, i_b_s, i_r_s, i_c_s,
                               s_lut_i,
                               s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
                               dim,
                               o,
                               o_b, o_b_s, o_r_s, o_c_s,
                               sparsity_block_size,
                               TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch, row, and column index
    spa_bat_i_idx = (pid_blk * s_lut_i_r_s + 0 * s_lut_i_c_s)
    spa_bat_i_msk = (spa_bat_i_idx >= 0 and spa_bat_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_bat_i = tl.load(s_lut_i + spa_bat_i_idx, mask=spa_bat_i_msk)

    spa_row_i_idx = (pid_blk * s_lut_i_r_s + 1 * s_lut_i_c_s)
    spa_row_i_msk = (spa_row_i_idx >= 0 and spa_row_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_row_i = tl.load(s_lut_i + spa_row_i_idx, mask=spa_row_i_msk)

    spa_col_i_idx = (pid_blk * s_lut_i_r_s + 2 * s_lut_i_c_s)
    spa_col_i_msk = (spa_col_i_idx >= 0 and spa_col_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_col_i = tl.load(s_lut_i + spa_col_i_idx, mask=spa_col_i_msk)

    blk_i_idx = (pid_blk * i_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_i_msk = (blk_i_idx >= 0 and blk_i_idx < i_b * i_b_s)
    blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk)

    dst_bat_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_bat_i, dtype=tl.int32)
    dst_row_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_row_i, dtype=tl.int32)
    dst_col_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_col_i, dtype=tl.int32)
    if dim == 0:
        dst_bat_idx = blk_i
    elif dim == 1:
        dst_row_idx = blk_i // sparsity_block_size
    elif dim == 2:
        dst_col_idx = blk_i // sparsity_block_size

    blk_v = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), 1, dtype=tl.int1)

    blk_o_idx = ((dst_bat_idx * o_b_s) +
                 (dst_row_idx * o_r_s) +
                 (dst_col_idx * o_c_s))
    blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_v, mask=blk_o_msk)
