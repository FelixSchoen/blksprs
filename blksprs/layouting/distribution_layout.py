import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_triton_block_size, validate_dimensions, validate_device, \
    validate_contiguous


def build_distribution_layout(indices: Tensor, sparsity_layout_indices: Tensor,
                              size_target: torch.Size,
                              sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    """Builds the sparsity layout of either the source of a gather or the target of a scatter operation.

    Args:
        indices (Tensor): The block-sparse indices tensor in compressed form used for the gather or scatter operation.
        sparsity_layout_indices (Tensor): The sparsity layout of the indices block-sparse tensor.
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

    output = torch.zeros(size_target[0], size_target[1] // sparsity_block_size, size_target[2] // sparsity_block_size,
                         device=indices.device, dtype=torch.int32)

    i_b, i_r, i_c = indices.size()
    i_b_s, i_r_s, i_c_s = indices.stride()
    s_l_i_b, s_l_i_r, s_l_i_c = sparsity_layout_indices.size()
    s_l_i_b_s, s_l_i_r_s, s_l_i_c_s = sparsity_layout_indices.stride()
    s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
    s_lut_i_r_s, s_lut_i_c_s = sparsity_lut_i.stride()
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = output.stride()

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    validate_triton_block_size(triton_block_size, sparsity_block_size)

    triton_grid = lambda meta: [i_b,
                                triton.cdiv(i_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(i_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_distribution_layout[triton_grid]
     (indices,
      i_b, i_b_s, i_r_s, i_c_s,
      sparsity_layout_indices,
      s_l_i_b, s_l_i_b_s, s_l_i_r, s_l_i_r_s, s_l_i_c, s_l_i_c_s,
      sparsity_lut_i,
      s_lut_i_r, s_lut_i_r_s, s_lut_i_c, s_lut_i_c_s,
      output,
      o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
      sparsity_block_size,
      triton_block_size))

    return output


@triton.jit
def kernel_distribution_layout(i,
                               i_b, i_b_s, i_r_s, i_c_s,
                               s_l_i,
                               s_l_i_b, s_l_i_b_s, s_l_i_r, s_l_i_r_s, s_l_i_c, s_l_i_c_s,
                               s_lut_i,
                               s_lut_i_r, s_lut_i_r_s, s_lut_i_c, s_lut_i_c_s,
                               o,
                               o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                               sparsity_block_size,
                               TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch, row, and column index
    spa_bat_i_idx = (pid_blk * s_lut_i_r_s + 0 * s_lut_i_c_s)
    spa_bat_i_msk = (spa_bat_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_bat_i = tl.load(s_lut_i + spa_bat_i_idx, mask=spa_bat_i_msk)

    spa_row_i_idx = (pid_blk * s_lut_i_r_s + 1 * s_lut_i_c_s)
    spa_row_i_msk = (spa_row_i_idx < s_lut_i_r * s_lut_i_r_s)
    spa_row_i = tl.load(s_lut_i + spa_row_i_idx, mask=spa_row_i_msk)

    blk_i_idx = (pid_blk * i_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
    blk_i_msk = (blk_i_idx < i_b * i_b_s)
    blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk)

    blk_i = blk_i // sparsity_block_size
    blk_v = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), 1, dtype=tl.int32)

    blk_o_idx = ((spa_bat_i * o_b_s) +
                 (spa_row_i * o_r_s) +
                 (blk_i * o_c_s))
    blk_o_msk = (blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_v, mask=blk_o_msk)

    # if tl.min(blk_x) != 0 or tl.max(blk_x) != 0:
    #     blk_o_idx = (pid_bat * o_b_s +
    #                  (((pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_r_s +
    #                   ((pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size) * o_c_s))
    #     blk_o_msk = (blk_o_idx < o_b * o_b_s)
    #     tl.store(o + blk_o_idx, 1, mask=blk_o_msk)
