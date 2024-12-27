import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import stride, get_triton_block_size


@triton.jit
def kernel_blocksparse_flow_pull(x,
                                 x_b, x_b_s, x_r_s, x_c_s,
                                 o,
                                 o_b, o_b_s, o_r_s, o_c_s,
                                 s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                                 s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                 r_lut,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current output block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
    spa_col_msk = (spa_col_idx >= 0 and spa_col_idx < s_lut_r * s_lut_r_s)
    spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

    # Get reverse sparsity index
    rev_idx_spa_idx = (spa_bat * s_l_o_b_s +
                       spa_row * s_l_o_r_s +
                       spa_col * s_l_o_c_s)
    rev_idx_spa_msk = (rev_idx_spa_idx >= 0 and rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
    rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa == -1:
        tl.device_assert(False)
        return

    blk_x_idx = (rev_idx_spa * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    blk_o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


@triton.jit
def kernel_blocksparse_flow_push(x,
                                 x_b, x_b_s, x_r_s, x_c_s,
                                 s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                                 s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                 r_lut,
                                 o,
                                 o_b, o_b_s, o_r_s, o_c_s,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get sparsity index of current input block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
    spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
    spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

    spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
    spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
    spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

    spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
    spa_col_msk = (spa_col_idx >= 0 and spa_col_idx < s_lut_r * s_lut_r_s)
    spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

    # Get reverse sparsity index
    rev_idx_spa_idx = (spa_bat * s_l_x_b_s +
                       spa_row * s_l_x_r_s +
                       spa_col * s_l_x_c_s)
    rev_idx_spa_msk = (rev_idx_spa_idx >= 0 and rev_idx_spa_idx < s_l_x_b * s_l_x_b_s)
    rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa == -1:
        tl.device_assert(False)
        return

    blk_x_idx = (pid_blk * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    blk_o_idx = (rev_idx_spa * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
    tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)


def flow_forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                 sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
    output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                         dtype=x.dtype, device=x.device)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = stride(x)
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = stride(output)
    s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
    s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)
    s_lut_r, s_lut_c = sparsity_lut.size()
    s_lut_r_s, s_lut_c_s = stride(sparsity_lut)

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    triton_grid = lambda meta: [o_b,
                                triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_blocksparse_flow_pull[triton_grid]
     (x,
      x_b, x_b_s, x_r_s, x_c_s,
      output,
      o_b, o_b_s, o_r_s, o_c_s,
      s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
      sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
      sparsity_reverse_lut,
      triton_block_size))

    # Save for backward pass
    ctx.sparsity_block_size = sparsity_block_size
    ctx.triton_block_size = triton_block_size

    return output
