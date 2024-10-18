import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_dtype_int, validate_sparsity_block_size, validate_triton_block_size


def gather_3d(src: Tensor, sparsity_layout_src: Tensor, idx_bat: Tensor, idx_col: Tensor, sparsity_layout_idx: Tensor,
              sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    src = src.contiguous()
    idx_bat = idx_bat.contiguous()
    idx_col = idx_col.contiguous()

    validate_dimensions(src, idx_bat, idx_col)
    validate_contiguous(src, idx_bat, idx_col)
    validate_dtype_int(idx_bat, idx_col)
    validate_device(src, idx_bat, idx_col)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src),
                      (idx_bat, sparsity_layout_idx), (idx_col, sparsity_layout_idx))
    validate_sparsity_block_size(sparsity_block_size, src, idx_bat, idx_col)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_x_flat = sparsity_layout_src.reshape(-1)
    sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                              (sparsity_layout_x_flat == 1) -
                              (1 * (sparsity_layout_x_flat == 0)))

    sparsity_lut_i = torch.nonzero(sparsity_layout_idx).contiguous()

    validate_contiguous(sparsity_layout_src, sparsity_reverse_lut_x,
                        sparsity_layout_idx, sparsity_lut_i)

    return _BlocksparseGather3D.apply(src, sparsity_layout_src, sparsity_reverse_lut_x,
                                      idx_bat, idx_col, sparsity_layout_idx, sparsity_lut_i,
                                      sparsity_block_size, triton_block_size)


class _BlocksparseGather3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                idx_bat: Tensor, idx_col: Tensor, sparsity_layout_i: Tensor, sparsity_lut_i: Tensor,
                sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
        output = torch.empty_like(idx_col, dtype=x.dtype)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = sparsity_layout_x.stride()
        i_b, i_r, i_c = idx_col.size()
        i_b_s, i_r_s, i_c_s = idx_col.stride()
        s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
        s_lut_i_r_s, s_lut_i_c_s = sparsity_lut_i.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseGather3D.kernel_blocksparse_gather_3d[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_reverse_lut_x,
          idx_bat,
          idx_col,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_i, s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.save_for_backward(sparsity_layout_x, idx_col, sparsity_layout_i)
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    @triton.jit
    def kernel_blocksparse_gather_3d(x,
                                     x_b, x_b_s, x_r_s, x_c_s,
                                     s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                                     r_lut_x,
                                     idx_bat,
                                     idx_col,
                                     i_b, i_b_s, i_r_s, i_c_s,
                                     o,
                                     o_b, o_b_s, o_r_s, o_c_s,
                                     s_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c_s,
                                     sparsity_block_size,
                                     TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Load batch index values
        blk_idx_bat_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_bat_msk = (blk_idx_bat_idx < i_b * i_b_s)
        blk_idx_bat = tl.load(idx_bat + blk_idx_bat_idx, mask=blk_idx_bat_msk).to(tl.int32)

        # Get position of current sparsity block consisting of its batch, row, and column index
        spa_bat_o_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_o_msk = (spa_bat_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_bat_o = tl.load(s_lut_o + spa_bat_o_idx, mask=spa_bat_o_msk)

        spa_row_o_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_o_msk = (spa_row_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_row_o = tl.load(s_lut_o + spa_row_o_idx, mask=spa_row_o_msk)

        # Load column index values
        blk_idx_col_idx = ((pid_blk * i_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                           ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_idx_col_msk = (blk_idx_col_idx < i_b * i_b_s)
        blk_idx_col = tl.load(idx_col + blk_idx_col_idx, mask=blk_idx_col_msk).to(tl.int32)

        # Get positions of sparsity blocks
        pos_spa_blk_x = blk_idx_col // sparsity_block_size
        pos_spa_col_x = blk_idx_col % sparsity_block_size

        # Load reverse sparsity indices for x
        rev_idx_spa_x_idx = ((blk_idx_bat * s_l_x_b_s) +
                             (spa_row_o * s_l_x_r_s) +
                             (pos_spa_blk_x * s_l_x_c_s))
        rev_idx_spa_x_msk = (rev_idx_spa_x_idx < s_l_x_b * s_l_x_b_s)
        rev_idx_spa_x = tl.load(r_lut_x + rev_idx_spa_x_idx, mask=rev_idx_spa_x_msk).to(tl.int32)

        # Load x values
        blk_x_idx = ((rev_idx_spa_x * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     (pos_spa_col_x * x_c_s))
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)
