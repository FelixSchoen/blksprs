import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_dtype_float, validate_device, \
    validate_sparsity, validate_dtype_int


def gather(x: Tensor, sparsity_layout_x: Tensor, i: Tensor, sparsity_layout_i: Tensor,
           sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    validate_dimensions(x, i)
    validate_contiguous(x, i)
    validate_dtype_float(x)
    validate_dtype_int(i)
    validate_device(x, i)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x), (i, sparsity_layout_i))

    sparsity_layout_x_flat = sparsity_layout_x.reshape(-1)
    sparsity_reverse_lut_x = ((torch.cumsum(sparsity_layout_x_flat, dim=-1) - 1) *
                              (sparsity_layout_x_flat == 1) -
                              (1 * (sparsity_layout_x_flat == 0)))

    sparsity_lut_i = torch.nonzero(sparsity_layout_i).contiguous()

    validate_contiguous(sparsity_layout_x, sparsity_reverse_lut_x,
                        sparsity_layout_i, sparsity_lut_i)

    return _BlocksparseGather.apply(x, sparsity_layout_x, sparsity_reverse_lut_x,
                                    i, sparsity_layout_i, sparsity_lut_i,
                                    sparsity_block_size, triton_block_size)


class _BlocksparseGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_x: Tensor, sparsity_reverse_lut_x: Tensor,
                i: Tensor, sparsity_layout_i: Tensor, sparsity_lut_i: Tensor,
                sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
        output = torch.empty_like(i, dtype=x.dtype)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = sparsity_layout_x.stride()
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = i.stride()
        s_lut_i_r, s_lut_i_c = sparsity_lut_i.size()
        s_lut_i_r_s, s_lut_i_c_s = sparsity_lut_i.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseGather.kernel_blocksparse_gather[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          sparsity_reverse_lut_x,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut_i, s_lut_i_r, s_lut_i_r_s, s_lut_i_c_s,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    @triton.jit
    def kernel_blocksparse_gather(x,
                                  x_b, x_b_s, x_r_s, x_c_s,
                                  s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                                  r_lut_x,
                                  i,
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

        # Get position of current sparsity block consisting of its batch, row, and column index
        spa_bat_o_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_o_msk = (spa_bat_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_bat_o = tl.load(s_lut_o + spa_bat_o_idx, mask=spa_bat_o_msk)

        spa_row_o_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_o_msk = (spa_row_o_idx < s_lut_o_r * s_lut_o_r_s)
        spa_row_o = tl.load(s_lut_o + spa_row_o_idx, mask=spa_row_o_msk)

        # Load index values
        blk_i_idx = ((pid_blk * i_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * i_c_s)[None, :])
        blk_i_msk = (blk_i_idx < i_b * i_b_s)
        blk_i = tl.load(i + blk_i_idx, mask=blk_i_msk).to(tl.int32)

        # Get positions of sparsity blocks
        pos_spa_blk_x = blk_i // sparsity_block_size
        pos_spa_col_x = blk_i % sparsity_block_size

        # Load reverse sparsity indices for x
        rev_idx_spa_x_idx = ((spa_bat_o * s_l_x_b_s) +
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
