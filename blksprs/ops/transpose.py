from typing import Any

import torch
import triton
from triton import language as tl
from torch import Tensor

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_dtype_float, validate_device


def transpose(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int, triton_block_size: int = None) -> (
        Tensor, Tensor):
    """Transposes a blocksparse tensor.

    Note:
         Returns the transposed tensor and the sparsity layout of the transposed tensor.

    """
    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float(x)
    validate_device(x)

    sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()

    sparsity_lut = torch.nonzero(sparsity_layout_t).contiguous()

    sparsity_layout_flat = sparsity_layout.reshape(-1)
    sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                             (sparsity_layout_flat == 1) -
                             (1 * (sparsity_layout_flat == 0)))
                            .reshape(sparsity_layout.size()).transpose(-1, -2).contiguous().reshape(-1))

    n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

    validate_contiguous(sparsity_layout_t, sparsity_lut, sparsity_reverse_lut)

    return _BlocksparseTranspose.apply(x, sparsity_layout_t, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                                       n_sparse_blocks, triton_block_size), sparsity_layout_t


class _BlocksparseTranspose(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor, sparsity_block_size: int,
                n_sparse_blocks: int, triton_block_size: int) -> (Tensor, Tensor):
        output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout.stride()
        s_lut_r, s_lut_c = sparsity_lut.shape
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseTranspose.kernel_blocksparse_transpose[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s,
          triton_block_size))

        # Save for backward pass
        ctx.sparsity_layout = sparsity_layout
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.sparsity_layout
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return transpose(grad_output, sparsity_layout, sparsity_block_size, triton_block_size)[0], None, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_transpose(x,
                                     x_b, x_b_s, x_r_s, x_c_s,
                                     s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                                     s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                     r_lut,
                                     o,
                                     o_b, o_b_s,
                                     TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current output block consisting of its batch, row, and column index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
        spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s)
        spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

        # Get reverse sparsity indices
        rev_idx_spa_idx = (spa_bat * s_l_b_s +
                           spa_row * s_l_r_s +
                           spa_col * s_l_c_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_b * s_l_b_s)
        rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        if rev_idx_spa == -1:
            assert False, "Invalid sparsity block"

        blk_x_idx = (rev_idx_spa * x_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        blk_x_t = tl.trans(blk_x)

        blk_o_idx = (pid_blk * o_b_s +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, blk_x_t, mask=blk_o_msk)
