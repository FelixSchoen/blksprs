import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.misc.exp import exp
from blksprs.ops.misc.row_wise import row_wise_sum, row_wise_max, row_wise_sub
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import get_triton_block_size, stride
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def softmax(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
            triton_block_size: int = None) -> BlksprsTensor:
    """Computes the softmax of a block-sparse tensor in compressed form.

    Note:
        Sparse blocks are not considered for the calculation of the softmax, i.e., all values are assumed to be ``-inf``.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        BlksprsTensor: The result of the softmax operation as a block-sparse tensor in compressed form.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_rws, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    sparsity_layout_rws_flat = sparsity_layout_rws.reshape(-1)
    sparsity_reverse_lut_rws = ((torch.cumsum(sparsity_layout_rws_flat, dim=-1) - 1) *
                                (sparsity_layout_rws_flat == 1) -
                                (1 * (sparsity_layout_rws_flat == 0)))

    validate_contiguous(sparsity_layout, sparsity_lut, sparsity_reverse_lut_rws)

    return BlksprsTensor(_BlocksparseSoftmax.apply(x, sparsity_layout,
                                                   sparsity_lut,
                                                   sparsity_reverse_lut_rws,
                                                   sparsity_block_size, triton_block_size))


class _BlocksparseSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout: Tensor,
                sparsity_lut: Tensor,
                sparsity_reverse_lut_rws: Tensor,
                sparsity_block_size: int, triton_block_size: int) -> Tensor:
        output = torch.empty_like(x)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        o_b, o_r, o_c = output.size()

        x_row_wise_max, sparsity_layout_rwm = row_wise_max(x, sparsity_layout, sparsity_block_size,
                                                           flag_slice_only=True,
                                                           triton_block_size=triton_block_size)
        x_scaled = row_wise_sub(x, sparsity_layout, x_row_wise_max, sparsity_block_size, triton_block_size)
        x_exp = exp(x_scaled, sparsity_block_size, triton_block_size=triton_block_size)
        x_exp_row_wise_sum, sparsity_layout_rws = row_wise_sum(x_exp, sparsity_layout, sparsity_block_size,
                                                               flag_slice_only=True,
                                                               triton_block_size=triton_block_size)

        s_b, s_r, s_c = x_exp_row_wise_sum.shape
        s_b_s, s_r_s, s_c_s = stride(x_exp_row_wise_sum)
        s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_rws.shape
        s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = stride(sparsity_layout_rws)

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseSoftmax.kernel_blocksparse_softmax[triton_grid]
         (x_exp,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          x_exp_row_wise_sum, s_b, s_b_s, s_r_s, s_c_s,
          s_l_s_b, s_l_s_b_s, s_l_s_r_s,
          sparsity_reverse_lut_rws,
          output,
          triton_block_size))

        # Save for backward pass
        ctx.save_for_backward(output, sparsity_layout, sparsity_lut)
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        o, sparsity_layout, sparsity_lut = ctx.saved_tensors
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        s, sparsity_layout_s = row_wise_sum(grad_output * o, sparsity_layout, sparsity_block_size, flag_slice_only=True,
                                            triton_block_size=triton_block_size)

        sparsity_layout_s_flat = sparsity_layout_s.reshape(-1)
        sparsity_reverse_lut_s = ((torch.cumsum(sparsity_layout_s_flat, dim=-1) - 1) *
                                  (sparsity_layout_s_flat == 1) -
                                  (1 * (sparsity_layout_s_flat == 0)))

        o_b, o_r, o_c = o.size()
        o_b_s, o_r_s, o_c_s = stride(o)
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut)
        s_b, s_r, s_c = s.size()
        s_b_s, s_r_s, s_c_s = stride(s)
        s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_s.size()
        s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = stride(sparsity_layout_s)

        grad_x = torch.empty_like(o, dtype=torch.float)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseSoftmax.kernel_blocksparse_softmax_grad_x[triton_grid]
         (grad_output,
          o_b, o_b_s, o_r_s, o_c_s,
          o,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          s,
          s_b, s_b_s, s_r_s, s_c_s,
          s_l_s_b, s_l_s_b_s, s_l_s_r_s,
          sparsity_reverse_lut_s,
          grad_x,
          o_b, o_b_s, o_r_s, o_c_s,
          triton_block_size
          ))

        return grad_x, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_softmax(x,
                                   x_b, x_b_s, x_r_s, x_c_s,
                                   s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                   s, s_b, s_b_s, s_r_s, s_c_s,
                                   s_l_s_b, s_l_s_b_s, s_l_s_r_s,
                                   r_lut_s,
                                   o,
                                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        # Get reverse sparsity indices for s
        rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                             spa_row * s_l_s_r_s)
        rev_idx_spa_s_msk = (rev_idx_spa_s_idx >= 0 and rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
        rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

        if rev_idx_spa_s == -1:
            tl.device_assert(False)
            return

        # Load x block
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load sum block
        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx >= 0 and blk_s_idx < s_b * s_b_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        # Compute softmax
        buf = tl.div_rn(blk_x, blk_s)

        # Store output
        tl.store(o + blk_x_idx, buf, mask=blk_x_msk)

    @staticmethod
    @triton.jit
    def kernel_blocksparse_softmax_grad_x(g,
                                          g_b, g_b_s, g_r_s, g_c_s,
                                          x,
                                          x_b, x_b_s, x_r_s, x_c_s,
                                          s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                          s,
                                          s_b, s_b_s, s_r_s, s_c_s,
                                          s_l_s_b, s_l_s_b_s, s_l_s_r_s,
                                          r_lut_s,
                                          o,
                                          o_b, o_b_s, o_r_s, o_c_s,
                                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx >= 0 and spa_bat_idx < s_lut_r * s_lut_r_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx >= 0 and spa_row_idx < s_lut_r * s_lut_r_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                             spa_row * s_l_s_r_s)
        rev_idx_spa_s_msk = (rev_idx_spa_s_idx >= 0 and rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
        rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

        if rev_idx_spa_s == -1:
            tl.device_assert(False)
            return

        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx >= 0 and blk_s_idx < s_b * s_b_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        blk_g_idx = ((pid_blk * g_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_c_s)[None, :])
        blk_g_msk = (blk_g_idx >= 0 and blk_g_idx < g_b * g_b_s)
        blk_g = tl.load(g + blk_g_idx, mask=blk_g_msk)

        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx >= 0 and blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        buf = blk_x * (blk_g - blk_s)

        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx >= 0 and blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
