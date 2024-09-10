import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.exp import BlocksparseExp
from blksprs.ops.row_wise_sum import BlocksparseRowWiseSum
from blksprs.ops.tools import BaseBlocksparse
from blksprs.utils.validation import validate_contiguous


class BlocksparseSoftmax(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

        self.blocksparse_exp = BlocksparseExp(sparsity_block_size, device, triton_block_size=triton_block_size)
        self.blocksparse_row_wise_sum = BlocksparseRowWiseSum(sparsity_block_size, device,
                                                              triton_block_size=triton_block_size, flag_slice_only=True)

    def forward(self, x: Tensor, sparsity_layout: Tensor) -> Tensor:
        self.validate_tensors(x)

        max_val = torch.max(x).item()
        x_scaled = x - max_val

        sparsity_lut = torch.nonzero(sparsity_layout).contiguous()

        sparsity_layout_rws, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
        sparsity_layout_rws_flat = sparsity_layout_rws.reshape(-1)
        sparsity_reverse_lut_rws = ((torch.cumsum(sparsity_layout_rws_flat, dim=-1) - 1) *
                                    (sparsity_layout_rws_flat == 1) -
                                    (1 * (sparsity_layout_rws_flat == 0)))

        return _BlocksparseSoftmax.apply(x_scaled, sparsity_layout,
                                         sparsity_lut,
                                         sparsity_reverse_lut_rws,
                                         self.sparsity_block_size, self.triton_block_size,
                                         self.blocksparse_exp, self.blocksparse_row_wise_sum, self.device)


class _BlocksparseSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout: Tensor,
                sparsity_lut: Tensor,
                sparsity_reverse_lut_rws: Tensor,
                sparsity_block_size: int, triton_block_size: int,
                blocksparse_exp: BlocksparseExp, blocksparse_row_wise_sum: BlocksparseRowWiseSum,
                device: torch.device) -> Tensor:
        validate_contiguous(x, sparsity_layout, sparsity_lut, sparsity_reverse_lut_rws)

        output = torch.zeros_like(x)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = x.stride()
        s_lut_r, s_lut_c = sparsity_lut.shape
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()
        o_b, o_r, o_c = output.shape

        x_exp = blocksparse_exp(x)
        x_exp_row_wise_sum, sparsity_layout_rws = blocksparse_row_wise_sum(x_exp, sparsity_layout)

        s_b, s_r, s_c = x_exp_row_wise_sum.shape
        s_b_s, s_r_s, s_c_s = x_exp_row_wise_sum.stride()
        s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_rws.shape
        s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = sparsity_layout_rws.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseSoftmax.kernel_blocksparse_softmax[triton_grid]
         (x_exp,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          x_exp_row_wise_sum, s_b, s_b_s, s_r_s, s_c_s,
          s_l_s_b, s_l_s_b_s, s_l_s_r,
          sparsity_reverse_lut_rws,
          output,
          triton_block_size))

        # Save for backward pass
        ctx.save_for_backward(output)
        ctx.sparsity_layout = sparsity_layout
        ctx.sparsity_lut = sparsity_lut
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size
        ctx.device = device

        return output

    @staticmethod
    def backward(ctx, grad_output):
        o = ctx.saved_tensors[0]
        sparsity_layout = ctx.sparsity_layout
        sparsity_lut = ctx.sparsity_lut
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size
        device = ctx.device

        blksprs_row_wise_sum = BlocksparseRowWiseSum(sparsity_block_size, device, flag_slice_only=True)
        s, sparsity_layout_s = blksprs_row_wise_sum(grad_output * o, sparsity_layout)

        sparsity_layout_s_flat = sparsity_layout_s.reshape(-1)
        sparsity_reverse_lut_s = ((torch.cumsum(sparsity_layout_s_flat, dim=-1) - 1) *
                                  (sparsity_layout_s_flat == 1) -
                                  (1 * (sparsity_layout_s_flat == 0)))

        o_b, o_r, o_c = o.size()
        o_b_s, o_r_s, o_c_s = o.stride()
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()
        s_b, s_r, s_c = s.size()
        s_b_s, s_r_s, s_c_s = s.stride()
        s_l_s_b, s_l_s_r, s_l_s_c = sparsity_layout_s.size()
        s_l_s_b_s, s_l_s_r_s, s_l_s_c_s = sparsity_layout_s.stride()

        grad_x = torch.zeros_like(o)

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

        return grad_x, None, None, None, None, None, None, None, None

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
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        # Get reverse sparsity indices for x
        rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                             spa_row * s_l_s_r_s)
        rev_idx_spa_s_msk = (rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
        rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

        if rev_idx_spa_s == -1:
            assert False, "Invalid sparsity block"

        # Load x block
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load sum block
        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx < s_b * s_b_s)
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
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
        spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s)
        spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

        rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                             spa_row * s_l_s_r_s)
        rev_idx_spa_s_msk = (rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s)
        rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx < s_b * s_b_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        blk_g_idx = ((pid_blk * g_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * g_c_s)[None, :])
        blk_g_msk = (blk_g_idx < g_b * g_b_s)
        blk_g = tl.load(g + blk_g_idx, mask=blk_g_msk)

        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        buf = blk_x * (blk_g - blk_s)

        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
