import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.exp import BlocksparseExp
from blksprs.ops.row_wise_sum import BlocksparseRowWiseSum
from blksprs.ops.tools import BaseBlocksparse


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

        sparsity_lut = torch.nonzero(sparsity_layout)

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
                device: torch.device):
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
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c, s_lut_c_s,
          x_exp_row_wise_sum, s_b, s_b_s, s_r, s_r_s, s_c_s,
          s_l_s_b, s_l_s_b_s, s_l_s_r, s_l_s_r_s,
          sparsity_reverse_lut_rws,
          output,
          triton_block_size))

        return output

    @staticmethod
    @triton.jit
    def kernel_blocksparse_softmax(x,
                                   x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                                   s_lut, s_lut_r, s_lut_r_s, s_lut_c, s_lut_c_s,
                                   s, s_b, s_b_s, s_r, s_r_s, s_c_s,
                                   s_l_s_b, s_l_s_b_s, s_l_s_r, s_l_s_r_s,
                                   r_lut_s,
                                   o,
                                   TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        # Get reverse sparsity indices for x
        rev_idx_spa_s_idx = (spa_bat * s_l_s_b_s +
                             spa_row * s_l_s_r_s)
        rev_idx_spa_s_msk = (rev_idx_spa_s_idx < s_l_s_b * s_l_s_b_s + s_l_s_r * s_l_s_r_s)
        rev_idx_spa_s = tl.load(r_lut_s + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

        if rev_idx_spa_s == -1:
            assert False, "Invalid sparsity block"

        # Load x block
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Load sum block
        blk_s_idx = (rev_idx_spa_s * s_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * s_r_s)[:, None] +
                     (tl.arange(0, 1) * s_c_s)[None, :])
        blk_s_msk = (blk_s_idx < s_b * s_b_s + s_r * s_r_s)
        blk_s = tl.load(s + blk_s_idx, mask=blk_s_msk)

        # Compute softmax
        buf = tl.div_rn(blk_x, blk_s)

        # Store output
        tl.store(o + blk_x_idx, buf, mask=blk_x_msk)
