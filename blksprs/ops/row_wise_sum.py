import torch
import triton
from triton import language as tl
from torch import Tensor

from blksprs.ops.tools import BaseBlocksparse


class BlocksparseRowWiseSum(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor) -> Tensor:
        self.validate_tensors(x)

        sparsity_lut = torch.nonzero(sparsity_layout)
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                (sparsity_layout_flat == 1) -
                                (1 * (sparsity_layout_flat == 0)))

        sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
        sparsity_lut_output = torch.nonzero(sparsity_layout_output)

        o_n_sparsity_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

        return (_BlocksparseRowWiseSum.apply(x,
                                             sparsity_layout, sparsity_lut, sparsity_reverse_lut,
                                             sparsity_layout_output, sparsity_lut_output, o_n_sparsity_blocks_output,
                                             self.sparsity_block_size, self.triton_block_size, self.device),
                sparsity_layout_output)


class _BlocksparseRowWiseSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_layout_output: Tensor, sparsity_lut_output: Tensor, o_n_sparsity_blocks_output: int,
                sparsity_block_size: int, triton_block_size: int,
                device: torch.device) -> Tensor:
        output = torch.zeros(size=(o_n_sparsity_blocks_output,
                                   sparsity_block_size,
                                   sparsity_block_size),
                             device=device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = sparsity_layout.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_o_r, s_lut_o_c = sparsity_lut_output.size()
        s_lut_o_r_s, s_lut_o_c_s = sparsity_lut_output.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        debug_tensor = torch.zeros_like(x)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseRowWiseSum.kernel_blocksparse_row_wise_sum[triton_grid]
         (x,
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          sparsity_lut_output, s_lut_o_r, s_lut_o_r_s, s_lut_o_c, s_lut_o_c_s,
          debug_tensor,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @staticmethod
    @triton.jit
    def kernel_blocksparse_row_wise_sum(x,
                                        x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                                        s_l_x_b, s_l_x_b_s, s_l_x_r, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
                                        r_lut_x,
                                        o,
                                        o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                                        s_lut_o, s_lut_o_r, s_lut_o_r_s, s_lut_o_c, s_lut_o_c_s,
                                        debug_tensor,
                                        sparsity_block_size,
                                        TRITON_BLOCK_SIZE: tl.constexpr):
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_o_r_s + 0 * s_lut_o_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_o_r * s_lut_o_r_s + s_lut_o_c * s_lut_o_c_s)
        spa_bat = tl.load(s_lut_o + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_o_r_s + 1 * s_lut_o_c_s)
        spa_row_msk = (spa_row_idx < s_lut_o_r * s_lut_o_r_s + s_lut_o_c * s_lut_o_c_s)
        spa_row = tl.load(s_lut_o + spa_row_idx, mask=spa_row_msk)

        buf = tl.zeros(shape=(TRITON_BLOCK_SIZE, 1), dtype=tl.float32)

        # Slide over triton block sized segments of input tensor
        for i_seg_tri in range(0, tl.cdiv(s_l_x_c * sparsity_block_size, TRITON_BLOCK_SIZE)):
            # Convert to segment index of sparsity layout
            i_seg_spa = (i_seg_tri * TRITON_BLOCK_SIZE) // sparsity_block_size
            # Calculate the triton segment index within a block
            i_seg_tri_mod = i_seg_tri % (sparsity_block_size // TRITON_BLOCK_SIZE)

            # Load reverse sparsity index for current block
            rev_idx_spa_idx = (spa_bat * s_l_x_b_s +
                               spa_row * s_l_x_r_s +
                               i_seg_spa * s_l_x_c_s)
            rev_idx_spa_msk = (rev_idx_spa_idx < s_l_x_b * s_l_x_b_s + s_l_x_r * s_l_x_r_s + s_l_x_c * s_l_x_c_s)
            rev_idx_spa = tl.load(r_lut_x + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

            # If block is present commence operations
            if rev_idx_spa >= 0:
                blk_idx = ((rev_idx_spa * x_b_s) +
                           ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                           ((i_seg_tri_mod * TRITON_BLOCK_SIZE +
                             tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
                blk_msk = (blk_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
                blk = tl.load(x + blk_idx, mask=blk_msk)

                buf = buf + tl.reshape(tl.sum(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

                # DEBUG
                blk = tl.full(value=rev_idx_spa, shape=(TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), dtype=tl.float32)
                d_idx = (rev_idx_spa * x_b_s +
                         ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                         ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
                d_msk = (d_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
                tl.store(debug_tensor + d_idx, blk, d_msk)

                # d_idx = (rev_idx_spa * x_b_s +
                #          ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                #          ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.arange(0, 1)) * x_c_s)[None, :])
                # d_msk = (d_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
                # tl.store(debug_tensor + d_idx, buf, d_msk)

        o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 (tl.arange(0, 1))[None, :])
        o_msk = (o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
        tl.store(o + o_idx, buf, o_msk)
