import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.tools import BaseBlocksparse
from blksprs.ops.conversion import BlocksparseToDense, BlocksparseToSparse


class BlocksparseSoftmax(BaseBlocksparse):
    # TODO At the moment uses standard softmax instead of blocksparse improvements

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

        self.blksprs_to_dense = BlocksparseToDense(sparsity_block_size, device)
        self.blksprs_to_sparse = BlocksparseToSparse(sparsity_block_size, device)

    def forward(self, x: Tensor, sparsity_layout: Tensor,
                fill_value_output: float = float(0), fill_value_softmax: float = float("-inf")) -> Tensor:
        self.validate_tensors(x)

        sparsity_lut = torch.nonzero(sparsity_layout)
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                (sparsity_layout_flat == 1) -
                                (1 * (sparsity_layout_flat == 0)))

        return _BlocksparseSoftmax.apply(x,
                                         sparsity_layout, sparsity_lut, sparsity_reverse_lut,
                                         self.sparsity_block_size, fill_value_output, fill_value_softmax,
                                         self.triton_block_size, self.device)

        x_dense = self.blksprs_to_dense(x, sparsity_layout, fill_value=fill_value_softmax)
        x_softmax = torch.softmax(x_dense, dim=-1)
        x_sparse = self.blksprs_to_sparse(x_softmax, sparsity_layout)

        return x_sparse


class _BlocksparseSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, fill_value_output: float, fill_value_softmax: int,
                triton_block_size: int, device: torch.device) -> Tensor:
        output = torch.full_like(x, fill_value_output)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseSoftmax.kernel_blocksparse_softmax[triton_grid]
         (x,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
          sparsity_lut,
          s_lut_r, s_lut_r_s, s_lut_c, s_lut_c_s,
          sparsity_reverse_lut,
          fill_value_softmax,
          sparsity_block_size,
          triton_block_size))

        return output

    @staticmethod
    @triton.jit
    def kernel_blocksparse_softmax(x,
                                   o,
                                   o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                                   s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
                                   s_lut,
                                   s_lut_r, s_lut_r_s,
                                   s_lut_c, s_lut_c_s,
                                   r_lut,
                                   fill_value: tl.float32,
                                   sparsity_block_size,
                                   TRITON_BLOCK_SIZE: tl.constexpr):
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)

        # Get position of current sparsity block consisting of its batch and row index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        # Initialise placeholder and maximum buffer
        plc_hld = tl.full(shape=(TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), value=fill_value, dtype=tl.float32)
        buf_max = tl.zeros(shape=(TRITON_BLOCK_SIZE, 1), dtype=tl.float32)

        # Slide over triton block sized segments of input tensor
        for i_seg_tri in range(0, tl.cdiv(s_l_c * sparsity_block_size, TRITON_BLOCK_SIZE)):
            # Convert to segment index of sparsity layout
            i_seg_spa = (i_seg_tri * TRITON_BLOCK_SIZE) // sparsity_block_size
            # Calculate the triton segment index within a block
            i_seg_tri_mod = i_seg_tri % (sparsity_block_size // TRITON_BLOCK_SIZE)

            # Load reverse sparsity index for current block
            rev_idx_spa_idx = (spa_bat * s_l_b_s +
                               spa_row * s_l_r_s +
                               i_seg_spa * s_l_c_s)
            rev_idx_spa_msk = (rev_idx_spa_idx < s_l_b * s_l_b_s + s_l_r * s_l_r_s + s_l_c * s_l_c_s)
            rev_idx_spa = tl.load(r_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

            # If block is present commence operations
            if rev_idx_spa >= 0:
                blk_idx = (rev_idx_spa * o_b_s +
                           (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                             tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                           ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
                blk_msk = (blk_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
                blk = tl.load(x + blk_idx, mask=blk_msk)

                buf_max = tl.maximum(buf_max, tl.reshape(tl.max(blk, axis=-1), (TRITON_BLOCK_SIZE, 1)))
            # Else use placeholder block
            else:
                blk = plc_hld

        # DEBUG
        o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((0 * TRITON_BLOCK_SIZE + tl.arange(0, 1)) * o_c_s)[None, :])
        o_msk = (o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
        tl.store(o + o_idx, buf_max, o_msk)

        # o_idx = (pid_blk * o_b_s +
        #          ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
        #          ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        # o_msk = (o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
        # tl.store(o + o_idx, blk, o_msk)
