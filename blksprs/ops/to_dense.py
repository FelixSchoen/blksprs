import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.blocksparse import BaseBlocksparse
from blksprs.ops.to_sparse import BlocksparseToSparse


class BlocksparseToDense(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor, fill_value: int = 0) -> Tensor:
        self.validate_tensors(x)

        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                (sparsity_layout_flat == 1) -
                                (1 * (sparsity_layout_flat == 0)))

        return _BlocksparseToDense.apply(x,
                                         sparsity_layout, sparsity_reverse_lut,
                                         self.sparsity_block_size, fill_value,
                                         self.triton_block_size, self.device)


class _BlocksparseToDense(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, fill_value: int,
                triton_block_size: int, device: torch.device) -> Tensor:
        output = torch.full(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                  sparsity_layout.size(2) * sparsity_block_size), fill_value=fill_value,
                            dtype=x.dtype, device=device)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToDense.kernel_blocksparse_to_dense[triton_grid]
         (x,
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.sparsity_layout = sparsity_layout
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size
        ctx.device = device

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.sparsity_layout
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size
        device = ctx.device

        return BlocksparseToSparse(sparsity_block_size, device, triton_block_size)(grad_output,
                                                                                   sparsity_layout), None, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_dense(x,
                                    x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                                    s_l_b, s_l_b_s, s_l_r, s_l_r_s, s_l_c, s_l_c_s,
                                    sparsity_reverse_lut,
                                    o,
                                    o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                                    sparsity_block_size,
                                    TRITON_BLOCK_SIZE: tl.constexpr):
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current block
        spa_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
        spa_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size

        # Get reverse sparsity index for current block
        rev_idx_spa_idx = (pid_blk * s_l_b_s + spa_row * s_l_r_s + spa_col * s_l_c_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_b * s_l_b_s + s_l_r * s_l_r_s + s_l_c * s_l_c_s)
        rev_idx_spa = tl.load(sparsity_reverse_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        # If block is present commence operations
        if rev_idx_spa >= 0:
            blk_idx = (rev_idx_spa * x_b_s +
                       (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                       (((pid_col % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
            blk_msk = (blk_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
            blk = tl.load(x + blk_idx, mask=blk_msk)

            o_idx = (pid_blk * o_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
            o_msk = (o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
            tl.store(o + o_idx, blk, o_msk)
