import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.ops.blocksparse import BaseBlocksparse
from blksprs.ops.to_dense import BlocksparseToDense


class BlocksparseToSparse(BaseBlocksparse):

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor, sparsity_layout: Tensor) -> Tensor:
        self.validate_tensors(x)

        sparsity_lut = torch.nonzero(sparsity_layout)
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

        return _BlocksparseToSparse.apply(x,
                                          sparsity_layout, sparsity_lut,
                                          self.sparsity_block_size, n_sparse_blocks,
                                          self.triton_block_size, self.device)


class _BlocksparseToSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int, device: torch.device) -> Tensor:
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size), device=device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToSparse.kernel_blocksparse_to_sparse[triton_grid]
         (x, x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c,
          s_lut_c_s,
          output, o_b_s, o_r_s, o_c_s,
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

        return BlocksparseToDense(sparsity_block_size, device, triton_block_size)(grad_output,
                                                                                  sparsity_layout), None, None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_sparse(x,
                                     x_b, x_b_s, x_r, x_r_s, x_c: tl.constexpr, x_c_s,
                                     s_lut, s_lut_r, s_lut_r_s, s_lut_c, s_lut_c_s,
                                     o,
                                     o_b_s, o_r_s, o_c_s,
                                     sparsity_block_size,
                                     TRITON_BLOCK_SIZE: tl.constexpr):
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current output block consisting of its batch, row, and column index
        spa_bat_idx = (pid_blk * s_lut_r_s + 0 * s_lut_c_s)
        spa_bat_msk = (spa_bat_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_bat = tl.load(s_lut + spa_bat_idx, mask=spa_bat_msk)

        spa_row_idx = (pid_blk * s_lut_r_s + 1 * s_lut_c_s)
        spa_row_msk = (spa_row_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_row = tl.load(s_lut + spa_row_idx, mask=spa_row_msk)

        spa_col_idx = (pid_blk * s_lut_r_s + 2 * s_lut_c_s)
        spa_col_msk = (spa_col_idx < s_lut_r * s_lut_r_s + s_lut_c * s_lut_c_s)
        spa_col = tl.load(s_lut + spa_col_idx, mask=spa_col_msk)

        # Load block from dense tensor
        blk_d_idx = (spa_bat * x_b_s +
                     ((spa_row * sparsity_block_size + pid_row * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((spa_col * sparsity_block_size + pid_col * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_d_msk = (blk_d_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
        blk_d = tl.load(x + blk_d_idx, mask=blk_d_msk)

        # Store block in sparse tensor
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE) * o_c_s))[None, :])
        blk_o_msk = (blk_o_idx < (pid_blk + 1) * o_b_s)
        tl.store(o + blk_o_idx, blk_d, mask=blk_o_msk)
