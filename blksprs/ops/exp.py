import torch
import triton
from triton import language as tl
from torch import Tensor

from blksprs.ops.tools import BaseBlocksparse


class BlocksparseExp(BaseBlocksparse):
    """Applies the element-wise exponential function to the input tensor.

    Returns a new tensor with the exponential of the elements of the input tensor.

    Note:
        This operation does not consider sparse blocks, i.e., these will not be set to ``e^0``.
        Consider this when converting back to dense tensors.
    """

    def __init__(self, sparsity_block_size: int, device: torch.device, triton_block_size: int = None) -> None:
        super().__init__(sparsity_block_size, device, triton_block_size=triton_block_size)

    def forward(self, x: Tensor) -> Tensor:
        self.validate_tensors(x)

        return _BlocksparseExp.apply(x, self.sparsity_block_size, self.triton_block_size, self.device)


class _BlocksparseExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_block_size: int, triton_block_size: int, device: torch.device):
        output = torch.zeros_like(x)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = x.stride()
        o_b, o_r, o_c = output.shape
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = BaseBlocksparse.get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseExp.kernel_blocksparse_exp[triton_grid]
         (x,
          x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
          output,
          o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
          triton_block_size))

        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        o = ctx.saved_tensors[0]

        grad_x = torch.mul(grad_output, o)

        return grad_x, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_exp(x,
                               x_b, x_b_s, x_r, x_r_s, x_c, x_c_s,
                               o,
                               o_b, o_b_s, o_r, o_r_s, o_c, o_c_s,
                               TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Load block
        blk_x_idx = ((pid_blk * x_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_x_msk = (blk_x_idx < x_b * x_b_s + x_r * x_r_s + x_c * x_c_s)
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

        # Compute exp
        buf = tl.exp(blk_x)

        # Store block
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
        blk_o_msk = (blk_o_idx < o_b * o_b_s + o_r * o_r_s + o_c * o_c_s)
        tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
