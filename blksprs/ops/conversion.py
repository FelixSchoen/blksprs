import torch
import triton
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_dtype_float, validate_device


def to_dense(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int, fill_value: float = 0,
             triton_block_size: int = None) -> Tensor:
    """Converts a blocksparse tensor to a dense tensor based on the given sparsity layout.

    The ``fill_value`` is used to fill the resulting dense tensor with a specific value (default ``0``) where the
     blocksparse tensor is not present.

    """
    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout)
    validate_dtype_float(x)
    validate_device(x)

    sparsity_layout_flat = sparsity_layout.reshape(-1)
    sparsity_reverse_lut = ((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                            (sparsity_layout_flat == 1) -
                            (1 * (sparsity_layout_flat == 0)))

    validate_contiguous(sparsity_reverse_lut)

    return _BlocksparseToDense.apply(x,
                                     sparsity_layout, sparsity_reverse_lut,
                                     sparsity_block_size, fill_value,
                                     triton_block_size)


class _BlocksparseToDense(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_reverse_lut: Tensor,
                sparsity_block_size: int, fill_value: float,
                triton_block_size: int) -> Tensor:
        output = torch.full(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                  sparsity_layout.size(2) * sparsity_block_size), fill_value=fill_value,
                            dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = x.stride()
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToDense.kernel_blocksparse_to_dense[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
          sparsity_reverse_lut,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.sparsity_layout = sparsity_layout
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.sparsity_layout
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return to_sparse(grad_output, sparsity_layout, sparsity_block_size, triton_block_size), None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_dense(x,
                                    x_b, x_b_s, x_r_s, x_c_s,
                                    s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                                    sparsity_reverse_lut,
                                    o,
                                    o_b, o_b_s, o_r_s, o_c_s,
                                    sparsity_block_size,
                                    TRITON_BLOCK_SIZE: tl.constexpr) -> None:
        # Get triton block indices
        pid_blk = tl.program_id(axis=0)
        pid_row = tl.program_id(axis=1)
        pid_col = tl.program_id(axis=2)

        # Get sparsity index of current block
        spa_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
        spa_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size

        # Get reverse sparsity index for current block
        rev_idx_spa_idx = (pid_blk * s_l_b_s + spa_row * s_l_r_s + spa_col * s_l_c_s)
        rev_idx_spa_msk = (rev_idx_spa_idx < s_l_b * s_l_b_s)
        rev_idx_spa = tl.load(sparsity_reverse_lut + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

        # If block is present commence operations
        if rev_idx_spa >= 0:
            blk_idx = (rev_idx_spa * x_b_s +
                       (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                       (((pid_col % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                         tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
            blk_msk = (blk_idx < x_b * x_b_s)
            blk = tl.load(x + blk_idx, mask=blk_msk)

            o_idx = (pid_blk * o_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
            o_msk = (o_idx < o_b * o_b_s)
            tl.store(o + o_idx, blk, o_msk)


def to_sparse(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int, triton_block_size: int = None) -> Tensor:
    """Converts a dense tensor to a blocksparse tensor based on the given sparsity layout.

    """
    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout)
    validate_dtype_float(x)
    validate_device(x)

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()
    n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

    validate_contiguous(sparsity_lut)

    return _BlocksparseToSparse.apply(x,
                                      sparsity_layout, sparsity_lut,
                                      sparsity_block_size, n_sparse_blocks,
                                      triton_block_size)


class _BlocksparseToSparse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor,
                sparsity_layout: Tensor, sparsity_lut: Tensor,
                sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size), device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = x.stride()
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = output.stride()
        s_lut_r, s_lut_c = sparsity_lut.size()
        s_lut_r_s, s_lut_c_s = sparsity_lut.stride()

        if triton_block_size is None:
            triton_block_size = get_triton_block_size(sparsity_block_size)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (_BlocksparseToSparse.kernel_blocksparse_to_sparse[triton_grid]
         (x, x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
          output, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          triton_block_size))

        ctx.sparsity_layout = sparsity_layout
        ctx.sparsity_block_size = sparsity_block_size
        ctx.triton_block_size = triton_block_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.sparsity_layout
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return to_dense(grad_output, sparsity_layout, sparsity_block_size,
                        triton_block_size=triton_block_size), None, None, None, None, None

    @staticmethod
    @triton.jit
    def kernel_blocksparse_to_sparse(x,
                                     x_b, x_b_s, x_r_s, x_c_s,
                                     s_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
                                     o,
                                     o_b_s, o_r_s, o_c_s,
                                     sparsity_block_size,
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

        # Load block from dense tensor
        blk_d_idx = (spa_bat * x_b_s +
                     ((spa_row * sparsity_block_size + pid_row * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                     ((spa_col * sparsity_block_size + pid_col * TRITON_BLOCK_SIZE +
                       tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_d_msk = (blk_d_idx < x_b * x_b_s)
        blk_d = tl.load(x + blk_d_idx, mask=blk_d_msk)

        # Store block in sparse tensor
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE) * o_c_s))[None, :])
        blk_o_msk = (blk_o_idx < (pid_blk + 1) * o_b_s)
        tl.store(o + blk_o_idx, blk_d, mask=blk_o_msk)
