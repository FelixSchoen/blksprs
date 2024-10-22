import torch
import triton
from sympy.utilities.iterables import partitions
from torch import Tensor
from triton import language as tl

from blksprs.utils.tools import get_triton_block_size

from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_triton_block_size


def split(x: Tensor, sparsity_layout: Tensor, partitions: int,
          sparsity_block_size: int, triton_block_size: int = None) -> (Tensor, Tensor):
    """Splits a block-sparse tensor in compressed form along the last dimension into partitions.

    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        partitions (int): The number of partitions to split the block-sparse tensor into.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: The block-sparse tensor split into partitions in compressed form.
        Tensor: The sparsity layout of the output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_output = (sparsity_layout
                              .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                       sparsity_layout.size(2) // partitions)
                              .permute(0, 2, 1, 3)
                              .reshape(sparsity_layout.size(0) * partitions, sparsity_layout.size(1),
                                       sparsity_layout.size(2) // partitions).contiguous())

    sparsity_lut = torch.nonzero(sparsity_layout_output).contiguous()

    sparsity_layout_flat = sparsity_layout.reshape(-1)
    sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                             (sparsity_layout_flat == 1) -
                             (1 * (sparsity_layout_flat == 0)))
                            .reshape(sparsity_layout.size())
                            .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                     sparsity_layout.size(2) // partitions)
                            .permute(0, 2, 1, 3).reshape(-1).contiguous())

    n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, sparsity_lut, sparsity_reverse_lut)

    return _BlocksparseSplit.apply(x, sparsity_layout_output, sparsity_lut, sparsity_reverse_lut, partitions,
                                   sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_output


class _BlocksparseSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                num_partitions: int, sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        ctx.num_partitions = num_partitions

        return forward_reorder(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                               n_sparse_blocks, triton_block_size)

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.saved_tensors[0]
        num_partitions = ctx.num_partitions
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return merge(grad_output, sparsity_layout, num_partitions,
                     sparsity_block_size, triton_block_size)[0], None, None, None, None, None, None, None


def merge(x: Tensor, sparsity_layout: Tensor, partitions: int,
          sparsity_block_size: int, triton_block_size: int = None) -> (Tensor, Tensor):
    """Merges the specified partitions of a block-sparse tensor in compressed form along the last dimension.

    Args:
        x (Tensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        partitions (int): The number of partitions to be merged.
        sparsity_block_size (int): The size of the sparsity blocks.
        triton_block_size (int): The block size to use for the triton kernel (default ``None``).

    Returns:
        Tensor: The merged block-sparse tensor in compressed form.
        Tensor: The sparsity layout of the output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_triton_block_size(triton_block_size, sparsity_block_size)

    sparsity_layout_output = (sparsity_layout.reshape(sparsity_layout.size(0) // partitions, partitions,
                                                      sparsity_layout.size(1), sparsity_layout.size(2))
                              .permute(0, 2, 1, 3)
                              .reshape(sparsity_layout.size(0) // partitions,
                                       sparsity_layout.size(1), sparsity_layout.size(2) * partitions).contiguous())

    sparsity_lut = torch.nonzero(sparsity_layout_output).contiguous()

    sparsity_layout_flat = sparsity_layout.reshape(-1)
    sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                             (sparsity_layout_flat == 1) -
                             (1 * (sparsity_layout_flat == 0)))
                            .reshape(sparsity_layout.size(0) // partitions, partitions,
                                     sparsity_layout.size(1), sparsity_layout.size(2))
                            .permute(0, 2, 1, 3)
                            .reshape(sparsity_layout.size(0) // partitions,
                                     sparsity_layout.size(1), sparsity_layout.size(2) * partitions)
                            .reshape(-1).contiguous())

    n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, sparsity_lut, sparsity_reverse_lut)

    return _BlocksparseMerge.apply(x, sparsity_layout_output, sparsity_lut, sparsity_reverse_lut, partitions,
                                   sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_output


class _BlocksparseMerge(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                num_partitions: int, sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        ctx.num_partitions = num_partitions

        return forward_reorder(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                               n_sparse_blocks, triton_block_size)

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.saved_tensors[0]
        num_partitions = ctx.num_partitions
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return split(grad_output, sparsity_layout, num_partitions,
                     sparsity_block_size, triton_block_size)[0], None, None, None, None, None, None, None


def forward_reorder(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                    sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
    output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                         dtype=x.dtype, device=x.device)

    x_b, x_r, x_c = x.size()
    x_b_s, x_r_s, x_c_s = x.stride()
    s_l_b, s_l_r, s_l_c = sparsity_layout_o.size()
    s_l_b_s, s_l_r_s, s_l_c_s = sparsity_layout_o.stride()
    s_lut_r, s_lut_c = sparsity_lut.shape
    s_lut_r_s, s_lut_c_s = sparsity_lut.stride()
    o_b, o_r, o_c = output.size()
    o_b_s, o_r_s, o_c_s = output.stride()

    if triton_block_size is None:
        triton_block_size = get_triton_block_size(sparsity_block_size)

    triton_grid = lambda meta: [o_b,
                                triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

    (kernel_blocksparse_reorder[triton_grid]
     (x,
      x_b, x_b_s, x_r_s, x_c_s,
      s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
      sparsity_lut, s_lut_r, s_lut_r_s, s_lut_c_s,
      sparsity_reverse_lut,
      output,
      o_b, o_b_s,
      triton_block_size))

    # Save for backward pass
    ctx.save_for_backward(sparsity_layout_o)
    ctx.sparsity_block_size = sparsity_block_size
    ctx.triton_block_size = triton_block_size

    return output


@triton.jit
def kernel_blocksparse_reorder(x,
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

    # Get reverse sparsity index
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

    blk_o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_o_msk = (blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)
