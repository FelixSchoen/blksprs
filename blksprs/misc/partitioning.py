import torch
from torch import Tensor

from blksprs.ops.repeat import forward_flow

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
                            .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                     sparsity_layout.size(2) // partitions)
                            .permute(0, 2, 1, 3).reshape(-1).contiguous())

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, sparsity_lut, sparsity_reverse_lut)

    return _BlocksparseSplit.apply(x, sparsity_layout_output, sparsity_lut, sparsity_reverse_lut, partitions,
                                   sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_output


class _BlocksparseSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                num_partitions: int, sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        ctx.save_for_backward(sparsity_layout_o)
        ctx.num_partitions = num_partitions

        return forward_flow(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
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

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, sparsity_lut, sparsity_reverse_lut)

    return _BlocksparseMerge.apply(x, sparsity_layout_output, sparsity_lut, sparsity_reverse_lut, partitions,
                                   sparsity_block_size, n_sparse_blocks, triton_block_size), sparsity_layout_output


class _BlocksparseMerge(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                num_partitions: int, sparsity_block_size: int, n_sparse_blocks: int, triton_block_size: int) -> Tensor:
        ctx.save_for_backward(sparsity_layout_o)
        ctx.num_partitions = num_partitions

        return forward_flow(ctx, x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                            n_sparse_blocks, triton_block_size)

    @staticmethod
    def backward(ctx, grad_output):
        sparsity_layout = ctx.saved_tensors[0]
        num_partitions = ctx.num_partitions
        sparsity_block_size = ctx.sparsity_block_size
        triton_block_size = ctx.triton_block_size

        return split(grad_output, sparsity_layout, num_partitions,
                     sparsity_block_size, triton_block_size)[0], None, None, None, None, None, None, None


