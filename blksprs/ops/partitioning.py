import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def split(x: BlksprsTensor, sparsity_layout: Tensor, partitions: int,
          dim: int, sparsity_block_size: int, lut: dict = None) -> (
        BlksprsTensor, Tensor):
    """Splits a block-sparse tensor in compressed form along the last dimension into partitions.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        partitions (int): The number of partitions to split the block-sparse tensor into.
        dim (int): The dimension along which to split the tensor. Currently only supports dim=2.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The block-sparse tensor split into partitions in compressed form.
        Tensor: The sparsity layout of the output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    adjusted_dim = dim % 3
    if adjusted_dim != 2:
        raise NotImplementedError("Currently only supports dim=2")

    lut = split_build_lut(lut, sparsity_layout, partitions)

    return BlksprsTensor(split_forward(
        x, lut["sparsity_layout_output"], lut["sparsity_lut"], lut["sparsity_reverse_lut"],
        partitions, adjusted_dim, sparsity_block_size, lut["n_sparse_blocks"])), lut["sparsity_layout_output"]


@triton_op("blksprs::split_forward", mutates_args={})
def split_forward(x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                  _: int, __: int, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                                 n_sparse_blocks)


def split_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    num_partitions = ctx.num_partitions
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return merge(grad_output, sparsity_layout, num_partitions, dim,
                 sparsity_block_size)[0], None, None, None, None, None, None, None


def split_build_lut(lut: dict, sparsity_layout: Tensor, partitions: int):
    if lut is None:
        lut = dict()

    if "sparsity_layout_output" not in lut:
        sparsity_layout_output = (sparsity_layout
                                  .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                           sparsity_layout.size(2) // partitions)
                                  .permute(0, 2, 1, 3)
                                  .reshape(sparsity_layout.size(0) * partitions, sparsity_layout.size(1),
                                           sparsity_layout.size(2) // partitions).contiguous())
        lut["sparsity_layout_output"] = sparsity_layout_output

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(lut["sparsity_layout_output"]).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut" not in lut:
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                 (sparsity_layout_flat == 1) -
                                 (1 * (sparsity_layout_flat == 0)))
                                .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                         sparsity_layout.size(2) // partitions)
                                .permute(0, 2, 1, 3).reshape(-1).contiguous())
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(lut["sparsity_layout_output"].to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(lut["sparsity_layout_output"], lut["sparsity_lut"], lut["sparsity_reverse_lut"])

    return lut


# noinspection PyUnusedLocal
def split_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, num_partitions, dim, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.num_partitions = num_partitions
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


split_forward.register_autograd(split_wrapper_backward, setup_context=split_setup_context)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def merge(x: BlksprsTensor, sparsity_layout: Tensor, partitions: int,
          dim: int, sparsity_block_size: int, lut: dict = None) -> (
        BlksprsTensor, Tensor):
    """Merges the specified partitions of a block-sparse tensor in compressed form along the last dimension.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        partitions (int): The number of partitions to be merged.
        dim (int): The dimension along which to merge the tensor. Currently only supports dim=2.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The merged block-sparse tensor in compressed form.
        Tensor: The sparsity layout of the output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    adjusted_dim = dim % 3
    if adjusted_dim != 2:
        raise NotImplementedError("Currently only supports dim=2")

    lut = merge_build_lut(lut, sparsity_layout, partitions)

    return BlksprsTensor(merge_forward(
        x, lut["sparsity_layout_output"], lut["sparsity_lut"], lut["sparsity_reverse_lut"],
        partitions, adjusted_dim, sparsity_block_size, lut["n_sparse_blocks"])), lut["sparsity_layout_output"]


@triton_op("blksprs::merge_forward", mutates_args={})
def merge_forward(x: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                  _: int, __: int, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                                 n_sparse_blocks)


def merge_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    num_partitions = ctx.num_partitions
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return split(grad_output, sparsity_layout, num_partitions, dim,
                 sparsity_block_size)[0], None, None, None, None, None, None, None


def merge_build_lut(lut: dict, sparsity_layout: Tensor, partitions: int):
    if lut is None:
        lut = dict()

    if "sparsity_layout_output" not in lut:
        sparsity_layout_output = (sparsity_layout.reshape(sparsity_layout.size(0) // partitions, partitions,
                                                          sparsity_layout.size(1), sparsity_layout.size(2))
                                  .permute(0, 2, 1, 3)
                                  .reshape(sparsity_layout.size(0) // partitions,
                                           sparsity_layout.size(1),
                                           sparsity_layout.size(2) * partitions).contiguous())
        lut["sparsity_layout_output"] = sparsity_layout_output

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(lut["sparsity_layout_output"]).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut" not in lut:
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
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(lut["sparsity_layout_output"].to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(lut["sparsity_layout_output"], lut["sparsity_lut"], lut["sparsity_reverse_lut"])

    return lut


# noinspection PyUnusedLocal
def merge_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, num_partitions, dim, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.num_partitions = num_partitions
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


merge_forward.register_autograd(merge_wrapper_backward, setup_context=merge_setup_context)
