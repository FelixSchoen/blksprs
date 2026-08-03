import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import as_base_tensor, build_layout_indices, build_packed_indices, \
    prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_positive_integer, validate_divisible, \
    ensure_contiguous, validate_dimension, validate_dtype_supported


@torch.amp.custom_fwd(device_type="cuda")
def split(x: BlksprsTensor, sparsity_layout: Tensor, partitions: int,
          dim: int, sparsity_block_size: int, layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Splits a compressed block-sparse tensor along its last dimension.

    For a layout with shape ``(B, R, C)`` and ``P`` partitions, the output layout has shape
    ``(B * P, R, C // P)``.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        partitions (int): The number of equal partitions.
        dim (int): The dimension along which to split the tensor. Only ``dim=2`` is supported.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The partitioned tensor in compressed form and its sparsity layout.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_supported(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    adjusted_dim = validate_dimension(dim)
    if adjusted_dim != 2:
        raise NotImplementedError("Currently only supports dim=2")
    validate_positive_integer(partitions, "partitions")
    validate_divisible(sparsity_layout.size(2), partitions, "Number of column blocks", "partitions")

    layout_cache = split_build_layout_cache(layout_cache, sparsity_layout, partitions)

    return BlksprsTensor.wrap(split_forward(
        x, layout_cache["sparsity_layout_output"], layout_cache["layout_indices"], layout_cache["packed_indices"],
        partitions, adjusted_dim, sparsity_block_size, layout_cache["n_sparse_blocks"])), layout_cache["sparsity_layout_output"]


@triton_op("blksprs::split_forward", mutates_args={})
def split_forward(x: Tensor, sparsity_layout_o: Tensor, layout_indices: Tensor, packed_indices: Tensor,
                  _: int, __: int, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, layout_indices, packed_indices, sparsity_block_size,
                                 n_sparse_blocks)


def split_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout = ctx.saved_tensors[0]
    num_partitions = ctx.num_partitions
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return merge(grad_output, sparsity_layout, num_partitions, dim,
                 sparsity_block_size)[0], None, None, None, None, None, None, None


def split_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor, partitions: int):
    layout_cache = prepare_layout_cache(
        layout_cache, "split", sparsity_layout, partitions)

    if "sparsity_layout_output" not in layout_cache:
        sparsity_layout_output = as_base_tensor(
            sparsity_layout
            .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                     sparsity_layout.size(2) // partitions)
            .permute(0, 2, 1, 3)
            .reshape(sparsity_layout.size(0) * partitions, sparsity_layout.size(1),
                     sparsity_layout.size(2) // partitions).contiguous())
        if sparsity_layout_output.data_ptr() == sparsity_layout.data_ptr():
            sparsity_layout_output = sparsity_layout_output.clone()
        layout_cache["sparsity_layout_output"] = sparsity_layout_output

    if "layout_indices" not in layout_cache:
        layout_indices = build_layout_indices(layout_cache["sparsity_layout_output"])
        layout_cache["layout_indices"] = layout_indices

    if "packed_indices" not in layout_cache:
        packed_indices = (build_packed_indices(sparsity_layout)
                                .reshape(sparsity_layout.size(0), sparsity_layout.size(1), partitions,
                                         sparsity_layout.size(2) // partitions)
                                .permute(0, 2, 1, 3).reshape(-1).contiguous())
        layout_cache["packed_indices"] = packed_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(layout_cache["sparsity_layout_output"].to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(layout_cache["sparsity_layout_output"], layout_cache["layout_indices"], layout_cache["packed_indices"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def split_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, num_partitions, dim, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.num_partitions = num_partitions
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


split_forward.register_autograd(split_wrapper_backward, setup_context=split_setup_context)


@torch.amp.custom_fwd(device_type="cuda")
def merge(x: BlksprsTensor, sparsity_layout: Tensor, partitions: int,
          dim: int, sparsity_block_size: int, layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Merges compressed block-sparse partitions along the last dimension.

    For a layout with shape ``(B, R, C)`` and ``P`` partitions, the output layout has shape
    ``(B // P, R, C * P)``.

    Args:
        x (BlksprsTensor): The compressed partitioned tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        partitions (int): The number of partitions to merge.
        dim (int): The dimension along which to merge the tensor. Only ``dim=2`` is supported.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The merged tensor in compressed form and its sparsity layout.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_supported(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    adjusted_dim = validate_dimension(dim)
    if adjusted_dim != 2:
        raise NotImplementedError("Currently only supports dim=2")
    validate_positive_integer(partitions, "partitions")
    validate_divisible(sparsity_layout.size(0), partitions, "Batch blocks", "partitions")

    layout_cache = merge_build_layout_cache(layout_cache, sparsity_layout, partitions)

    return BlksprsTensor.wrap(merge_forward(
        x, layout_cache["sparsity_layout_output"], layout_cache["layout_indices"], layout_cache["packed_indices"],
        partitions, adjusted_dim, sparsity_block_size, layout_cache["n_sparse_blocks"])), layout_cache["sparsity_layout_output"]


@triton_op("blksprs::merge_forward", mutates_args={})
def merge_forward(x: Tensor, sparsity_layout_o: Tensor, layout_indices: Tensor, packed_indices: Tensor,
                  _: int, __: int, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, layout_indices, packed_indices, sparsity_block_size,
                                 n_sparse_blocks)


def merge_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout = ctx.saved_tensors[0]
    num_partitions = ctx.num_partitions
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return split(grad_output, sparsity_layout, num_partitions, dim,
                 sparsity_block_size)[0], None, None, None, None, None, None, None


def merge_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor, partitions: int):
    layout_cache = prepare_layout_cache(
        layout_cache, "merge", sparsity_layout, partitions)

    if "sparsity_layout_output" not in layout_cache:
        sparsity_layout_output = as_base_tensor(
            sparsity_layout.reshape(sparsity_layout.size(0) // partitions, partitions,
                                    sparsity_layout.size(1), sparsity_layout.size(2))
            .permute(0, 2, 1, 3)
            .reshape(sparsity_layout.size(0) // partitions,
                     sparsity_layout.size(1),
                     sparsity_layout.size(2) * partitions).contiguous())
        if sparsity_layout_output.data_ptr() == sparsity_layout.data_ptr():
            sparsity_layout_output = sparsity_layout_output.clone()
        layout_cache["sparsity_layout_output"] = sparsity_layout_output

    if "layout_indices" not in layout_cache:
        layout_indices = build_layout_indices(layout_cache["sparsity_layout_output"])
        layout_cache["layout_indices"] = layout_indices

    if "packed_indices" not in layout_cache:
        packed_indices = (build_packed_indices(sparsity_layout)
                                .reshape(sparsity_layout.size(0) // partitions, partitions,
                                         sparsity_layout.size(1), sparsity_layout.size(2))
                                .permute(0, 2, 1, 3)
                                .reshape(sparsity_layout.size(0) // partitions,
                                         sparsity_layout.size(1), sparsity_layout.size(2) * partitions)
                                .reshape(-1).contiguous())
        layout_cache["packed_indices"] = packed_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(layout_cache["sparsity_layout_output"].to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(layout_cache["sparsity_layout_output"], layout_cache["layout_indices"], layout_cache["packed_indices"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def merge_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, num_partitions, dim, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.num_partitions = num_partitions
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


merge_forward.register_autograd(merge_wrapper_backward, setup_context=merge_setup_context)
