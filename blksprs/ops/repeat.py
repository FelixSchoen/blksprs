import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward, flow_push_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import as_base_tensor, build_layout_indices, build_packed_indices, \
    prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_layout, validate_sparsity_block_size, validate_shape, \
    validate_non_negative_integer, validate_non_negative_integer_tuple, ensure_contiguous, validate_dtype_supported


@torch.amp.custom_fwd(device_type="cuda")
def repeat(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
           sparsity_block_size: int, sparsity_layout_output: Tensor | None = None, layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Repeats a compressed block-sparse tensor along each dimension.

    ``repeats`` contains the non-negative repeat count for each of the three dimensions. A zero count produces an empty
    output along the corresponding dimension.

    Note:
        When ``sparsity_layout_output`` is provided, only blocks active in both the repeated input layout and the target
        layout are retained.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout_x (Tensor): The sparsity layout of ``x``.
        repeats (tuple[int, int, int]): The non-negative repeat counts for the three dimensions.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor, optional): The target sparsity layout (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The repeated tensor in compressed form and its sparsity layout.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_supported(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_non_negative_integer_tuple(repeats, 3, "repeats")
    if sparsity_layout_output is not None:
        sparsity_layout_output = ensure_contiguous(sparsity_layout_output)
        validate_sparsity_layout(sparsity_layout_output)
        validate_contiguous(sparsity_layout_output)
        validate_device(sparsity_layout_output, x)
        validate_shape(
            sparsity_layout_output,
            (
                sparsity_layout_x.size(0) * repeats[0],
                sparsity_layout_x.size(1) * repeats[1],
                sparsity_layout_x.size(2) * repeats[2],
            ),
            "Output sparsity layout",
        )

    layout_cache = repeat_build_layout_cache(layout_cache, sparsity_layout_x, repeats, sparsity_layout_output)

    return BlksprsTensor.wrap(repeat_forward(
        x, sparsity_layout_x, layout_cache["sparsity_layout_o"], layout_cache["layout_indices"],
        layout_cache["packed_indices"], sparsity_block_size, layout_cache["n_sparse_blocks"])), layout_cache["sparsity_layout_o"]


@torch.amp.custom_fwd(device_type="cuda")
def repeat_interleave(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: int,
                      sparsity_block_size: int, sparsity_layout_output: Tensor | None = None, layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Repeats and interleaves compressed matrices along the batch dimension.

    Each matrix is repeated ``repeats`` times and placed consecutively in the output. A zero count produces an empty
    output.

    Note:
        When ``sparsity_layout_output`` is provided, only blocks active in both the repeated input layout and the target
        layout are retained.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout_x (Tensor): The sparsity layout of ``x``.
        repeats (int): The non-negative number of times to repeat each matrix.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor, optional): The target sparsity layout (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The repeated tensor in compressed form and its sparsity layout.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_supported(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)
    validate_non_negative_integer(repeats, "repeats")
    if sparsity_layout_output is not None:
        sparsity_layout_output = ensure_contiguous(sparsity_layout_output)
        validate_sparsity_layout(sparsity_layout_output)
        validate_contiguous(sparsity_layout_output)
        validate_device(sparsity_layout_output, x)
        validate_shape(
            sparsity_layout_output,
            (
                sparsity_layout_x.size(0) * repeats,
                sparsity_layout_x.size(1),
                sparsity_layout_x.size(2),
            ),
            "Output sparsity layout",
        )

    layout_cache = repeat_interleave_build_layout_cache(layout_cache, sparsity_layout_x, repeats, sparsity_layout_output)

    return BlksprsTensor.wrap(repeat_forward(
        x, sparsity_layout_x, layout_cache["sparsity_layout_o"], layout_cache["layout_indices"],
        layout_cache["packed_indices"], sparsity_block_size, layout_cache["n_sparse_blocks"])), layout_cache["sparsity_layout_o"]


@triton_op("blksprs::repeat_forward", mutates_args={})
def repeat_forward(x: Tensor, _: Tensor, sparsity_layout_o: Tensor, layout_indices: Tensor,
                   packed_indices: Tensor,
                   sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, layout_indices, packed_indices, sparsity_block_size,
                                 n_sparse_blocks)


def repeat_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout_x, sparsity_layout_o, layout_indices, packed_indices = ctx.saved_tensors
    sparsity_block_size = ctx.sparsity_block_size
    n_sparse_blocks = torch.sum(sparsity_layout_x.to(torch.int)).item()

    return flow_push_forward(grad_output, sparsity_layout_o, layout_indices,
                             packed_indices, sparsity_block_size,
                             n_sparse_blocks), None, None, None, None, None, None


def repeat_build_layout_cache(layout_cache: dict | None, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
                     sparsity_layout_output: Tensor | None):
    layout_cache = prepare_layout_cache(
        layout_cache, "repeat", sparsity_layout_x, repeats, sparsity_layout_output)

    if "sparsity_layout_o" not in layout_cache:
        sparsity_layout_o = as_base_tensor(
            sparsity_layout_x.repeat(repeats[0], repeats[1], repeats[2]))
        layout_cache["sparsity_layout_o"] = sparsity_layout_o

    if sparsity_layout_output is not None:
        sparsity_layout_o = as_base_tensor(
            torch.logical_and(layout_cache["sparsity_layout_o"], sparsity_layout_output))
        layout_cache["sparsity_layout_o"] = sparsity_layout_o

    if "layout_indices" not in layout_cache:
        layout_indices = build_layout_indices(layout_cache["sparsity_layout_o"])
        layout_cache["layout_indices"] = layout_indices

    if "packed_indices" not in layout_cache:
        packed_indices = (build_packed_indices(sparsity_layout_x)
                                .reshape(sparsity_layout_x.size())
                                .repeat(repeats[0], repeats[1], repeats[2])
                                .reshape(-1).contiguous())
        layout_cache["packed_indices"] = packed_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(layout_cache["sparsity_layout_o"].to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(layout_cache["sparsity_layout_o"], layout_cache["layout_indices"], layout_cache["packed_indices"])

    return finalize_layout_cache(layout_cache)


def repeat_interleave_build_layout_cache(layout_cache: dict | None, sparsity_layout_x: Tensor, repeats: int,
                                sparsity_layout_output: Tensor | None):
    layout_cache = prepare_layout_cache(
        layout_cache, "repeat_interleave", sparsity_layout_x, repeats, sparsity_layout_output)

    if "sparsity_layout_o" not in layout_cache:
        sparsity_layout_o = as_base_tensor(
            torch.repeat_interleave(sparsity_layout_x, repeats, dim=0).contiguous())
        layout_cache["sparsity_layout_o"] = sparsity_layout_o

    if sparsity_layout_output is not None:
        sparsity_layout_o = as_base_tensor(
            torch.logical_and(layout_cache["sparsity_layout_o"], sparsity_layout_output))
        layout_cache["sparsity_layout_o"] = sparsity_layout_o

    if "layout_indices" not in layout_cache:
        layout_indices = build_layout_indices(layout_cache["sparsity_layout_o"])
        layout_cache["layout_indices"] = layout_indices

    if "packed_indices" not in layout_cache:
        packed_indices = (build_packed_indices(sparsity_layout_x)
                                .reshape(sparsity_layout_x.size())
                                .repeat_interleave(repeats, dim=0)
                                .reshape(-1).contiguous())
        layout_cache["packed_indices"] = packed_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(layout_cache["sparsity_layout_o"].to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(layout_cache["sparsity_layout_o"], layout_cache["layout_indices"], layout_cache["packed_indices"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def repeat_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, sparsity_layout_o, layout_indices, packed_indices, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_x, sparsity_layout_o, layout_indices, packed_indices)
    ctx.sparsity_block_size = sparsity_block_size


repeat_forward.register_autograd(repeat_wrapper_backward, setup_context=repeat_setup_context)
