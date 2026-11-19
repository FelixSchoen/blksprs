from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

import blksprs as bs
from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul_fast
from blksprs.ops.conversion import to_sparse
from blksprs.ops.matmul import matmul
from blksprs.ops.repeat import repeat
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.public_op import public_op
from blksprs.utils.tools import prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import ensure_contiguous, validate_contiguous, validate_device, \
    validate_dimensions, validate_dtype_float, validate_dtype_supported, validate_shape, validate_sparsity, \
    validate_sparsity_block_size


def _validate_linear_dtypes(x: Tensor, weight: Tensor, bias: Tensor | None) -> None:
    tensors = (x, weight) if bias is None else (x, weight, bias)
    if torch.is_autocast_enabled():
        for tensor in tensors:
            validate_dtype_float(tensor)
    else:
        validate_dtype_float(*tensors)


@public_op
def apply_torch_linear(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                       linear: nn.Linear, bias: nn.Parameter | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Applies a PyTorch linear layer to a block-sparse tensor.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        linear (nn.Linear): The linear layer to apply.
        bias (nn.Parameter, optional): An explicit bias overriding ``linear.bias`` (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The projected tensor in compressed form and its sparsity layout.

    """
    # Extract weight; bias uses the explicit override if provided, otherwise falls back to linear.bias
    w = linear.weight
    b = bias if bias is not None else linear.bias
    _validate_linear_dtypes(x, w, b)
    validate_sparsity_block_size(sparsity_block_size, w)

    # Convert w to block-sparse representation
    sparsity_layout_w_t = torch.ones(size=(sparsity_layout.size(0), w.size(1) // sparsity_block_size,
                                           w.size(0) // sparsity_block_size), dtype=torch.bool, device=x.device)
    w_t_bs = to_sparse(w.transpose(-1, -2).unsqueeze(0).repeat(sparsity_layout.size(0), 1, 1),
                       sparsity_layout_w_t, sparsity_block_size)

    # Compute output sparsity layout
    # A bias can make every output block non-zero, including when the input has no feature blocks.
    if b is not None:
        sparsity_layout_output = torch.ones(
            size=(sparsity_layout.size(0), sparsity_layout.size(1), w.size(0) // sparsity_block_size),
            dtype=torch.bool,
            device=x.device,
        )
    else:
        sparsity_layout_output = build_sparsity_layout_matmul_fast(sparsity_layout, sparsity_layout_w_t)

    # Apply weights
    xw = matmul(x, sparsity_layout, BlksprsTensor.wrap(w_t_bs.to(x.dtype)), sparsity_layout_w_t, sparsity_layout_output,
                sparsity_block_size)
    interim: Tensor = xw

    # Apply bias
    if b is not None:
        b_slice = b.unsqueeze(0).unsqueeze(0).repeat(1, sparsity_block_size, 1)
        validate_sparsity_block_size(sparsity_block_size, b_slice)
        sparsity_layout_b_slice = torch.ones(size=(1, b_slice.size(1) // sparsity_block_size,
                                                   b_slice.size(2) // sparsity_block_size), dtype=torch.bool,
                                             device=x.device)
        b_slice_bs = to_sparse(b_slice, sparsity_layout_b_slice, sparsity_block_size)
        b_bs, sparsity_layout_b = repeat(b_slice_bs, sparsity_layout_b_slice,
                                         (sparsity_layout.size(0), sparsity_layout_output.size(1), 1),
                                         sparsity_block_size,
                                         sparsity_layout_output=sparsity_layout_output)
        interim = interim + b_bs.to(interim.dtype)

    return cast(BlksprsTensor, interim), sparsity_layout_output


@public_op
def apply_torch_linear_cached(x: BlksprsTensor,
                              sparsity_layout: Tensor,
                              sparsity_block_size: int,
                              linear: nn.Linear,
                              bias: nn.Parameter | None = None,
                              layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Applies a PyTorch linear layer while caching its layout metadata.

    The cache holds only non-differentiable metadata derived from ``sparsity_layout``, the sparsity block size, and the
    parameter shapes. The compressed weight and bias are derived from the current parameters on every call, so that
    optimizer steps, reloaded parameters, gradient accumulation, and activation checkpointing always use the current
    values and a fresh autograd graph. A warm cache avoids host synchronisation when validation is disabled.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        linear (nn.Linear): The linear layer to apply.
        bias (nn.Parameter, optional): An explicit bias overriding ``linear.bias`` (default ``None``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The projected tensor in compressed form and its sparsity layout.

    """
    w = linear.weight
    b = bias if bias is not None else linear.bias
    _validate_linear_dtypes(x, w, b)
    validate_sparsity_block_size(sparsity_block_size, w)
    if b is not None:
        validate_shape(b, (w.size(0),), "Bias")

    n_batches, n_row_blocks = sparsity_layout.size(0), sparsity_layout.size(1)
    n_blocks_out = w.size(0) // sparsity_block_size

    layout_cache = apply_torch_linear_build_layout_cache(
        layout_cache, sparsity_layout, sparsity_block_size, x.device, tuple(w.shape), b is not None)

    interim: Tensor = matmul(x, layout_cache["sparsity_layout_x"],
                             BlksprsTensor.wrap(_build_linear_weight_blocks(w, sparsity_block_size, x.dtype)),
                             layout_cache["sparsity_layout_w_t"],
                             layout_cache["sparsity_layout_o"],
                             sparsity_block_size, layout_cache=layout_cache["matmul_layout_cache"])

    if b is not None:
        # Every output block is active, so the bias broadcasts over the stacked row blocks.
        b_blocks = b.to(interim.dtype).reshape(n_blocks_out, 1, sparsity_block_size)
        interim = (interim.view(n_batches * n_row_blocks, n_blocks_out, sparsity_block_size, sparsity_block_size)
                   + b_blocks).flatten(0, 1)

    output_layout = layout_cache["sparsity_layout_output"]
    finalize_layout_cache(layout_cache)
    return cast(BlksprsTensor, interim), output_layout


@torch.compiler.disable
def apply_torch_linear_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor, sparsity_block_size: int,
                                          device: torch.device, weight_shape: tuple[int, ...],
                                          has_bias: bool) -> dict:
    """Prepares the layout metadata of :func:`apply_torch_linear_cached`.

    The caller finalises the cache once the nested matrix multiplication has populated its own cache.

    """
    layout_cache = prepare_layout_cache(
        layout_cache,
        "apply_torch_linear",
        sparsity_layout,
        sparsity_block_size,
        device,
        weight_shape,
        has_bias,
    )

    if "sparsity_layout_output" not in layout_cache:
        n_batches, n_row_blocks = sparsity_layout.size(0), sparsity_layout.size(1)
        n_blocks_in = weight_shape[1] // sparsity_block_size
        n_blocks_out = weight_shape[0] // sparsity_block_size

        # Stacking the batches along the rows preserves the order of the compressed blocks, so that a single copy of
        # the weight serves all batches.
        sparsity_layout_x = sparsity_layout.reshape(1, n_batches * n_row_blocks, sparsity_layout.size(2))
        sparsity_layout_w_t = torch.ones(size=(1, n_blocks_in, n_blocks_out), dtype=torch.bool, device=device)

        # A bias can make every output block non-zero, including when the input has no feature blocks.
        if has_bias:
            sparsity_layout_o = torch.ones(size=(1, n_batches * n_row_blocks, n_blocks_out), dtype=torch.bool,
                                           device=device)
        else:
            sparsity_layout_o = build_sparsity_layout_matmul_fast(sparsity_layout_x, sparsity_layout_w_t)

        layout_cache["sparsity_layout_x"] = sparsity_layout_x
        layout_cache["sparsity_layout_w_t"] = sparsity_layout_w_t
        layout_cache["sparsity_layout_o"] = sparsity_layout_o
        layout_cache["sparsity_layout_output"] = sparsity_layout_o.reshape(n_batches, n_row_blocks, n_blocks_out)

    layout_cache.setdefault("matmul_layout_cache", dict())

    return layout_cache


def _build_linear_weight_blocks(w: Tensor, sparsity_block_size: int, dtype: torch.dtype) -> Tensor:
    # Compresses the transposed weight under a full single-batch layout. The copy keeps the autograd graph of the
    # current parameter and is never cached, so it cannot become stale.
    n_blocks_in = w.size(1) // sparsity_block_size
    n_blocks_out = w.size(0) // sparsity_block_size

    w_t_blocks = w.to(dtype).t().reshape(n_blocks_in, sparsity_block_size, n_blocks_out, sparsity_block_size)

    return w_t_blocks.permute(0, 2, 1, 3).contiguous().view(n_blocks_in * n_blocks_out, sparsity_block_size,
                                                            sparsity_block_size)


@public_op
def apply_torch_normalisation(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                              normalisation: nn.Module) -> BlksprsTensor:
    """Applies a row-wise PyTorch normalisation module to a block-sparse tensor.

    Empty rows are removed before applying the module so that compressed rows remain contiguous.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        normalisation (nn.Module): The normalisation module to apply.

    Returns:
        BlksprsTensor: The normalised tensor in compressed form.

    """
    return apply_function_applicable_row_wise(x, sparsity_layout, sparsity_block_size, normalisation)


@public_op
def apply_torch_dropout(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                        dropout: nn.Dropout) -> BlksprsTensor:
    """Applies a PyTorch dropout module to a block-sparse tensor.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        dropout (nn.Dropout): The dropout module to apply.

    Returns:
        BlksprsTensor: The result in compressed form.

    """
    return apply_function_applicable_row_wise(x, sparsity_layout, sparsity_block_size, dropout)


@public_op
def apply_function_applicable_row_wise(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                                       function: Callable) -> BlksprsTensor:
    """Applies a callable independently to the packed rows of a block-sparse tensor.

    The callable must preserve the tensor shape and CUDA device and must not mix values between the packed rows. It may
    return another supported dtype.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        function (Callable): The row-wise callable to apply.

    Returns:
        BlksprsTensor: The result in compressed form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    if not callable(function):
        raise TypeError("Row-wise function must be callable")

    sparsity_layout_packed = _pack_layout(sparsity_layout)
    blksprs_pseudo_dense = bs.ops.to_dense(x, sparsity_layout_packed, sparsity_block_size)
    function_output = function(blksprs_pseudo_dense)

    if not isinstance(function_output, Tensor):
        raise TypeError("Row-wise function must return a Tensor")

    validate_shape(function_output, blksprs_pseudo_dense.shape, "Row-wise function output")
    validate_device(function_output, blksprs_pseudo_dense)
    validate_dtype_supported(function_output)

    blksprs_sparse = bs.ops.to_sparse(function_output, sparsity_layout_packed, sparsity_block_size)

    return blksprs_sparse


def _pack_layout(sparsity_layout: Tensor) -> Tensor:
    sparsity_layout_reshaped = sparsity_layout.reshape(1, sparsity_layout.size(0) * sparsity_layout.size(1),
                                                       sparsity_layout.size(2))
    non_zero_rows = torch.any(sparsity_layout_reshaped, dim=-1)

    if not torch.any(non_zero_rows):
        return sparsity_layout_reshaped

    sparsity_layout_filtered = sparsity_layout_reshaped[non_zero_rows].unsqueeze(0)

    return sparsity_layout_filtered
