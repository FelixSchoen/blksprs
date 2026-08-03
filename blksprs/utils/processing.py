from collections.abc import Callable

import torch
from torch import Tensor, nn

import blksprs as bs
from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul_fast
from blksprs.ops.conversion import to_sparse, to_sparse_shaped
from blksprs.ops.matmul import matmul
from blksprs.ops.repeat import repeat
from blksprs.utils.blksprs_tensor import BlksprsTensor
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


@torch.amp.custom_fwd(device_type="cuda")
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
    interim = xw

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

    return interim, sparsity_layout_output


@torch.amp.custom_fwd(device_type="cuda")
def apply_torch_linear_cached(x: BlksprsTensor,
                              sparsity_layout: Tensor,
                              sparsity_block_size: int,
                              linear: nn.Linear,
                              bias: nn.Parameter | None = None,
                              layout_cache: dict | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Applies a PyTorch linear layer while caching its compressed parameters.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        linear (nn.Linear): The linear layer to apply.
        bias (nn.Parameter, optional): An explicit bias overriding ``linear.bias`` (default ``None``).
        layout_cache (dict, optional): Reusable layout and parameter cache (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The projected tensor in compressed form and its sparsity layout.

    """
    w = linear.weight
    b = bias if bias is not None else linear.bias
    _validate_linear_dtypes(x, w, b)
    validate_sparsity_block_size(sparsity_block_size, w)

    layout_cache = prepare_layout_cache(
        layout_cache,
        "apply_torch_linear",
        sparsity_layout,
        sparsity_block_size,
        x.device,
        x.dtype,
        torch.is_grad_enabled(),
        w,
        b,
    )

    if "sparsity_layout_w_t" not in layout_cache:
        layout_cache["sparsity_layout_w_t"] = torch.ones(
            size=(sparsity_layout.size(0),
                  w.size(1) // sparsity_block_size,
                  w.size(0) // sparsity_block_size),
            dtype=torch.bool,
            device=x.device,
        )

    if "w_t_bs" not in layout_cache:
        w_t_dense = w.transpose(-1, -2).unsqueeze(0).repeat(
            sparsity_layout.size(0), 1, 1
        ).contiguous()
        layout_cache["w_t_bs"] = to_sparse_shaped(
            w_t_dense,
            layout_cache["sparsity_layout_w_t"],
            sparsity_block_size,
            layout_cache=layout_cache.setdefault("weight_layout_cache", dict()),
        )

    if "sparsity_layout_output" not in layout_cache:
        if b is not None:
            layout_cache["sparsity_layout_output"] = torch.ones(
                size=(sparsity_layout.size(0),
                      sparsity_layout.size(1),
                      w.size(0) // sparsity_block_size),
                dtype=torch.bool,
                device=x.device,
            )
        else:
            layout_cache["sparsity_layout_output"] = build_sparsity_layout_matmul_fast(
                sparsity_layout,
                layout_cache["sparsity_layout_w_t"],
            )

    interim = matmul(
        x,
        sparsity_layout,
        BlksprsTensor.wrap(layout_cache["w_t_bs"].to(x.dtype)),
        layout_cache["sparsity_layout_w_t"],
        layout_cache["sparsity_layout_output"],
        sparsity_block_size,
    )

    if b is not None:
        if "bias_slice_layout" not in layout_cache:
            b_slice = b.unsqueeze(0).unsqueeze(0).repeat(
                1, sparsity_block_size, 1
            ).contiguous()
            validate_sparsity_block_size(sparsity_block_size, b_slice)
            layout_cache["bias_slice_layout"] = torch.ones(
                size=(1,
                      b_slice.size(1) // sparsity_block_size,
                      b_slice.size(2) // sparsity_block_size),
                dtype=torch.bool,
                device=x.device,
            )
            layout_cache["bias_slice_bs"] = to_sparse_shaped(
                b_slice,
                layout_cache["bias_slice_layout"],
                sparsity_block_size,
                layout_cache=layout_cache.setdefault("bias_slice_layout_cache", dict()),
            )

        b_bs, _ = repeat(
            layout_cache["bias_slice_bs"],
            layout_cache["bias_slice_layout"],
            (sparsity_layout.size(0), layout_cache["sparsity_layout_output"].size(1), 1),
            sparsity_block_size,
            sparsity_layout_output=layout_cache["sparsity_layout_output"],
            layout_cache=layout_cache.setdefault("bias_repeat_layout_cache", dict()),
        )
        interim = interim + b_bs.to(interim.dtype)

    output_layout = layout_cache["sparsity_layout_output"]
    finalize_layout_cache(layout_cache)
    return interim, output_layout


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
