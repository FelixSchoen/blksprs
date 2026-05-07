from collections.abc import Callable

import torch
from torch import Tensor, nn

import blksprs as bs
from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul_fast, build_sparsity_layout_matmul_outer
from blksprs.ops.conversion import to_sparse, to_sparse_shaped
from blksprs.ops.matmul import matmul
from blksprs.ops.repeat import repeat
from blksprs.utils.blksprs_tensor import BlksprsTensor


@torch.amp.custom_fwd(device_type="cuda")
def apply_torch_linear(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                       linear: nn.Linear, bias: nn.Parameter = None) -> (BlksprsTensor, Tensor):
    # Extract weight; bias uses the explicit override if provided, otherwise falls back to linear.bias
    w = linear.weight
    b = bias if bias is not None else linear.bias

    # Convert w to block-sparse representation
    sparsity_layout_w_t = torch.ones(size=(sparsity_layout.size(0), w.size(1) // sparsity_block_size,
                                           w.size(0) // sparsity_block_size), dtype=torch.bool, device=x.device)
    w_t_bs = to_sparse(w.transpose(-1, -2).unsqueeze(0).repeat(sparsity_layout.size(0), 1, 1),
                       sparsity_layout_w_t, sparsity_block_size)

    # Compute output sparsity layout
    # When bias is present, use outer bound (logical_or) to account for bias contributions everywhere
    # When no bias, use tighter inner bound (logical_and) for the matmul-only result
    if b is not None:
        sparsity_layout_output = build_sparsity_layout_matmul_outer(sparsity_layout, sparsity_layout_w_t)
    else:
        sparsity_layout_output = build_sparsity_layout_matmul_fast(sparsity_layout, sparsity_layout_w_t)

    # Apply weights
    xw = matmul(x, sparsity_layout, BlksprsTensor.wrap(w_t_bs.to(x.dtype)), sparsity_layout_w_t, sparsity_layout_output,
                sparsity_block_size)
    interim = xw

    # Apply bias
    if b is not None:
        b_slice = b.unsqueeze(0).unsqueeze(0).repeat(1, sparsity_block_size, 1)
        sparsity_layout_b_slice = torch.ones(size=(1, b_slice.size(1) // sparsity_block_size,
                                                   b_slice.size(2) // sparsity_block_size), dtype=torch.bool,
                                             device=x.device)
        b_slice_bs = to_sparse(b_slice, sparsity_layout_b_slice, sparsity_block_size)
        b_bs, sparsity_layout_b = repeat(b_slice_bs, sparsity_layout_b_slice,
                                         (sparsity_layout.size(0), sparsity_layout_output.size(1), 1),
                                         sparsity_block_size,
                                         sparsity_layout_output=sparsity_layout_output)
        interim = interim + b_bs

    return interim, sparsity_layout_output


@torch.amp.custom_fwd(device_type="cuda")
def apply_torch_linear_cached(x: BlksprsTensor,
                              sparsity_layout: Tensor,
                              sparsity_block_size: int,
                              linear: nn.Linear,
                              bias: nn.Parameter = None,
                              lut: dict = None) -> (BlksprsTensor, Tensor):
    """Applies a linear layer to a sparse tensor while caching packed weights.

    Args:
        x (BlksprsTensor): Input tensor in compressed form.
        sparsity_layout (Tensor): Sparsity layout of ``x``.
        sparsity_block_size (int): Size of the sparsity blocks.
        linear (nn.Linear): Linear projection to apply.
        bias (nn.Parameter, optional): Explicit bias override.
        lut (dict, optional): Mutable cache dictionary reused across calls.

    Returns:
        BlksprsTensor: Projected output tensor in compressed form.
        Tensor: Sparsity layout of the projected output.
    """
    if lut is None:
        lut = dict()

    w = linear.weight
    b = bias if bias is not None else linear.bias

    cache_key = (
        x.device.type,
        x.device.index,
        str(x.dtype),
        w.data_ptr(),
        int(w._version),
        tuple(w.shape),
        str(w.dtype),
        None if b is None else b.data_ptr(),
        None if b is None else int(b._version),
        int(sparsity_layout.size(0)),
        int(sparsity_block_size),
    )
    if lut.get("cache_key") != cache_key:
        lut.clear()
        lut["cache_key"] = cache_key

    if "sparsity_layout_w_t" not in lut:
        lut["sparsity_layout_w_t"] = torch.ones(
            size=(sparsity_layout.size(0),
                  w.size(1) // sparsity_block_size,
                  w.size(0) // sparsity_block_size),
            dtype=torch.bool,
            device=x.device,
        )

    if "w_t_bs" not in lut:
        w_t_dense = w.transpose(-1, -2).unsqueeze(0).repeat(
            sparsity_layout.size(0), 1, 1
        ).contiguous()
        lut["w_t_bs"] = to_sparse_shaped(
            w_t_dense,
            lut["sparsity_layout_w_t"],
            sparsity_block_size,
            lut=lut.setdefault("weight_lut", dict()),
        )

    if "sparsity_layout_output" not in lut:
        if b is not None:
            lut["sparsity_layout_output"] = build_sparsity_layout_matmul_outer(
                sparsity_layout,
                lut["sparsity_layout_w_t"],
            )
        else:
            lut["sparsity_layout_output"] = build_sparsity_layout_matmul_fast(
                sparsity_layout,
                lut["sparsity_layout_w_t"],
            )

    interim = matmul(
        x,
        sparsity_layout,
        BlksprsTensor.wrap(lut["w_t_bs"].to(x.dtype)),
        lut["sparsity_layout_w_t"],
        lut["sparsity_layout_output"],
        sparsity_block_size,
    )

    if b is not None:
        if "bias_slice_layout" not in lut:
            b_slice = b.unsqueeze(0).unsqueeze(0).repeat(
                1, sparsity_block_size, 1
            ).contiguous()
            lut["bias_slice_layout"] = torch.ones(
                size=(1,
                      b_slice.size(1) // sparsity_block_size,
                      b_slice.size(2) // sparsity_block_size),
                dtype=torch.bool,
                device=x.device,
            )
            lut["bias_slice_bs"] = to_sparse_shaped(
                b_slice,
                lut["bias_slice_layout"],
                sparsity_block_size,
                lut=lut.setdefault("bias_slice_lut", dict()),
            )

        b_bs, _ = repeat(
            lut["bias_slice_bs"],
            lut["bias_slice_layout"],
            (sparsity_layout.size(0), lut["sparsity_layout_output"].size(1), 1),
            sparsity_block_size,
            sparsity_layout_output=lut["sparsity_layout_output"],
            lut=lut.setdefault("bias_repeat_lut", dict()),
        )
        interim = interim + b_bs

    return interim, lut["sparsity_layout_output"]


def apply_torch_normalisation(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                              normalisation: nn.Module) -> BlksprsTensor:
    return apply_function_applicable_row_wise(x, sparsity_layout, sparsity_block_size, normalisation)


def apply_torch_dropout(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                        dropout: nn.Dropout) -> BlksprsTensor:
    return apply_function_applicable_row_wise(x, sparsity_layout, sparsity_block_size, dropout)


def apply_function_applicable_row_wise(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                                       function: Callable) -> BlksprsTensor:
    sparsity_layout_packed = _pack_layout(sparsity_layout)
    blksprs_pseudo_dense = bs.ops.to_dense(x, sparsity_layout_packed, sparsity_block_size)
    normalisation_out = function(blksprs_pseudo_dense)
    blksprs_sparse = bs.ops.to_sparse(normalisation_out, sparsity_layout_packed, sparsity_block_size)

    return blksprs_sparse


def _pack_layout(sparsity_layout: Tensor) -> BlksprsTensor:
    sparsity_layout_reshaped = sparsity_layout.reshape(1, sparsity_layout.size(0) * sparsity_layout.size(1),
                                                       sparsity_layout.size(2))
    non_zero_rows = torch.any(sparsity_layout_reshaped, dim=-1)

    if not torch.any(non_zero_rows):
        return sparsity_layout_reshaped

    sparsity_layout_filtered = sparsity_layout_reshaped[non_zero_rows].unsqueeze(0)

    return sparsity_layout_filtered
