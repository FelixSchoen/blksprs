from collections.abc import Callable

import torch
from torch import Tensor, nn

import blksprs as bs
from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul_fast
from blksprs.ops.conversion import to_sparse
from blksprs.ops.matmul import matmul
from blksprs.ops.repeat import repeat
from blksprs.utils.blksprs_tensor import BlksprsTensor


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def apply_torch_linear(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                       linear: nn.Linear, bias: nn.Parameter = None) -> (BlksprsTensor, Tensor):
    # Extract weight and bias
    w = linear.weight
    b = linear.bias

    # Convert w to block-sparse representation
    sparsity_layout_w_t = torch.ones(size=(sparsity_layout.size(0), w.size(1) // sparsity_block_size,
                                           w.size(0) // sparsity_block_size), dtype=torch.bool, device=x.device)
    w_t_bs = to_sparse(w.transpose(-1, -2).unsqueeze(0).repeat(sparsity_layout.size(0), 1, 1),
                       sparsity_layout_w_t, sparsity_block_size)

    # Apply weights
    sparsity_layout_xw = build_sparsity_layout_matmul_fast(sparsity_layout, sparsity_layout_w_t)
    xw = matmul(x, sparsity_layout, BlksprsTensor(w_t_bs.to(x.dtype)), sparsity_layout_w_t, sparsity_layout_xw, sparsity_block_size)
    interim = xw

    # Apply bias
    if bias is not None:
        b = bias
    if b is not None:
        b_slice = b.unsqueeze(0).unsqueeze(0).repeat(1, sparsity_block_size, 1)
        sparsity_layout_b_slice = torch.ones(size=(1, b_slice.size(1) // sparsity_block_size,
                                                   b_slice.size(2) // sparsity_block_size), dtype=torch.bool,
                                             device=x.device)
        b_slice_bs = to_sparse(b_slice, sparsity_layout_b_slice, sparsity_block_size)
        b_bs, sparsity_layout_b = repeat(b_slice_bs, sparsity_layout_b_slice,
                                         (sparsity_layout.size(0), sparsity_layout_xw.size(1), 1), sparsity_block_size,
                                         sparsity_layout_output=sparsity_layout_xw)
        interim = interim + b_bs

    return interim, sparsity_layout_xw


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
    sparsity_layout_filtered = sparsity_layout_reshaped[non_zero_rows].unsqueeze(0)

    return sparsity_layout_filtered
