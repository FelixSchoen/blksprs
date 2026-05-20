import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import build_packed_indices
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda")
def transpose(x: BlksprsTensor, sparsity_layout: Tensor,
              sparsity_block_size: int, layout_cache: dict = None) -> (BlksprsTensor, Tensor):
    """Transposes a block-sparse tensor in compressed form.

    Note:
         Returns the transposed tensor and the sparsity layout of the transposed tensor.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): A dictionary containing the layout cache data for the operation (default ``None``).

    Returns:
        BlksprsTensor: The transposed block-sparse tensor in compressed form.
        Tensor: The sparsity layout of the transposed tensor.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_cache = transpose_build_layout_cache(layout_cache, sparsity_layout)

    return BlksprsTensor.wrap(transpose_forward(x, layout_cache["sparsity_layout_t"],
                                                layout_cache["layout_indices"], layout_cache["packed_indices"],
                                                sparsity_block_size, layout_cache["n_sparse_blocks"])), layout_cache["sparsity_layout_t"]


@triton_op("blksprs::transpose_forward", mutates_args={})
def transpose_forward(x: Tensor, sparsity_layout_o: Tensor,
                      layout_indices: Tensor, packed_indices: Tensor,
                      sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        x_t = x.transpose(-1, -2).contiguous()
        return flow_pull_forward(x_t, sparsity_layout_o, layout_indices, packed_indices,
                                 sparsity_block_size, n_sparse_blocks)


def transpose_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return transpose(grad_output, sparsity_layout, sparsity_block_size)[
        0], None, None, None, None, None


def transpose_build_layout_cache(layout_cache: dict, sparsity_layout: Tensor):
    if layout_cache is None:
        layout_cache = dict()

    if "sparsity_layout_t" not in layout_cache:
        sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()
        layout_cache["sparsity_layout_t"] = sparsity_layout_t

    if "layout_indices" not in layout_cache:
        layout_indices = torch.nonzero(layout_cache["sparsity_layout_t"]).contiguous()
        layout_cache["layout_indices"] = layout_indices

    if "packed_indices" not in layout_cache:
        packed_indices = (build_packed_indices(sparsity_layout)
                                .reshape(sparsity_layout.size()).transpose(-1, -2).contiguous().reshape(-1))
        layout_cache["packed_indices"] = packed_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(layout_cache["sparsity_layout_t"], layout_cache["layout_indices"], layout_cache["packed_indices"])

    return layout_cache


# noinspection PyUnusedLocal
def transpose_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.sparsity_block_size = sparsity_block_size


transpose_forward.register_autograd(transpose_wrapper_backward, setup_context=transpose_setup_context)
