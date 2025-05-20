import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def transpose(x: BlksprsTensor, sparsity_layout: Tensor,
              sparsity_block_size: int, lut: dict = None) -> (BlksprsTensor, Tensor):
    """Transposes a block-sparse tensor in compressed form.

    Note:
         Returns the transposed tensor and the sparsity layout of the transposed tensor.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: The transposed block-sparse tensor in compressed form.
        Tensor: The sparsity layout of the transposed tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = transpose_build_lut(lut, sparsity_layout)

    return BlksprsTensor(transpose_forward(x, lut["sparsity_layout_t"],
                                           lut["sparsity_lut"], lut["sparsity_reverse_lut"],
                                           sparsity_block_size, lut["n_sparse_blocks"])), lut["sparsity_layout_t"]


@triton_op("blksprs::transpose_forward", mutates_args={})
def transpose_forward(x: Tensor, sparsity_layout_o: Tensor,
                      sparsity_lut: Tensor, sparsity_reverse_lut: Tensor,
                      sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        x_t = x.transpose(-1, -2).contiguous()
        return flow_pull_forward(x_t, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut,
                                 sparsity_block_size, n_sparse_blocks)


def transpose_wrapper_backward(ctx, grad_output):
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return transpose(grad_output, sparsity_layout, sparsity_block_size)[
        0], None, None, None, None, None


def transpose_build_lut(lut: dict, sparsity_layout: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_layout_t" not in lut:
        sparsity_layout_t = sparsity_layout.transpose(-1, -2).contiguous()
        lut["sparsity_layout_t"] = sparsity_layout_t

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(lut["sparsity_layout_t"]).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut" not in lut:
        sparsity_layout_flat = sparsity_layout.reshape(-1)
        sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                 (sparsity_layout_flat == 1) -
                                 (1 * (sparsity_layout_flat == 0)))
                                .reshape(sparsity_layout.size()).transpose(-1, -2).contiguous().reshape(-1))
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(sparsity_layout.to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(lut["sparsity_layout_t"], lut["sparsity_lut"], lut["sparsity_reverse_lut"])

    return lut


# noinspection PyUnusedLocal
def transpose_setup_context(ctx, inputs, output):
    (_, sparsity_layout_o, _, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_o)
    ctx.sparsity_block_size = sparsity_block_size


transpose_forward.register_autograd(transpose_wrapper_backward, setup_context=transpose_setup_context)
