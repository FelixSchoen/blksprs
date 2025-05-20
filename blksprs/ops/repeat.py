import torch
from torch import Tensor
from torch._library import triton_op

from blksprs.ops.flow import flow_pull_forward, flow_push_forward
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, \
    validate_sparsity, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def repeat(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
           sparsity_block_size: int, sparsity_layout_output: Tensor = None, lut: dict = None) -> (
        BlksprsTensor, Tensor):
    """Repeats a block-spare tensor in compressed form according to the given repeats.
    
    Repeats is a 3-tuple of integers, where each integer represents the number of times the tensor should be repeated in
        the first, second and third dimension respectively.
        
    Note:
        An output sparsity layout can be provided, in which case only the indicated blocks are filled. This may result
        in blocks not being present in the output that were present in the input if the output sparsity layout indicates
        them to be sparse.
    
    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (tuple[int, int, int]): The number of times the tensor should be repeated in the first, second and
            third dimension respectively.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: A block-sparse tensor in compressed form containing the repeated values.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = repeat_build_lut(lut, sparsity_layout_x, repeats, sparsity_layout_output)

    return BlksprsTensor(repeat_forward(
        x, sparsity_layout_x, lut["sparsity_layout_o"], lut["sparsity_lut"],
        lut["sparsity_reverse_lut"], sparsity_block_size, lut["n_sparse_blocks"])), lut["sparsity_layout_o"]


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def repeat_interleave(x: BlksprsTensor, sparsity_layout_x: Tensor, repeats: int,
                      sparsity_block_size: int, sparsity_layout_output: Tensor = None, lut: dict = None) -> (
        BlksprsTensor, Tensor):
    """Repeats and interleaves the block-sparse tensor in compressed form.

    Repeats each matrix contained in the tensors by ``repeats`` amount and places them consecutively in the output
        tensor.

    Note:
        In similar fashion to the regular ``repeat`` an output sparsity layout can be provided. In this case only
        non-sparse blocks will be filled.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        repeats (int): The number of times to repeat the matrices.
        sparsity_block_size (int): The size of the sparsity blocks.
        sparsity_layout_output (Tensor): The desired sparsity layout of the output tensor (default ``None``).
        lut (dict, optional): A dictionary containing the look-up tables for the operation (default ``None``).

    Returns:
        BlksprsTensor: A block-sparse tensor in compressed form containing the repeated and interleaved matrices.
        Tensor: The sparsity layout of the resulting output tensor.

    """
    x = x.contiguous()

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)

    lut = repeat_interleave_build_lut(lut, sparsity_layout_x, repeats, sparsity_layout_output)

    return BlksprsTensor(repeat_forward(
        x, sparsity_layout_x, lut["sparsity_layout_o"], lut["sparsity_lut"],
        lut["sparsity_reverse_lut"], sparsity_block_size, lut["n_sparse_blocks"])), lut["sparsity_layout_o"]


@triton_op("blksprs::repeat_forward", mutates_args={})
def repeat_forward(x: Tensor, _: Tensor, sparsity_layout_o: Tensor, sparsity_lut: Tensor,
                   sparsity_reverse_lut: Tensor,
                   sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        return flow_pull_forward(x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size,
                                 n_sparse_blocks)


def repeat_wrapper_backward(ctx, grad_output):
    sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut = ctx.saved_tensors
    sparsity_block_size = ctx.sparsity_block_size
    n_sparse_blocks = torch.sum(sparsity_layout_x.to(torch.int)).item()

    return flow_push_forward(grad_output, sparsity_layout_o, sparsity_lut,
                             sparsity_reverse_lut, sparsity_block_size,
                             n_sparse_blocks), None, None, None, None, None, None


def repeat_build_lut(lut: dict, sparsity_layout_x: Tensor, repeats: tuple[int, int, int],
                     sparsity_layout_output: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_layout_o" not in lut:
        sparsity_layout_o = sparsity_layout_x.repeat(repeats[0], repeats[1], repeats[2])
        lut["sparsity_layout_o"] = sparsity_layout_o

    if sparsity_layout_output is not None:
        sparsity_layout_o = torch.logical_and(lut["sparsity_layout_o"], sparsity_layout_output)
        lut["sparsity_layout_o"] = sparsity_layout_o

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(lut["sparsity_layout_o"]).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut" not in lut:
        sparsity_layout_flat = sparsity_layout_x.reshape(-1)
        sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                 (sparsity_layout_flat == 1) -
                                 (1 * (sparsity_layout_flat == 0)))
                                .reshape(sparsity_layout_x.size())
                                .repeat(repeats[0], repeats[1], repeats[2])
                                .reshape(-1).contiguous())
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(lut["sparsity_layout_o"].to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout_o, lut["sparsity_lut"], lut["sparsity_reverse_lut"])

    return lut


def repeat_interleave_build_lut(lut: dict, sparsity_layout_x: Tensor, repeats: int,
                                sparsity_layout_output: Tensor):
    if lut is None:
        lut = dict()

    if "sparsity_layout_o" not in lut:
        sparsity_layout_o = torch.repeat_interleave(sparsity_layout_x, repeats, dim=0).contiguous()
        lut["sparsity_layout_o"] = sparsity_layout_o

    if sparsity_layout_output is not None:
        sparsity_layout_o = torch.logical_and(lut["sparsity_layout_o"], sparsity_layout_output)
        lut["sparsity_layout_o"] = sparsity_layout_o

    if "sparsity_lut" not in lut:
        sparsity_lut = torch.nonzero(lut["sparsity_layout_o"]).contiguous()
        lut["sparsity_lut"] = sparsity_lut

    if "sparsity_reverse_lut" not in lut:
        sparsity_layout_flat = sparsity_layout_x.reshape(-1)
        sparsity_reverse_lut = (((torch.cumsum(sparsity_layout_flat, dim=-1) - 1) *
                                 (sparsity_layout_flat == 1) -
                                 (1 * (sparsity_layout_flat == 0)))
                                .reshape(sparsity_layout_x.size())
                                .repeat_interleave(repeats, dim=0)
                                .reshape(-1).contiguous())
        lut["sparsity_reverse_lut"] = sparsity_reverse_lut

    if "n_sparse_blocks" not in lut:
        n_sparse_blocks = torch.sum(lut["sparsity_layout_o"].to(torch.int)).item()
        lut["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout_o, lut["sparsity_lut"], lut["sparsity_reverse_lut"])

    return lut


# noinspection PyUnusedLocal
def repeat_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout_x, sparsity_layout_o, sparsity_lut, sparsity_reverse_lut)
    ctx.sparsity_block_size = sparsity_block_size


repeat_forward.register_autograd(repeat_wrapper_backward, setup_context=repeat_setup_context)
