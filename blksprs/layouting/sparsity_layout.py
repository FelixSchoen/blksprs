import math

import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs, prune_autotune_configs_conversion
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import build_layout_indices, stride, can_use_int32_indexing
from blksprs.utils.validation import validate_dimensions, validate_device, \
    validate_contiguous, validate_sparsity, validate_sparsity_block_size, \
    validate_sparsity_layout, validate_dtype_supported, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda")
def build_sparsity_layout(x: Tensor, sparsity_block_size: int) -> Tensor:
    """Builds the sparsity layout of a three-dimensional dense tensor.

    Args:
        x (Tensor): The dense tensor whose active blocks are represented by the layout.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        Tensor: A Boolean sparsity layout marking every block containing a non-zero or NaN value.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity_block_size(sparsity_block_size, x)

    return Tensor(build_sparsity_layout_operation(x, sparsity_block_size))


@triton_op("blksprs::build_sparsity_layout", mutates_args={})
def build_sparsity_layout_operation(x: Tensor, sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size,
                             dtype=torch.bool, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(x, output)

        (wrap_triton(build_sparsity_layout_kernel)[triton_grid]
         (x,
          x_b, x_r, x_c, x_b_s, x_r_s, x_c_s,
          output,
          o_b, o_r, o_c, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def build_sparsity_layout_kernel(x,
                                 x_b, x_r, x_c, x_b_s, x_r_s, x_c_s,
                                 o,
                                 o_b, o_r, o_c, o_b_s, o_r_s, o_c_s,
                                 sparsity_block_size,
                                 USE_INT64: tl.constexpr,
                                 TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    # Get triton block indices
    pid_bat = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Load x values
    row_offsets = pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    col_offsets = pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    blk_x_idx = (pid_bat * x_b_s +
                 (row_offsets * x_r_s)[:, None] +
                 (col_offsets * x_c_s)[None, :])
    blk_x_msk = ((pid_bat < tl.cast(x_b, index_dtype)) &
                 (row_offsets[:, None] < tl.cast(x_r, index_dtype)) &
                 (col_offsets[None, :] < tl.cast(x_c, index_dtype)))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Store sparsity layout value
    if tl.max(blk_x != 0):
        out_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
        out_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size
        blk_o_idx = (pid_bat * o_b_s +
                     (out_row * o_r_s) +
                     (out_col * o_c_s))
        blk_o_msk = ((pid_bat < tl.cast(o_b, index_dtype)) &
                     (out_row < tl.cast(o_r, index_dtype)) &
                     (out_col < tl.cast(o_c, index_dtype)))
        tl.store(o + blk_o_idx, 1, mask=blk_o_msk)


@torch.amp.custom_fwd(device_type="cuda")
def build_sparsity_layout_adaption(x: BlksprsTensor, sparsity_layout_from: Tensor,
                                   sparsity_block_size_from: int, sparsity_block_size_to: int) -> Tensor:
    """Builds the sparsity layout for a compressed tensor at a different block size.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout_from (Tensor): The sparsity layout of ``x``.
        sparsity_block_size_from (int): The current size of the sparsity blocks.
        sparsity_block_size_to (int): The desired size of the sparsity blocks.

    Returns:
        Tensor: The sparsity layout of ``x`` at the target block size.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout_from)
    validate_dtype_supported(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size_from, (x, sparsity_layout_from))
    validate_sparsity_block_size(sparsity_block_size_from, x)
    validate_sparsity_block_size(sparsity_block_size_to)

    layout_indices = build_layout_indices(sparsity_layout_from)

    validate_contiguous(sparsity_layout_from, layout_indices)

    return Tensor(build_sparsity_layout_adaption_operation(
        x, sparsity_layout_from, layout_indices, sparsity_block_size_from, sparsity_block_size_to))


@triton_op("blksprs::build_sparsity_layout_adaption", mutates_args={})
def build_sparsity_layout_adaption_operation(x: Tensor, sparsity_layout_from: Tensor, layout_indices: Tensor,
                                             sparsity_block_size_from: int, sparsity_block_size_to: int) -> Tensor:
    with torch.no_grad():
        o_b = sparsity_layout_from.size(0)
        o_r = math.ceil(sparsity_layout_from.size(1) * sparsity_block_size_from / sparsity_block_size_to)
        o_c = math.ceil(sparsity_layout_from.size(2) * sparsity_block_size_from / sparsity_block_size_to)

        output = torch.zeros(o_b, o_r, o_c, dtype=torch.bool, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_r, lidx_c = layout_indices.size()
        lidx_r_s, lidx_c_s = stride(layout_indices)
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(x, layout_indices, output)

        (wrap_triton(build_sparsity_layout_adaption_kernel)[triton_grid]
         (x,
          x_b, x_r, x_c, x_b_s, x_r_s, x_c_s,
          layout_indices, lidx_r, lidx_r_s, lidx_c_s,
          output,
          o_b, o_r, o_c, o_b_s, o_r_s, o_c_s,
          sparsity_block_size_from,
          sparsity_block_size_to,
          USE_INT64=use_int64))

        return output


@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size_from", "sparsity_block_size_to"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_conversion},
    reset_to_zero=["o"]
)
@triton.jit
def build_sparsity_layout_adaption_kernel(x,
                                          x_b, x_r, x_c, x_b_s, x_r_s, x_c_s,
                                          lidx, lidx_r, lidx_r_s, lidx_c_s,
                                          o,
                                          o_b, o_r, o_c, o_b_s, o_r_s, o_c_s,
                                          sparsity_block_size_from,
                                          sparsity_block_size_to,
                                          USE_INT64: tl.constexpr,
                                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get sparsity index of current output block consisting of its batch, row, and column index
    spa_bat_idx = (pid_blk * lidx_r_s + 0 * lidx_c_s)
    spa_bat_msk = ((spa_bat_idx >= 0) &
                   (spa_bat_idx < tl.cast(lidx_r, index_dtype) * lidx_r_s))
    spa_bat = tl.cast(tl.load(lidx + spa_bat_idx, mask=spa_bat_msk, other=0), index_dtype)

    spa_row_idx = (pid_blk * lidx_r_s + 1 * lidx_c_s)
    spa_row_msk = ((spa_row_idx >= 0) &
                   (spa_row_idx < tl.cast(lidx_r, index_dtype) * lidx_r_s))
    spa_row = tl.cast(tl.load(lidx + spa_row_idx, mask=spa_row_msk, other=0), index_dtype)

    spa_col_idx = (pid_blk * lidx_r_s + 2 * lidx_c_s)
    spa_col_msk = ((spa_col_idx >= 0) &
                   (spa_col_idx < tl.cast(lidx_r, index_dtype) * lidx_r_s))
    spa_col = tl.cast(tl.load(lidx + spa_col_idx, mask=spa_col_msk, other=0), index_dtype)

    # Load x values
    row_offsets = pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    col_offsets = pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    blk_x_idx = ((pid_blk * x_b_s) +
                 (row_offsets * x_r_s)[:, None] +
                 (col_offsets * x_c_s)[None, :])
    blk_x_msk = ((pid_blk < tl.cast(x_b, index_dtype)) &
                 (row_offsets[:, None] < tl.cast(x_r, index_dtype)) &
                 (col_offsets[None, :] < tl.cast(x_c, index_dtype)))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Store sparsity layout value
    if tl.max(blk_x != 0):
        out_row = ((pid_row * TRITON_BLOCK_SIZE + spa_row * sparsity_block_size_from)
                   // sparsity_block_size_to)
        out_col = ((pid_col * TRITON_BLOCK_SIZE + spa_col * sparsity_block_size_from)
                   // sparsity_block_size_to)
        blk_o_idx = ((spa_bat * o_b_s) +
                     (out_row * o_r_s) +
                     (out_col * o_c_s))
        blk_o_msk = ((spa_bat < tl.cast(o_b, index_dtype)) &
                     (out_row < tl.cast(o_r, index_dtype)) &
                     (out_col < tl.cast(o_c, index_dtype)))
        tl.store(o + blk_o_idx, 1, mask=blk_o_msk)


@torch.amp.custom_fwd(device_type="cuda")
def build_sparsity_layout_matmul(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor) -> Tensor:
    """Builds the exact structural layout product for matrix multiplication.

    Args:
        sparsity_layout_x (Tensor): The sparsity layout of the left operand.
        sparsity_layout_y (Tensor): The sparsity layout of the right operand.

    Returns:
        Tensor: A layout marking blocks with at least one structurally active multiplication path.

    """
    _validate_sparsity_layout_matmul_inputs(sparsity_layout_x, sparsity_layout_y)
    return Tensor(torch.matmul(
        sparsity_layout_x.to(torch.float), sparsity_layout_y.to(torch.float)).to(torch.bool))


@torch.amp.custom_fwd(device_type="cuda")
def build_sparsity_layout_matmul_fast(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor) -> Tensor:
    """Builds a fast conservative layout approximation for matrix multiplication.

    Note:
        A block at ``(i, j)`` is active when row ``i`` of the left layout and column ``j`` of the right layout each
        contain at least one active block. The active blocks do not need to share an inner-dimension position, so the
        result may contain blocks that the exact structural product would omit.

    Args:
        sparsity_layout_x (Tensor): The sparsity layout of the left operand.
        sparsity_layout_y (Tensor): The sparsity layout of the right operand.

    Returns:
        Tensor: A conservative approximation of the output sparsity layout.

    """
    _validate_sparsity_layout_matmul_inputs(sparsity_layout_x, sparsity_layout_y)
    sparsity_layout_x_slice = torch.any(sparsity_layout_x, dim=-1, keepdim=True)
    sparsity_layout_y_slice = torch.any(sparsity_layout_y, dim=-2).unsqueeze(1)

    return Tensor(torch.logical_and(sparsity_layout_x_slice, sparsity_layout_y_slice))


@torch.amp.custom_fwd(device_type="cuda")
def build_sparsity_layout_matmul_outer(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor) -> Tensor:
    """Builds an outer layout approximation for matrix multiplication.

    A block at ``(i, j)`` is active when row ``i`` of the left layout or column ``j`` of the right layout contains an
    active block. This is useful when operations beyond the matrix multiplication, such as bias addition, may populate
    additional output blocks.

    Note:
        This approximation is looser than :func:`build_sparsity_layout_matmul_fast` and may therefore include more
        blocks than the exact structural product.

    Args:
        sparsity_layout_x (Tensor): The sparsity layout of the left operand.
        sparsity_layout_y (Tensor): The sparsity layout of the right operand.

    Returns:
        Tensor: An outer approximation of the output sparsity layout.

    """
    _validate_sparsity_layout_matmul_inputs(sparsity_layout_x, sparsity_layout_y)
    sparsity_layout_x_slice = torch.any(sparsity_layout_x, dim=-1, keepdim=True)
    sparsity_layout_y_slice = torch.any(sparsity_layout_y, dim=-2).unsqueeze(1)

    return Tensor(torch.logical_or(sparsity_layout_x_slice, sparsity_layout_y_slice))


def _validate_sparsity_layout_matmul_inputs(sparsity_layout_x: Tensor, sparsity_layout_y: Tensor) -> None:
    validate_sparsity_layout(sparsity_layout_x, sparsity_layout_y)
    validate_device(sparsity_layout_x, sparsity_layout_y)

    if sparsity_layout_x.size(0) != sparsity_layout_y.size(0):
        raise ValueError("Batch dimensions of sparsity layouts must match")
    if sparsity_layout_x.size(-1) != sparsity_layout_y.size(-2):
        raise ValueError("Inner dimensions of sparsity layouts must match")


def build_sparsity_layout_full(x: Tensor, sparsity_block_size: int) -> Tensor:
    """Builds a fully populated sparsity layout for a dense tensor.

    Args:
        x (Tensor): A three-dimensional dense tensor.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        Tensor: A boolean sparsity layout in which every block is active.

    """
    validate_dimensions(x)
    validate_device(x)
    validate_sparsity_block_size(sparsity_block_size, x)
    return torch.ones(size=(x.size(0), x.size(1) // sparsity_block_size, x.size(2) // sparsity_block_size),
                      dtype=torch.bool, device=x.device)
