import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import build_layout_indices, stride, can_use_int32_indexing, cast_for_autocast
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity_layout, validate_sparsity_block_size, validate_shape, validate_divisible, ensure_contiguous, \
    validate_same_dtype, validate_dtype_supported


@torch.amp.custom_fwd(device_type="cuda")
def broadcast_add(x: Tensor, y: Tensor, sparsity_layout_output: Tensor,
                  sparsity_block_size: int) -> BlksprsTensor:
    """Adds two dense row slices and returns selected output blocks in compressed form.

    Note:
        This operation does not support gradient computation.

    Args:
        x (Tensor): The dense input tensor with shape ``(B, M)``.
        y (Tensor): The dense input tensor with shape ``(B, N)``.
        sparsity_layout_output (Tensor): The sparsity layout of the output tensor.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        BlksprsTensor: The selected blocks of ``x[:, :, None] + y[:, None, :]`` in compressed form.

    """
    x, y = ensure_contiguous(x, y)
    x, y = cast_for_autocast(x, y)

    validate_dimensions(x, dims=2)
    validate_dimensions(y, dims=2)
    validate_same_dtype(x, y)
    validate_dtype_supported(x, y)
    validate_dimensions(sparsity_layout_output)
    validate_sparsity_layout(sparsity_layout_output)
    validate_device(x, y, sparsity_layout_output)
    validate_contiguous(x, y, sparsity_layout_output)
    if x.size(0) != y.size(0):
        raise ValueError("Batch dimensions of tensors must match")
    validate_sparsity_block_size(sparsity_block_size)
    validate_divisible(x.size(-1), sparsity_block_size, "Tensor sizes", "sparsity block size")
    validate_divisible(y.size(-1), sparsity_block_size, "Tensor sizes", "sparsity block size")
    validate_shape(
        sparsity_layout_output,
        (x.size(0), x.size(-1) // sparsity_block_size, y.size(-1) // sparsity_block_size),
        "Output sparsity layout",
    )

    layout_indices_o = build_layout_indices(sparsity_layout_output)

    n_sparse_blocks = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout_output, layout_indices_o)

    return BlksprsTensor.wrap(broadcast_add_forward(x, y, layout_indices_o, sparsity_block_size, n_sparse_blocks))


def broadcast_sub(x: Tensor, y: Tensor, sparsity_layout_output: Tensor,
                  sparsity_block_size: int) -> BlksprsTensor:
    """Subtracts two dense row slices and returns selected output blocks in compressed form.

    This is a convenience wrapper around :func:`broadcast_add` with ``y`` negated.

    Args:
        x (Tensor): The dense input tensor with shape ``(B, M)``.
        y (Tensor): The dense input tensor with shape ``(B, N)``.
        sparsity_layout_output (Tensor): The sparsity layout of the output tensor.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        BlksprsTensor: The selected blocks of ``x[:, :, None] - y[:, None, :]`` in compressed form.

    """
    if y.dtype == torch.bool:
        raise ValueError("Broadcast subtraction does not support bool tensors")
    return broadcast_add(x, torch.neg(y), sparsity_layout_output, sparsity_block_size)


@triton_op("blksprs::broadcast_add_forward", mutates_args={})
def broadcast_add_forward(x: Tensor, y: Tensor,
                          layout_indices_o: Tensor,
                          sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(n_sparse_blocks, sparsity_block_size, sparsity_block_size, dtype=x.dtype, device=x.device)

        x_b, x_c = x.size()
        x_b_s, x_c_s = stride(x)
        y_b, y_c = y.size()
        y_b_s, y_c_s = stride(y)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        lidx_o_r, lidx_o_c = layout_indices_o.size()
        lidx_o_r_s, lidx_o_c_s = stride(layout_indices_o)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(x, y, output, layout_indices_o)

        (wrap_triton(broadcast_add_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_c_s,
          y,
          y_b, y_b_s, y_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          layout_indices_o, lidx_o_r, lidx_o_r_s, lidx_o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


@triton.autotune(
    configs=get_autotune_configs("broadcast"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def broadcast_add_kernel(x,
                         x_b, x_b_s, x_c_s,
                         y,
                         y_b, y_b_s, y_c_s,
                         o,
                         o_b, o_b_s, o_r_s, o_c_s,
                         lidx_o, lidx_o_r, lidx_o_r_s, lidx_o_c_s,
                         sparsity_block_size,
                         USE_INT64: tl.constexpr,
                         TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32

    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch, row, and column index
    spa_val_idx = pid_blk * lidx_o_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_o_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx_o + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat_o = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row_o = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col_o = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Load x block
    blk_x_idx = (spa_bat_o * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + spa_row_o * sparsity_block_size +
                   tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
    blk_x_msk = ((blk_x_idx >= 0) &
                 (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Load y block
    blk_y_idx = (spa_bat_o * y_b_s +
                 ((pid_col * TRITON_BLOCK_SIZE + spa_col_o * sparsity_block_size +
                   tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * y_c_s)[None, :])
    blk_y_msk = ((blk_y_idx >= 0) &
                 (blk_y_idx < tl.cast(y_b, index_dtype) * y_b_s))
    blk_y = tl.load(y + blk_y_idx, mask=blk_y_msk, other=0)

    # Compute sum
    blk_x, blk_y = tl.broadcast(tl.trans(blk_x), blk_y)
    buf = blk_x + blk_y

    # Store result
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
    blk_o_msk = ((blk_o_idx >= 0) &
                 (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
    tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
