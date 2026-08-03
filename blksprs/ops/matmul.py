import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.ops.transpose import transpose
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import build_layout_indices, stride, build_packed_indices, can_use_int32_indexing, \
    cast_for_autocast, prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_layout, validate_sparsity_block_size, validate_dtype_float, \
    validate_shape, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda")
def matmul(x: BlksprsTensor, sparsity_layout_x: Tensor,
           y: BlksprsTensor, sparsity_layout_y: Tensor,
           sparsity_layout_output: Tensor,
           sparsity_block_size: int, layout_cache: dict | None = None) -> BlksprsTensor:
    """Multiplies two compressed block-sparse tensors.

    Only blocks marked active by ``sparsity_layout_output`` are calculated.

    Args:
        x (BlksprsTensor): The compressed left operand.
        sparsity_layout_x (Tensor): The sparsity layout of ``x``.
        y (BlksprsTensor): The compressed right operand.
        sparsity_layout_y (Tensor): The sparsity layout of ``y``.
        sparsity_layout_output (Tensor): The sparsity layout of the output tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: The matrix product in compressed form.

    """
    x, y = ensure_contiguous(x, y)
    x, y = cast_for_autocast(x, y)

    validate_dimensions(x, y)
    validate_contiguous(x, y, sparsity_layout_output)
    validate_dtype_float(x, y)
    validate_device(x, y, sparsity_layout_output)
    validate_sparsity(sparsity_block_size,
                      (x, sparsity_layout_x), (y, sparsity_layout_y))
    validate_sparsity_layout(sparsity_layout_output)
    if sparsity_layout_x.size(0) != sparsity_layout_y.size(0):
        raise ValueError("Batch dimensions of tensors must match")
    if sparsity_layout_x.size(-1) != sparsity_layout_y.size(-2):
        raise ValueError("Inner dimensions of tensors must match")
    validate_shape(
        sparsity_layout_output,
        (sparsity_layout_x.size(0), sparsity_layout_x.size(1), sparsity_layout_y.size(2)),
        "Output sparsity layout",
    )
    validate_sparsity_block_size(sparsity_block_size, x, y)

    layout_cache = matmul_build_layout_cache(layout_cache, sparsity_layout_x,
                           sparsity_layout_y, sparsity_layout_output)

    return BlksprsTensor.wrap(matmul_forward(x, y,
                                             sparsity_layout_x, layout_cache["packed_indices_x"],
                                             sparsity_layout_y, layout_cache["packed_indices_y"],
                                             sparsity_layout_output, layout_cache["layout_indices_o"],
                                             sparsity_block_size, layout_cache["n_sparse_blocks"]))


@triton_op("blksprs::matmul_forward", mutates_args={})
def matmul_forward(x: Tensor, y: Tensor,
                   sparsity_layout_x: Tensor, packed_indices_x: Tensor,
                   sparsity_layout_y: Tensor, packed_indices_y: Tensor,
                   _: Tensor, layout_indices_o: Tensor,
                   sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        y_b, y_r, y_c = y.size()
        y_b_s, y_r_s, y_c_s = stride(y)
        s_l_y_b, s_l_y_r, s_l_y_c = sparsity_layout_y.size()
        s_l_y_b_s, s_l_y_r_s, s_l_y_c_s = stride(sparsity_layout_y)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        lidx_o_r, lidx_o_c = layout_indices_o.size()
        lidx_o_r_s, lidx_o_c_s = stride(layout_indices_o)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            sparsity_layout_x,
            packed_indices_x,
            y,
            sparsity_layout_y,
            packed_indices_y,
            output,
            layout_indices_o,
        )

        (wrap_triton(matmul_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_r, s_l_x_b_s, s_l_x_r_s,
          s_l_x_c, s_l_x_c_s,
          packed_indices_x,
          y,
          y_b, y_b_s, y_r_s, y_c_s,
          s_l_y_b, s_l_y_r, s_l_y_b_s, s_l_y_r_s,
          s_l_y_c, s_l_y_c_s,
          packed_indices_y,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          layout_indices_o,
          lidx_o_r, lidx_o_r_s, lidx_o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


def matmul_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_o = ctx.saved_tensors
    sparsity_block_size = ctx.sparsity_block_size

    x_t, sparsity_layout_x_t = transpose(
        x, sparsity_layout_x, sparsity_block_size)
    y_t, sparsity_layout_y_t = transpose(
        y, sparsity_layout_y, sparsity_block_size)

    grad_x = matmul(grad_output, sparsity_layout_o, y_t, sparsity_layout_y_t, sparsity_layout_x,
                    sparsity_block_size)
    grad_y = matmul(x_t, sparsity_layout_x_t, grad_output, sparsity_layout_o, sparsity_layout_y,
                    sparsity_block_size)

    return grad_x, grad_y, None, None, None, None, None, None, None, None


@triton.autotune(
    configs=get_autotune_configs("matmul"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def matmul_kernel(x,
                  x_b, x_b_s, x_r_s, x_c_s,
                  s_l_x_b, s_l_x_r, s_l_x_b_s, s_l_x_r_s, s_l_x_c, s_l_x_c_s,
                  pidx_x,
                  y,
                  y_b, y_b_s, y_r_s, y_c_s,
                  s_l_y_b, s_l_y_r, s_l_y_b_s, s_l_y_r_s, s_l_y_c, s_l_y_c_s,
                  pidx_y,
                  o,
                  o_b, o_b_s, o_r_s, o_c_s,
                  lidx_o,
                  lidx_o_r, lidx_o_r_s,
                  lidx_o_c_s,
                  sparsity_block_size,
                  USE_INT64: tl.constexpr,
                  TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    index_dtype = tl.int64 if USE_INT64 else tl.int32
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

    # Setup buffer
    buf = tl.zeros(shape=(TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE),
                   dtype=tl.float32)

    # Slide over triton block sized segments of input tensors
    for i_seg_tri in range(0, tl.cdiv(s_l_x_c * sparsity_block_size, TRITON_BLOCK_SIZE)):
        # Convert to segment index of sparsity layout
        i_seg_spa = (i_seg_tri * TRITON_BLOCK_SIZE) // sparsity_block_size
        # Calculate the triton segment index within a block
        i_seg_tri_mod = i_seg_tri % (sparsity_block_size // TRITON_BLOCK_SIZE)

        # Get packed indices for input tensors x and y
        # These are either -1 if the block is empty or equal to the index of the block in the sparse tensor

        # Get packed indices for x
        packed_idx_x_idx = (spa_bat_o * s_l_x_b_s +
                             spa_row_o * s_l_x_r_s +
                             i_seg_spa * s_l_x_c_s)
        packed_idx_x_msk = ((spa_bat_o >= 0) &
                             (spa_bat_o < tl.cast(s_l_x_b, index_dtype)) &
                             (spa_row_o >= 0) &
                             (spa_row_o < tl.cast(s_l_x_r, index_dtype)) &
                             (i_seg_spa >= 0) &
                             (i_seg_spa < tl.cast(s_l_x_c, index_dtype)))
        packed_idx_x = tl.cast(
            tl.load(pidx_x + packed_idx_x_idx, mask=packed_idx_x_msk, other=-1), index_dtype)

        # Get packed indices for y
        packed_idx_y_idx = (spa_bat_o * s_l_y_b_s +
                             i_seg_spa * s_l_y_r_s + spa_col_o * s_l_y_c_s)
        packed_idx_y_msk = ((spa_bat_o >= 0) &
                             (spa_bat_o < tl.cast(s_l_y_b, index_dtype)) &
                             (i_seg_spa >= 0) &
                             (i_seg_spa < tl.cast(s_l_y_r, index_dtype)) &
                             (spa_col_o >= 0) &
                             (spa_col_o < tl.cast(s_l_y_c, index_dtype)))
        packed_idx_y = tl.cast(
            tl.load(pidx_y + packed_idx_y_idx, mask=packed_idx_y_msk, other=-1), index_dtype)

        # If both blocks are present commence calculation
        if packed_idx_x >= 0 and packed_idx_y >= 0:
            blk_x_idx = ((tl.cast(packed_idx_x, index_dtype) * x_b_s) +
                         ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                         ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
            blk_x_msk = ((blk_x_idx >= 0) &
                         (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
            blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

            blk_y_idx = ((tl.cast(packed_idx_y, index_dtype) * y_b_s) +
                         ((i_seg_tri_mod * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * y_r_s)[:, None] +
                         ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * y_c_s)[None, :])
            blk_y_msk = ((blk_y_idx >= 0) &
                         (blk_y_idx < tl.cast(y_b, index_dtype) * y_b_s))
            blk_y = tl.load(y + blk_y_idx, mask=blk_y_msk, other=0)

            # Perform matrix multiplication
            buf += tl.dot(blk_x, blk_y)

    # Cast buffer
    buf = tl.cast(buf, o.dtype.element_ty)

    # Store output
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
    blk_o_msk = ((blk_o_idx >= 0) &
                 (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
    tl.store(o + blk_o_idx, buf, mask=blk_o_msk)


def matmul_build_layout_cache(layout_cache: dict | None, sparsity_layout_x: Tensor, sparsity_layout_y: Tensor, sparsity_layout_output: Tensor):
    layout_cache = prepare_layout_cache(
        layout_cache, "matmul", sparsity_layout_x, sparsity_layout_y, sparsity_layout_output)

    if "packed_indices_x" not in layout_cache:
        layout_cache["packed_indices_x"] = build_packed_indices(sparsity_layout_x)

    if "packed_indices_y" not in layout_cache:
        layout_cache["packed_indices_y"] = build_packed_indices(sparsity_layout_y)

    if "layout_indices_o" not in layout_cache:
        layout_indices_o = build_layout_indices(sparsity_layout_output)
        layout_cache["layout_indices_o"] = layout_indices_o

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(
            sparsity_layout_output.to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout_x, layout_cache["packed_indices_x"],
                        sparsity_layout_y, layout_cache["packed_indices_y"],
                        sparsity_layout_output, layout_cache["layout_indices_o"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def matmul_setup_context(ctx, inputs, output):
    (x, y, sparsity_layout_x, _, sparsity_layout_y, _,
     sparsity_layout_o, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(x, sparsity_layout_x, y,
                          sparsity_layout_y, sparsity_layout_o)
    ctx.sparsity_block_size = sparsity_block_size


matmul_forward.register_autograd(
    matmul_wrapper_backward, setup_context=matmul_setup_context)
