import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.layouting.sparsity_layout import build_sparsity_layout_adaption
from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs, prune_autotune_configs_conversion
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import as_base_tensor, build_layout_indices, stride, build_packed_indices, \
    can_use_int32_indexing, prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_sparsity_block_size, validate_sparsity_dense, ensure_contiguous, \
    validate_sparsity_layout, validate_shape, validate_dtype_supported


def to_blksprs(x: Tensor, sparsity_layout: Tensor, sparsity_block_size: int,
               layout_cache: dict | None = None) -> BlksprsTensor:
    """Converts a three-dimensional dense tensor to compressed form.

    This is an alias of :func:`to_sparse`.

    Args:
        x (Tensor): The dense input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: ``x`` in compressed form.

    """
    return to_sparse(x, sparsity_layout, sparsity_block_size, layout_cache=layout_cache)


def to_sparse_shaped(x: Tensor,
                     sparsity_layout: Tensor,
                     sparsity_block_size: int,
                     layout_cache: dict | None = None) -> BlksprsTensor:
    """Converts a three-dimensional dense tensor to compressed form.

    This convenience alias of :func:`to_sparse` makes it explicit that ``x`` already has shape ``(B, rows, cols)`` and
    does not need a preceding :func:`blksprs.utils.do_shape_blocksparse` call.

    Args:
        x (Tensor): The dense input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: ``x`` in compressed form.

    """
    return to_sparse(x, sparsity_layout, sparsity_block_size, layout_cache=layout_cache)


def is_row_striped_layout(sparsity_layout: Tensor, layout_cache: dict | None = None) -> bool:
    """Checks whether a sparsity layout is dense along columns for active rows.

    A row-striped layout marks either all blocks or no blocks in a given row.
    This pattern appears in sequence-feature tensors where only selected
    sequence rows are active while the feature dimension stays dense.

    Args:
        sparsity_layout (Tensor): The sparsity layout to inspect.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        bool: ``True`` if the layout is row-striped, ``False`` otherwise.

    """
    validate_sparsity_layout(sparsity_layout)

    if layout_cache is None:
        layout_cache = dict()

    row_striped_build_layout_cache(layout_cache, sparsity_layout)
    return layout_cache["is_row_striped"]


def to_sparse_row_striped(x: Tensor,
                          sparsity_layout: Tensor,
                          sparsity_block_size: int,
                          layout_cache: dict | None = None) -> BlksprsTensor:
    """Converts a three-dimensional dense tensor with row-striped sparsity to compressed form.

    This specialised path is faster than :func:`to_sparse` when the sparsity
    layout activates complete feature rows, because it can gather contiguous
    row blocks directly instead of running the generic sparse conversion
    kernel.

    Args:
        x (Tensor): The dense input tensor.
        sparsity_layout (Tensor): The row-striped sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: ``x`` in compressed form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity_dense(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_cache = row_striped_build_layout_cache(layout_cache, sparsity_layout)

    if not layout_cache["is_row_striped"]:
        raise ValueError(
            "to_sparse_row_striped requires a row-striped sparsity layout.")

    if layout_cache["n_sparse_blocks"] == 0:
        # Keep the empty result connected to ``x``. Constructing an unrelated
        # empty tensor would drop autograd for valid all-sparse and
        # zero-dimensional layouts.
        output = x.reshape(-1)[:0].reshape(
            0, sparsity_block_size, sparsity_block_size)
        return BlksprsTensor.wrap(output)

    if layout_cache["n_active_row_blocks"] == layout_cache["n_total_row_blocks"]:
        x_blocks = (
            x.reshape(
                x.size(0),
                sparsity_layout.size(1),
                sparsity_block_size,
                sparsity_layout.size(2),
                sparsity_block_size,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        return BlksprsTensor.wrap(x_blocks.reshape(-1, sparsity_block_size, sparsity_block_size))

    x_blocks_flat = (
        x.reshape(
            x.size(0),
            sparsity_layout.size(1),
            sparsity_block_size,
            sparsity_layout.size(2),
            sparsity_block_size,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, sparsity_layout.size(2), sparsity_block_size, sparsity_block_size)
        .contiguous()
    )
    selected_rows = x_blocks_flat.index_select(
        0, layout_cache["active_row_flat_indices"])

    return BlksprsTensor.wrap(selected_rows.reshape(-1, sparsity_block_size, sparsity_block_size))


@torch.amp.custom_fwd(device_type="cuda")
def to_sparse(x: Tensor, sparsity_layout: Tensor,
              sparsity_block_size: int, layout_cache: dict | None = None) -> BlksprsTensor:
    """Converts a three-dimensional dense tensor to compressed block-sparse form.

    Args:
        x (Tensor): The dense input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: ``x`` in compressed form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity_dense(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_cache = to_sparse_build_layout_cache(layout_cache, sparsity_layout)

    if sparsity_layout.size(1) == 1 and sparsity_layout.size(2) == 1 and torch.all(sparsity_layout):
        return BlksprsTensor.wrap(x)

    return BlksprsTensor.wrap(to_sparse_forward(x, sparsity_layout,
                                                layout_cache["layout_indices"], sparsity_block_size, layout_cache["n_sparse_blocks"]))


@triton_op("blksprs::to_sparse_forward", mutates_args={})
def to_sparse_forward(x: Tensor, _: Tensor,
                      layout_indices: Tensor, sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.empty(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_r, lidx_c = layout_indices.size()
        lidx_r_s, lidx_c_s = stride(layout_indices)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(x, layout_indices, output)

        (wrap_triton(to_sparse_kernel)[triton_grid]
         (x, x_b, x_b_s, x_r_s, x_c_s,
          layout_indices, lidx_r, lidx_r_s, lidx_c_s,
          output, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


def to_sparse_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return to_dense(grad_output, sparsity_layout, sparsity_block_size), None, None, None, None


@triton.autotune(
    configs=get_autotune_configs("conversion"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def to_sparse_kernel(x,
                     x_b, x_b_s, x_r_s, x_c_s,
                     lidx, lidx_r, lidx_r_s, lidx_c_s,
                     o,
                     o_b_s, o_r_s, o_c_s,
                     sparsity_block_size,
                     USE_INT64: tl.constexpr,
                     TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get sparsity index of current output block consisting of its batch, row, and column index
    spa_val_idx = pid_blk * lidx_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Load block from dense tensor
    blk_d_idx = (spa_bat * x_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + spa_row * sparsity_block_size +
                   tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + spa_col * sparsity_block_size +
                   tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
    blk_d_msk = ((blk_d_idx >= 0) &
                 (blk_d_idx < tl.cast(x_b, index_dtype) * x_b_s))
    blk_d = tl.load(x + blk_d_idx, mask=blk_d_msk, other=0)

    # Store block in sparse tensor
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
    blk_o_msk = ((blk_o_idx >= 0) &
                 (blk_o_idx < (pid_blk + 1) * o_b_s))
    tl.store(o + blk_o_idx, blk_d, mask=blk_o_msk)


def to_sparse_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor):
    layout_cache = prepare_layout_cache(layout_cache, "to_sparse", sparsity_layout)

    if "layout_indices" not in layout_cache:
        layout_indices = build_layout_indices(sparsity_layout)
        layout_cache["layout_indices"] = layout_indices

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = int(sparsity_layout.to(torch.int64).sum().item())
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout, layout_cache["layout_indices"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def to_sparse_setup_context(ctx, inputs, output):
    (_, sparsity_layout, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout, )
    ctx.sparsity_block_size = sparsity_block_size


to_sparse_forward.register_autograd(
    to_sparse_wrapper_backward, setup_context=to_sparse_setup_context)


def from_blksprs(x: BlksprsTensor, sparsity_layout: Tensor,
                 sparsity_block_size: int, fill_value: float = 0, layout_cache: dict | None = None) -> Tensor:
    """Converts a compressed block-sparse tensor to three-dimensional dense form.

    This is an alias of :func:`to_dense`.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        fill_value (float, optional): The value used for inactive blocks (default ``0``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        Tensor: ``x`` in three-dimensional dense form.

    """
    return to_dense(x, sparsity_layout, sparsity_block_size, fill_value=fill_value, layout_cache=layout_cache)


def to_dense_shaped(x: BlksprsTensor,
                    sparsity_layout: Tensor,
                    sparsity_block_size: int,
                    fill_value: float = 0,
                    layout_cache: dict | None = None) -> Tensor:
    """Converts a compressed block-sparse tensor to three-dimensional dense form.

    This convenience alias of :func:`to_dense` makes it explicit that the result retains shape ``(B, rows, cols)`` and
    does not need a subsequent :func:`blksprs.utils.undo_shape_blocksparse` call.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        fill_value (float, optional): The value used for inactive blocks (default ``0``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        Tensor: ``x`` in three-dimensional dense form.

    """
    return to_dense(x, sparsity_layout, sparsity_block_size, fill_value=fill_value, layout_cache=layout_cache)


def to_dense_row_striped(x: BlksprsTensor,
                         sparsity_layout: Tensor,
                         sparsity_block_size: int,
                         fill_value: float = 0,
                         layout_cache: dict | None = None) -> Tensor:
    """Converts a compressed row-striped tensor to three-dimensional dense form.

    This is the inverse of :func:`to_sparse_row_striped`.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The row-striped sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        fill_value (float, optional): The value used for inactive blocks (default ``0``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        Tensor: ``x`` in three-dimensional dense form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_cache = row_striped_build_layout_cache(layout_cache, sparsity_layout)

    if not layout_cache["is_row_striped"]:
        raise ValueError(
            "to_dense_row_striped requires a row-striped sparsity layout.")

    output_blocks_flat = torch.full(
        (
            sparsity_layout.size(0) * sparsity_layout.size(1),
            sparsity_layout.size(2),
            sparsity_block_size,
            sparsity_block_size,
        ),
        fill_value=fill_value,
        dtype=x.dtype,
        device=x.device,
    )

    # ``index_copy_`` also establishes the autograd edge for empty sources.
    # Skipping it when no sparse blocks are active would turn the dense result
    # into a constant and prevent callers from backpropagating through a valid
    # empty compressed tensor.
    source_rows = x.reshape(
        layout_cache["n_active_row_blocks"],
        sparsity_layout.size(2),
        sparsity_block_size,
        sparsity_block_size,
    )
    output_blocks_flat.index_copy_(
        0, layout_cache["active_row_flat_indices"], source_rows)

    return (
        output_blocks_flat.reshape(
            sparsity_layout.size(0),
            sparsity_layout.size(1),
            sparsity_layout.size(2),
            sparsity_block_size,
            sparsity_block_size,
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            sparsity_layout.size(0),
            sparsity_layout.size(1) * sparsity_block_size,
            sparsity_layout.size(2) * sparsity_block_size,
        )
        .contiguous()
    )


@torch.amp.custom_fwd(device_type="cuda")
def to_dense(x: BlksprsTensor, sparsity_layout: Tensor,
             sparsity_block_size: int, fill_value: float = 0, layout_cache: dict | None = None) -> Tensor:
    """Converts a compressed block-sparse tensor to three-dimensional dense form.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout (Tensor): The sparsity layout of ``x``.
        sparsity_block_size (int): The size of the sparsity blocks.
        fill_value (float, optional): The value used for inactive blocks (default ``0``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        Tensor: ``x`` in three-dimensional dense form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_cache = to_dense_build_layout_cache(layout_cache, sparsity_layout)

    if sparsity_layout.size(1) == 1 and sparsity_layout.size(2) == 1 and torch.all(sparsity_layout):
        return Tensor(x)

    return Tensor(to_dense_forward(x, sparsity_layout,
                                   layout_cache["packed_indices"], sparsity_block_size, fill_value))


@triton_op("blksprs::to_dense_forward", mutates_args={})
def to_dense_forward(x: Tensor, sparsity_layout: Tensor,
                     packed_indices: Tensor,
                     sparsity_block_size: int, fill_value: float) -> Tensor:
    with torch.no_grad():
        output = torch.full(size=(sparsity_layout.size(0), sparsity_layout.size(1) * sparsity_block_size,
                                  sparsity_layout.size(2) * sparsity_block_size), fill_value=fill_value,
                            dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.shape
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_b, s_l_r, s_l_c = sparsity_layout.size()
        s_l_b_s, s_l_r_s, s_l_c_s = stride(sparsity_layout)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(x, sparsity_layout, packed_indices, output)

        (wrap_triton(to_dense_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
          packed_indices,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


def to_dense_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout = ctx.saved_tensors[0]
    sparsity_block_size = ctx.sparsity_block_size

    return to_sparse(grad_output, sparsity_layout, sparsity_block_size), None, None, None, None


@triton.autotune(
    configs=get_autotune_configs("conversion"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    restore_value=["o"]
)
@triton.jit
def to_dense_kernel(x,
                    x_b, x_b_s, x_r_s, x_c_s,
                    s_l_b, s_l_b_s, s_l_r_s, s_l_c_s,
                    packed_indices,
                    o,
                    o_b, o_b_s, o_r_s, o_c_s,
                    sparsity_block_size,
                    USE_INT64: tl.constexpr,
                    TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get sparsity index of current block
    spa_row = (pid_row * TRITON_BLOCK_SIZE) // sparsity_block_size
    spa_col = (pid_col * TRITON_BLOCK_SIZE) // sparsity_block_size

    # Get packed index for current block
    packed_idx_idx = (pid_blk * s_l_b_s + spa_row *
                       s_l_r_s + spa_col * s_l_c_s)
    packed_idx_msk = ((packed_idx_idx >= 0) &
                       (packed_idx_idx < tl.cast(s_l_b, index_dtype) * s_l_b_s))
    packed_idx = tl.cast(tl.load(packed_indices +
                          packed_idx_idx, mask=packed_idx_msk, other=-1), index_dtype)

    # If block is present commence operations
    if packed_idx >= 0:
        blk_idx = (tl.cast(packed_idx, index_dtype) * x_b_s +
                   (((pid_row % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                     tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                   (((pid_col % (sparsity_block_size // TRITON_BLOCK_SIZE)) * TRITON_BLOCK_SIZE +
                     tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_msk = ((blk_idx >= 0) &
                   (blk_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk = tl.load(x + blk_idx, mask=blk_msk, other=0)

        o_idx = (pid_blk * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
        o_msk = ((o_idx >= 0) &
                 (o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.store(o + o_idx, blk, o_msk)


def to_dense_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor):
    layout_cache = prepare_layout_cache(layout_cache, "to_dense", sparsity_layout)

    if "packed_indices" not in layout_cache:
        layout_cache["packed_indices"] = build_packed_indices(sparsity_layout)

    validate_contiguous(layout_cache["packed_indices"])

    return finalize_layout_cache(layout_cache)


def row_striped_build_layout_cache(layout_cache: dict | None, sparsity_layout: Tensor):
    layout_cache = prepare_layout_cache(layout_cache, "row_striped", sparsity_layout)

    if "active_row_mask" not in layout_cache:
        active_row_mask = as_base_tensor(
            torch.all(sparsity_layout, dim=-1).contiguous())
        layout_cache["active_row_mask"] = active_row_mask

    if "is_row_striped" not in layout_cache:
        dense_rows = layout_cache["active_row_mask"].unsqueeze(
            -1).expand_as(sparsity_layout)
        layout_cache["is_row_striped"] = bool(torch.equal(sparsity_layout, dense_rows))

    if not layout_cache["is_row_striped"]:
        return finalize_layout_cache(layout_cache)

    if "active_row_flat_indices" not in layout_cache:
        active_row_flat_indices = as_base_tensor(torch.nonzero(
            layout_cache["active_row_mask"].reshape(-1), as_tuple=False
        ).squeeze(-1).contiguous())
        layout_cache["active_row_flat_indices"] = active_row_flat_indices

    if "n_active_row_blocks" not in layout_cache:
        layout_cache["n_active_row_blocks"] = int(
            layout_cache["active_row_flat_indices"].numel())

    if "n_total_row_blocks" not in layout_cache:
        layout_cache["n_total_row_blocks"] = int(
            sparsity_layout.size(0) * sparsity_layout.size(1))

    if "n_sparse_blocks" not in layout_cache:
        layout_cache["n_sparse_blocks"] = int(
            layout_cache["n_active_row_blocks"] * sparsity_layout.size(2))

    validate_contiguous(layout_cache["active_row_mask"])
    if layout_cache["n_active_row_blocks"] > 0:
        validate_contiguous(layout_cache["active_row_flat_indices"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def to_dense_setup_context(ctx, inputs, output):
    (_, sparsity_layout, _, sparsity_block_size, _) = inputs

    ctx.save_for_backward(sparsity_layout)
    ctx.sparsity_block_size = sparsity_block_size


to_dense_forward.register_autograd(
    to_dense_wrapper_backward, setup_context=to_dense_setup_context)


@torch.amp.custom_fwd(device_type="cuda")
def adapt_layout(x: BlksprsTensor, sparsity_layout_from: Tensor, sparsity_block_size_from: int,
                 sparsity_block_size_to: int, sparsity_layout_to: Tensor | None = None) -> tuple[BlksprsTensor, Tensor]:
    """Converts a compressed tensor to a different sparsity layout or block size.

    When the logical row or column extent is not divisible by a larger target
    block size, the output is extended to the next complete block and the
    trailing values are zero-padded. Gradient computation maps only the
    original logical extent back to ``x``.

    Args:
        x (BlksprsTensor): The compressed input tensor.
        sparsity_layout_from (Tensor): The sparsity layout of ``x``.
        sparsity_block_size_from (int): The current size of the sparsity blocks.
        sparsity_block_size_to (int): The target size of the sparsity blocks.
        sparsity_layout_to (Tensor, optional): The target sparsity layout. When omitted, it is derived from ``x``
            (default ``None``).

    Returns:
        tuple[BlksprsTensor, Tensor]: The adapted compressed tensor and its sparsity layout.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x, sparsity_layout_from)
    validate_device(x)
    validate_dtype_supported(x)
    validate_sparsity(sparsity_block_size_from, (x, sparsity_layout_from))
    validate_sparsity_block_size(sparsity_block_size_from, x)
    validate_sparsity_block_size(sparsity_block_size_to)

    packed_indices_from = build_packed_indices(sparsity_layout_from)

    if sparsity_layout_to is None:
        sparsity_layout_to = build_sparsity_layout_adaption(x, sparsity_layout_from,
                                                            sparsity_block_size_from, sparsity_block_size_to)
    else:
        sparsity_layout_to = as_base_tensor(ensure_contiguous(sparsity_layout_to))

    validate_sparsity_layout(sparsity_layout_to)
    validate_device(sparsity_layout_to, x)
    expected_shape = (
        sparsity_layout_from.size(0),
        (sparsity_layout_from.size(1) * sparsity_block_size_from + sparsity_block_size_to - 1)
        // sparsity_block_size_to,
        (sparsity_layout_from.size(2) * sparsity_block_size_from + sparsity_block_size_to - 1)
        // sparsity_block_size_to,
    )
    validate_shape(sparsity_layout_to, expected_shape, "Target sparsity layout")

    layout_indices_to = build_layout_indices(sparsity_layout_to)

    n_sparse_blocks_to = torch.sum(sparsity_layout_to.to(torch.int)).item()

    validate_contiguous(packed_indices_from,
                        sparsity_layout_to, layout_indices_to)

    if (sparsity_block_size_from == sparsity_block_size_to) and torch.equal(sparsity_layout_from, sparsity_layout_to):
        return BlksprsTensor.wrap(x), sparsity_layout_to

    return BlksprsTensor.wrap(adapt_layout_forward(x,
                                                   sparsity_layout_from, packed_indices_from,
                                                   sparsity_block_size_from,
                                                   sparsity_layout_to, layout_indices_to,
                                                   sparsity_block_size_to,
                                                   n_sparse_blocks_to)), sparsity_layout_to


@triton_op("blksprs::adapt_layout_forward", mutates_args={})
def adapt_layout_forward(x: Tensor,
                         sparsity_layout_from: Tensor, packed_indices_from: Tensor,
                         sparsity_block_size_from: int,
                         _: Tensor, layout_indices_to: Tensor,
                         sparsity_block_size_to: int,
                         n_sparse_blocks_to: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks_to, sparsity_block_size_to, sparsity_block_size_to),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_from.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_from)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        lidx_o_r, lidx_o_c = layout_indices_to.size()
        lidx_o_r_s, lidx_o_c_s = stride(layout_indices_to)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            sparsity_layout_from,
            packed_indices_from,
            output,
            layout_indices_to,
        )

        (wrap_triton(adapt_layout_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_r, s_l_x_c, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          packed_indices_from,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          layout_indices_to, lidx_o_r, lidx_o_r_s, lidx_o_c_s,
          sparsity_block_size_from,
          sparsity_block_size_to,
          USE_INT64=use_int64))

        return output


def adapt_layout_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout_from, sparsity_layout_to = ctx.saved_tensors
    sparsity_block_size_from = ctx.sparsity_block_size_from
    sparsity_block_size_to = ctx.sparsity_block_size_to

    # A coarser target block can extend the physical output beyond the source
    # extent. The reverse adaptation therefore expects the source layout on
    # that padded physical grid. Trailing inactive blocks retain the original
    # compressed ordering while allowing the reverse kernel to discard the
    # padded rows and columns correctly.
    reverse_layout_shape = (
        sparsity_layout_from.size(0),
        (sparsity_layout_to.size(1) * sparsity_block_size_to + sparsity_block_size_from - 1)
        // sparsity_block_size_from,
        (sparsity_layout_to.size(2) * sparsity_block_size_to + sparsity_block_size_from - 1)
        // sparsity_block_size_from,
    )
    padding_rows = reverse_layout_shape[1] - sparsity_layout_from.size(1)
    padding_columns = reverse_layout_shape[2] - sparsity_layout_from.size(2)
    if padding_rows > 0 or padding_columns > 0:
        padded_sparsity_layout_from = torch.zeros(
            reverse_layout_shape,
            dtype=sparsity_layout_from.dtype,
            device=sparsity_layout_from.device,
        )
        padded_sparsity_layout_from[
            :,
            :sparsity_layout_from.size(1),
            :sparsity_layout_from.size(2),
        ] = sparsity_layout_from
        sparsity_layout_from = padded_sparsity_layout_from

    return adapt_layout(
        grad_output, sparsity_layout_to, sparsity_block_size_to, sparsity_block_size_from,
        sparsity_layout_to=sparsity_layout_from)[0], None, None, None, None, None, None, None


@triton.autotune(
    configs=get_autotune_configs("conversion"),
    key=["sparsity_block_size_from", "sparsity_block_size_to"],
    prune_configs_by={"early_config_prune": prune_autotune_configs_conversion},
    reset_to_zero=["o"]
)
@triton.jit
def adapt_layout_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_l_x_b, s_l_x_r, s_l_x_c, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                        pidx_x,
                        o,
                        o_b, o_b_s, o_r_s, o_c_s,
                        lidx_o, lidx_o_r, lidx_o_r_s, lidx_o_c_s,
                        sparsity_block_size_from,
                        sparsity_block_size_to,
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

    # Get equivalent sparsity block in from layout
    spa_bat_x = spa_bat_o
    spa_row_x = (spa_row_o * sparsity_block_size_to + pid_row *
                 TRITON_BLOCK_SIZE) // sparsity_block_size_from
    spa_col_x = (spa_col_o * sparsity_block_size_to + pid_col *
                 TRITON_BLOCK_SIZE) // sparsity_block_size_from

    # Get packed indices for x
    packed_idx_x_idx = (spa_bat_x * s_l_x_b_s +
                         spa_row_x * s_l_x_r_s +
                         spa_col_x * s_l_x_c_s)
    packed_idx_x_msk = ((spa_bat_x >= 0) &
                         (spa_bat_x < tl.cast(s_l_x_b, index_dtype)) &
                         (spa_row_x >= 0) &
                         (spa_row_x < tl.cast(s_l_x_r, index_dtype)) &
                         (spa_col_x >= 0) &
                         (spa_col_x < tl.cast(s_l_x_c, index_dtype)))
    packed_idx_x = tl.cast(
        tl.load(pidx_x + packed_idx_x_idx, mask=packed_idx_x_msk, other=-1), index_dtype)

    # If block is present commence operations
    if packed_idx_x >= 0:
        # Calculate triton block size shifts
        shift_row_x = ((spa_row_o * sparsity_block_size_to + pid_row * TRITON_BLOCK_SIZE)
                       % sparsity_block_size_from) // TRITON_BLOCK_SIZE
        shift_col_x = ((spa_col_o * sparsity_block_size_to + pid_col * TRITON_BLOCK_SIZE)
                       % sparsity_block_size_from) // TRITON_BLOCK_SIZE

        # Load x values
        blk_x_idx = ((tl.cast(packed_idx_x, index_dtype) * x_b_s) +
                     ((shift_row_x * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                     ((shift_col_x * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_x_msk = ((blk_x_idx >= 0) &
                     (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

        # Store output
        blk_o_idx = ((pid_blk * o_b_s) +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
        blk_o_msk = ((blk_o_idx >= 0) &
                     (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


# noinspection PyUnusedLocal
def adapt_layout_setup_context(ctx, inputs, output):
    (_, sparsity_layout_from, _, sparsity_block_size_from,
     sparsity_layout_to, _, sparsity_block_size_to, _) = inputs

    ctx.save_for_backward(sparsity_layout_from, sparsity_layout_to)
    ctx.sparsity_block_size_from = sparsity_block_size_from
    ctx.sparsity_block_size_to = sparsity_block_size_to


adapt_layout_forward.register_autograd(
    adapt_layout_wrapper_backward, setup_context=adapt_layout_setup_context)
