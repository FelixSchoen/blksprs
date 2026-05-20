import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride, build_packed_indices, can_use_int32_indexing
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, validate_sparsity, \
    validate_sparsity_block_size, validate_dtype_float, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda")
def row_wise_sum(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                 flag_slice_only: bool = False) -> (BlksprsTensor, Tensor):
    """Computes the row-wise sum of a block-sparse tensor.

    Returns a block-sparse tensor in compressed form with only one block per row, where the first entry contains the sum
        of the corresponding row.

    Note:
        If ``flag_slice_only`` is set the output will be of shape ``[x.size(0), x.size(1), 1]``.

    Note:
        This operation does not support gradient computation.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        flag_slice_only (bool, optional): If set the output will be of shape ``[x.size(0), x.size(1), 1]``
            (default ``False``).

    Returns:
        tuple[BlksprsTensor, Tensor]: A tuple containing a block-sparse tensor in compressed form containing the row-wise sum
            of the input and the sparsity layout of the output tensor.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_indices = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    packed_indices_output = build_packed_indices(sparsity_layout_output)

    n_sparse_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout, layout_indices,
                        sparsity_layout_output, packed_indices_output)

    return BlksprsTensor.wrap(row_wise_sum_forward(
        x, layout_indices, sparsity_layout_output, packed_indices_output,
        sparsity_block_size, n_sparse_blocks_output, flag_slice_only)), sparsity_layout_output


@triton_op("blksprs::row_wise_sum_forward", mutates_args={})
def row_wise_sum_forward(x: Tensor, layout_indices: Tensor,
                         sparsity_layout_output: Tensor, packed_indices_output: Tensor,
                         sparsity_block_size: int, n_sparse_blocks_output: int,
                         flag_slice_only: bool = False) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(
            size=(n_sparse_blocks_output, sparsity_block_size, 1 if flag_slice_only else sparsity_block_size),
            dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_x_r, lidx_x_c = layout_indices.size()
        lidx_x_r_s, lidx_x_c_s = stride(layout_indices)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            layout_indices,
            output,
            sparsity_layout_output,
            packed_indices_output)

        (wrap_triton(row_wise_sum_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          layout_indices, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
          output,
          o_b, o_b_s, o_r_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s,
          packed_indices_output,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("row_wise"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def row_wise_sum_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        lidx_x, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
                        o,
                        o_b, o_b_s, o_r_s,
                        s_l_o_b, s_l_o_b_s, s_l_o_r_s,
                        pidx_o,
                        sparsity_block_size,
                        USE_INT64: tl.constexpr,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32

    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * lidx_x_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx_x + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Load packed index for current block
    packed_idx_idx = (spa_bat_x * s_l_o_b_s +
                       spa_row_x * s_l_o_r_s)
    packed_idx_msk = ((packed_idx_idx >= 0) &
                       (packed_idx_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
    packed_idx = tl.cast(tl.load(pidx_o + packed_idx_idx, mask=packed_idx_msk, other=-1), tl.int32)

    if packed_idx >= 0:
        blk_idx = ((pid_blk * x_b_s) +
                   ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                   ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_msk = ((blk_idx >= 0) &
                   (blk_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk = tl.load(x + blk_idx, mask=blk_msk, other=0)

        buf = tl.reshape(tl.sum(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (tl.cast(packed_idx, index_dtype) * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 (tl.cast(tl.arange(0, 1), index_dtype))[None, :])
        o_msk = ((o_idx >= 0) &
                 (o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.atomic_add(o + o_idx, buf, o_msk)


@torch.amp.custom_fwd(device_type="cuda")
def row_wise_max(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                 flag_slice_only: bool = False) -> (BlksprsTensor, Tensor):
    """Computes the row-wise max of a block-sparse tensor.

    Returns a block-sparse tensor in compressed form with only one block per row, where the first entry contains the
        maximum of the corresponding row.

    Note:
        If ``flag_slice_only`` is set the output will be of shape ``[x.size(0), x.size(1), 1]``.

    Note:
        This operation does not support gradient computation.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout (Tensor): The sparsity layout of the block-sparse tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        flag_slice_only (bool, optional): If set the output will be of shape ``[x.size(0), x.size(1), 1]``
            (default ``False``).

    Returns:
        tuple[BlksprsTensor, Tensor]: A tuple containing a block-sparse tensor in compressed form containing the row-wise max
            of the input and the sparsity layout of the output tensor.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_dtype_float(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_indices = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    packed_indices_output = build_packed_indices(sparsity_layout_output)

    n_sparse_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout, layout_indices,
                        sparsity_layout_output, packed_indices_output)

    return BlksprsTensor.wrap(
        row_wise_max_forward(x, layout_indices, sparsity_layout_output, packed_indices_output, sparsity_block_size,
                             n_sparse_blocks_output, flag_slice_only)), sparsity_layout_output


@triton_op("blksprs::row_wise_max_forward", mutates_args={})
def row_wise_max_forward(x: Tensor, layout_indices: Tensor,
                         sparsity_layout_output: Tensor, packed_indices_output: Tensor,
                         sparsity_block_size: int, n_sparse_blocks_output: int,
                         flag_slice_only: bool = False) -> Tensor:
    with torch.no_grad():
        output = torch.full(size=(n_sparse_blocks_output,
                                  sparsity_block_size,
                                  1 if flag_slice_only else sparsity_block_size),
                            fill_value=torch.finfo(x.dtype).min,
                            device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_x_r, lidx_x_c = layout_indices.size()
        lidx_x_r_s, lidx_x_c_s = stride(layout_indices)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            layout_indices,
            output,
            sparsity_layout_output,
            packed_indices_output)

        (wrap_triton(row_wise_max_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          layout_indices, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
          output,
          o_b, o_b_s, o_r_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s,
          packed_indices_output,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("row_wise"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    restore_value=["o"]
)
@triton.jit
def row_wise_max_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        lidx_x, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
                        o,
                        o_b, o_b_s, o_r_s,
                        s_l_o_b, s_l_o_b_s, s_l_o_r_s,
                        pidx_o,
                        sparsity_block_size,
                        USE_INT64: tl.constexpr,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32

    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * lidx_x_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx_x + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Load packed index for current block
    packed_idx_idx = (spa_bat_x * s_l_o_b_s +
                       spa_row_x * s_l_o_r_s)
    packed_idx_msk = ((packed_idx_idx >= 0) &
                       (packed_idx_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
    packed_idx = tl.cast(tl.load(pidx_o + packed_idx_idx, mask=packed_idx_msk, other=-1), tl.int32)

    if packed_idx >= 0:
        blk_idx = ((pid_blk * x_b_s) +
                   ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                   ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_msk = ((blk_idx >= 0) &
                   (blk_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk = tl.load(x + blk_idx, mask=blk_msk, other=float("-inf"))

        buf = tl.reshape(tl.max(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (tl.cast(packed_idx, index_dtype) * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 (tl.cast(tl.arange(0, 1), index_dtype))[None, :])
        o_msk = ((o_idx >= 0) &
                 (o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.atomic_max(o + o_idx, buf, o_msk)


@torch.amp.custom_fwd(device_type="cuda")
def row_wise_add(x: BlksprsTensor, sparsity_layout_x: Tensor, y: Tensor,
                 sparsity_block_size: int) -> BlksprsTensor:
    """For each row in ``y`` adds the value to each value in the corresponding row of the block-sparse tensor ``x``.

    Note:
        This operation does not support gradient computation.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        y (BlksprsTensor): A block-sparse tensor in compressed form with only one value per row and a single column of sparse blocks.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        BlksprsTensor: The values of ``x`` with the first value of ``y`` in each row added to them as a block-sparse tensor in
            compressed form.

    """
    x, y = ensure_contiguous(x, y)

    validate_dimensions(x, y)
    validate_contiguous(x, y)
    validate_dtype_float(x)
    validate_dtype_float(y)
    validate_device(x, y)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)

    layout_indices_x = torch.nonzero(sparsity_layout_x).contiguous()

    sparsity_layout_rwm, _ = torch.max(sparsity_layout_x, dim=-1, keepdim=True)
    packed_indices_rwm = build_packed_indices(sparsity_layout_rwm)
    n_sparse_blocks_rwm = torch.sum(sparsity_layout_rwm.to(torch.int)).item()

    if y.size(0) != n_sparse_blocks_rwm:
        raise ValueError("Row-wise tensor does not conform to input sparsity layout")
    if y.size(-2) != sparsity_block_size or y.size(-1) not in (1, sparsity_block_size):
        raise ValueError(
            "Row-wise tensor blocks must have sparsity block size rows and either 1 or sparsity block size columns")

    validate_contiguous(sparsity_layout_x, layout_indices_x, packed_indices_rwm)

    return BlksprsTensor.wrap(row_wise_add_forward(x, layout_indices_x, sparsity_layout_rwm,
                                                   packed_indices_rwm, y, sparsity_block_size))


def row_wise_sub(x: BlksprsTensor, sparsity_layout_x: Tensor, y: Tensor,
                 sparsity_block_size: int) -> BlksprsTensor:
    """Wrapper for ``row_wise_add`` with negated y.

    """
    return row_wise_add(x, sparsity_layout_x, torch.neg(y), sparsity_block_size)


@triton_op("blksprs::row_wise_add_forward", mutates_args={})
def row_wise_add_forward(x: Tensor, layout_indices_x: Tensor,
                         sparsity_layout_x_rwm: Tensor, packed_indices_rwm: Tensor,
                         y: Tensor, sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros_like(x)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_r, lidx_c = layout_indices_x.size()
        lidx_r_s, lidx_c_s = stride(layout_indices_x)
        y_b, y_r, y_c = y.size()
        y_b_s, y_r_s, y_c_s = stride(y)
        s_l_y_b, s_l_y_r, s_l_y_c = sparsity_layout_x_rwm.size()
        s_l_y_b_s, s_l_y_r_s, s_l_y_c_s = stride(sparsity_layout_x_rwm)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            layout_indices_x,
            y,
            sparsity_layout_x_rwm,
            packed_indices_rwm,
            output)

        (wrap_triton(row_wise_add_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          layout_indices_x, lidx_r, lidx_r_s, lidx_c_s,
          y, y_b, y_b_s, y_r_s, y_c_s,
          s_l_y_b, s_l_y_b_s, s_l_y_r_s,
          packed_indices_rwm,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("row_wise"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def row_wise_add_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        lidx_x, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
                        y, y_b, y_b_s, y_r_s, y_c_s,
                        s_l_y_b, s_l_y_b_s, s_l_y_r_s,
                        pidx_y,
                        o,
                        o_b, o_b_s, o_r_s, o_c_s,
                        sparsity_block_size,
                        USE_INT64: tl.constexpr,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32

    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * lidx_x_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx_x + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Get packed indices for s
    packed_idx_s_idx = (spa_bat_x * s_l_y_b_s +
                         spa_row_x * s_l_y_r_s)
    packed_idx_s_msk = ((packed_idx_s_idx >= 0) &
                         (packed_idx_s_idx < tl.cast(s_l_y_b, index_dtype) * s_l_y_b_s))
    packed_idx_s = tl.cast(tl.load(pidx_y + packed_idx_s_idx, mask=packed_idx_s_msk, other=-1), tl.int32)

    if packed_idx_s == -1:
        tl.device_assert(False)
        return

    # Load x block
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
    blk_x_msk = ((blk_x_idx >= 0) &
                 (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Load sum block
    blk_s_idx = (tl.cast(packed_idx_s, index_dtype) * y_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * y_r_s)[:, None] +
                 (tl.cast(tl.arange(0, 1), index_dtype) * y_c_s)[None, :])
    blk_s_msk = ((blk_s_idx >= 0) &
                 (blk_s_idx < tl.cast(y_b, index_dtype) * y_b_s))
    blk_s = tl.load(y + blk_s_idx, mask=blk_s_msk, other=0)

    # Compute exp
    buf = blk_x + tl.broadcast_to(blk_s, (TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE))

    # Store block
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
    blk_o_msk = ((blk_o_idx >= 0) &
                 (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
    tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
