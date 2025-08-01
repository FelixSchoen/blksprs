import torch
import triton
from torch import Tensor
from torch._library.triton import wrap_triton, triton_op
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride
from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_device, validate_sparsity, \
    validate_sparsity_block_size, ensure_contiguous


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def row_wise_sum(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                 flag_slice_only: bool = False) -> (BlksprsTensor, Tensor):
    """Computes the row-wise sum of a block-sparse tensor.

    Returns a block-sparse tensor in compressed form with only one block per row, where the first entry contains the sum
        of the corresponding row.

    Note:
        If ``flag_slice_only`` is set the output will be of shape ``[x.size(0), x.size(1), 1]``.

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
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    sparsity_layout_output_flat = sparsity_layout_output.reshape(-1)
    sparsity_reverse_lut_output = ((torch.cumsum(sparsity_layout_output_flat, dim=-1) - 1) *
                                   (sparsity_layout_output_flat == 1) -
                                   (1 * (sparsity_layout_output_flat == 0)))

    n_sparse_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout, sparsity_lut,
                        sparsity_layout_output, sparsity_reverse_lut_output)

    return BlksprsTensor(row_wise_sum_forward(
        x, sparsity_lut, sparsity_layout_output, sparsity_reverse_lut_output,
        sparsity_block_size, n_sparse_blocks_output, flag_slice_only)), sparsity_layout_output


@triton_op("blksprs::row_wise_sum_forward", mutates_args={})
def row_wise_sum_forward(x: Tensor, sparsity_lut: Tensor,
                         sparsity_layout_output: Tensor, sparsity_reverse_lut_output: Tensor,
                         sparsity_block_size: int, n_sparse_blocks_output: int,
                         flag_slice_only: bool = False) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(
            size=(n_sparse_blocks_output, sparsity_block_size, 1 if flag_slice_only else sparsity_block_size),
            dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_x_r, s_lut_x_c = sparsity_lut.size()
        s_lut_x_r_s, s_lut_x_c_s = stride(sparsity_lut)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(row_wise_sum_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
          output,
          o_b, o_b_s, o_r_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s,
          sparsity_reverse_lut_output,
          sparsity_block_size))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def row_wise_sum_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                        o,
                        o_b, o_b_s, o_r_s,
                        s_l_o_b, s_l_o_b_s, s_l_o_r_s,
                        r_lut_o,
                        sparsity_block_size,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * s_lut_x_r_s + tl.arange(0, 4) * s_lut_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(s_lut_x + spa_val_idx, mask=spa_val_msk)

    spa_bat_x = tl.sum(spa_val * (tl.arange(0, 4) == 0))
    spa_row_x = tl.sum(spa_val * (tl.arange(0, 4) == 1))
    spa_col_x = tl.sum(spa_val * (tl.arange(0, 4) == 2))

    # Load reverse sparsity index for current block
    rev_idx_spa_idx = (spa_bat_x * s_l_o_b_s +
                       spa_row_x * s_l_o_r_s)
    rev_idx_spa_msk = (rev_idx_spa_idx >= 0 and rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
    rev_idx_spa = tl.load(r_lut_o + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa >= 0:
        blk_idx = ((pid_blk * x_b_s) +
                   ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                   ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_msk = (blk_idx >= 0 and
                   blk_idx < x_b * x_b_s)
        blk = tl.load(x + blk_idx, mask=blk_msk)

        buf = tl.reshape(tl.sum(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (rev_idx_spa * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 (tl.arange(0, 1))[None, :])
        o_msk = (o_idx >= 0 and
                 o_idx < o_b * o_b_s)
        tl.atomic_add(o + o_idx, buf, o_msk)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def row_wise_max(x: BlksprsTensor, sparsity_layout: Tensor, sparsity_block_size: int,
                 flag_slice_only: bool = False) -> (BlksprsTensor, Tensor):
    """Computes the row-wise max of a block-sparse tensor.

    Returns a block-sparse tensor in compressed form with only one block per row, where the first entry contains the
        maximum of the corresponding row.

    Note:
        If ``flag_slice_only`` is set the output will be of shape ``[x.size(0), x.size(1), 1]``.

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
    # TODO Fix for triton bug, see https://github.com/triton-lang/triton/issues/6376, should be fixed with the upcoming 3.4.0 release
    x = torch.where(x == -0.0, torch.tensor(0.0), x)
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout))
    validate_sparsity_block_size(sparsity_block_size, x)

    sparsity_lut = torch.nonzero(sparsity_layout).contiguous()

    sparsity_layout_output, _ = torch.max(sparsity_layout, dim=-1, keepdim=True)
    sparsity_layout_output_flat = sparsity_layout_output.reshape(-1)
    sparsity_reverse_lut_output = ((torch.cumsum(sparsity_layout_output_flat, dim=-1) - 1) *
                                   (sparsity_layout_output_flat == 1) -
                                   (1 * (sparsity_layout_output_flat == 0)))

    n_sparse_blocks_output = torch.sum(sparsity_layout_output.to(torch.int)).item()

    validate_contiguous(sparsity_layout, sparsity_lut,
                        sparsity_layout_output, sparsity_reverse_lut_output)

    return BlksprsTensor(
        row_wise_max_forward(x, sparsity_lut, sparsity_layout_output, sparsity_reverse_lut_output, sparsity_block_size,
                             n_sparse_blocks_output, flag_slice_only)), sparsity_layout_output


@triton_op("blksprs::row_wise_max_forward", mutates_args={})
def row_wise_max_forward(x: Tensor, sparsity_lut: Tensor,
                         sparsity_layout_output: Tensor, sparsity_reverse_lut_output: Tensor,
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
        s_lut_x_r, s_lut_x_c = sparsity_lut.size()
        s_lut_x_r_s, s_lut_x_c_s = stride(sparsity_lut)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_output.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_output)

        triton_grid = lambda meta: [x_b,
                                    triton.cdiv(x_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(row_wise_max_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
          output,
          o_b, o_b_s, o_r_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s,
          sparsity_reverse_lut_output,
          sparsity_block_size))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    restore_value=["o"]
)
@triton.jit
def row_wise_max_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                        o,
                        o_b, o_b_s, o_r_s,
                        s_l_o_b, s_l_o_b_s, s_l_o_r_s,
                        r_lut_o,
                        sparsity_block_size,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * s_lut_x_r_s + tl.arange(0, 4) * s_lut_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(s_lut_x + spa_val_idx, mask=spa_val_msk)

    spa_bat_x = tl.sum(spa_val * (tl.arange(0, 4) == 0))
    spa_row_x = tl.sum(spa_val * (tl.arange(0, 4) == 1))
    spa_col_x = tl.sum(spa_val * (tl.arange(0, 4) == 2))

    # Load reverse sparsity index for current block
    rev_idx_spa_idx = (spa_bat_x * s_l_o_b_s +
                       spa_row_x * s_l_o_r_s)
    rev_idx_spa_msk = (rev_idx_spa_idx >= 0 and rev_idx_spa_idx < s_l_o_b * s_l_o_b_s)
    rev_idx_spa = tl.load(r_lut_o + rev_idx_spa_idx, mask=rev_idx_spa_msk).to(tl.int32)

    if rev_idx_spa >= 0:
        blk_idx = ((pid_blk * x_b_s) +
                   ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                   ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
        blk_msk = (blk_idx >= 0 and
                   blk_idx < x_b * x_b_s)
        blk = tl.load(x + blk_idx, mask=blk_msk)

        buf = tl.reshape(tl.max(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

        o_idx = (rev_idx_spa * o_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 (tl.arange(0, 1))[None, :])
        o_msk = (o_idx >= 0 and
                 o_idx < o_b * o_b_s)
        tl.atomic_max(o + o_idx, buf, o_msk)


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def row_wise_add(x: BlksprsTensor, sparsity_layout_x: Tensor, y: Tensor,
                 sparsity_block_size: int) -> BlksprsTensor:
    """For each row in ``y`` adds the value to each value in the corresponding row of the block-sparse tensor ``x``.

    Args:
        x (BlksprsTensor): A block-sparse tensor in compressed form.
        sparsity_layout_x (Tensor): The sparsity layout of the block-sparse tensor.
        y (BlksprsTensor): A block-sparse tensor in compressed form with only one value per row and a single column of sparse blocks.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        BlksprsTensor: The values of ``x`` with the first value of ``y`` in each row added to them as a block-sparse tensor in
            compressed form.

    """
    x = ensure_contiguous(x)

    validate_dimensions(x)
    validate_contiguous(x)
    validate_device(x)
    validate_sparsity(sparsity_block_size, (x, sparsity_layout_x))
    validate_sparsity_block_size(sparsity_block_size, x)

    sparsity_lut_x = torch.nonzero(sparsity_layout_x).contiguous()

    sparsity_layout_rwm, _ = torch.max(sparsity_layout_x, dim=-1, keepdim=True)
    sparsity_layout_rwm_flat = sparsity_layout_rwm.reshape(-1)
    sparsity_reverse_lut_rwm = ((torch.cumsum(sparsity_layout_rwm_flat, dim=-1) - 1) *
                                (sparsity_layout_rwm_flat == 1) -
                                (1 * (sparsity_layout_rwm_flat == 0)))

    validate_contiguous(sparsity_layout_x, sparsity_lut_x, sparsity_reverse_lut_rwm)

    return BlksprsTensor(row_wise_add_forward(x, sparsity_lut_x, sparsity_layout_rwm,
                                              sparsity_reverse_lut_rwm, y, sparsity_block_size))


def row_wise_sub(x: BlksprsTensor, sparsity_layout_x: Tensor, y: Tensor,
                 sparsity_block_size: int) -> BlksprsTensor:
    """Wrapper for ``row_wise_add`` with negated y.

    """
    return row_wise_add(x, sparsity_layout_x, torch.neg(y), sparsity_block_size)


@triton_op("blksprs::row_wise_add_forward", mutates_args={})
def row_wise_add_forward(x: Tensor, sparsity_lut_x: Tensor,
                         sparsity_layout_x_rwm: Tensor, sparsity_reverse_x_lut_rwm: Tensor,
                         y: Tensor, sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros_like(x)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_lut_r, s_lut_c = sparsity_lut_x.size()
        s_lut_r_s, s_lut_c_s = stride(sparsity_lut_x)
        y_b, y_r, y_c = y.size()
        y_b_s, y_r_s, y_c_s = stride(y)
        s_l_y_b, s_l_y_r, s_l_y_c = sparsity_layout_x_rwm.size()
        s_l_y_b_s, s_l_y_r_s, s_l_y_c_s = stride(sparsity_layout_x_rwm)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [o_b,
                                    triton.cdiv(o_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        (wrap_triton(row_wise_add_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          sparsity_lut_x, s_lut_r, s_lut_r_s, s_lut_c_s,
          y, y_b, y_b_s, y_r_s, y_c_s,
          s_l_y_b, s_l_y_b_s, s_l_y_r_s,
          sparsity_reverse_x_lut_rwm,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs(),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def row_wise_add_kernel(x,
                        x_b, x_b_s, x_r_s, x_c_s,
                        s_lut_x, s_lut_x_r, s_lut_x_r_s, s_lut_x_c_s,
                        y, y_b, y_b_s, y_r_s, y_c_s,
                        s_l_y_b, s_l_y_b_s, s_l_y_r_s,
                        r_lut_y,
                        o,
                        o_b, o_b_s, o_r_s, o_c_s,
                        sparsity_block_size,
                        TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    pid_blk = tl.program_id(axis=0)
    pid_row = tl.program_id(axis=1)
    pid_col = tl.program_id(axis=2)

    # Get position of current sparsity block consisting of its batch and row index
    spa_val_idx = pid_blk * s_lut_x_r_s + tl.arange(0, 4) * s_lut_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(s_lut_x + spa_val_idx, mask=spa_val_msk)

    spa_bat_x = tl.sum(spa_val * (tl.arange(0, 4) == 0))
    spa_row_x = tl.sum(spa_val * (tl.arange(0, 4) == 1))
    spa_col_x = tl.sum(spa_val * (tl.arange(0, 4) == 2))

    # Get reverse sparsity indices for s
    rev_idx_spa_s_idx = (spa_bat_x * s_l_y_b_s +
                         spa_row_x * s_l_y_r_s)
    rev_idx_spa_s_msk = (rev_idx_spa_s_idx >= 0 and rev_idx_spa_s_idx < s_l_y_b * s_l_y_b_s)
    rev_idx_spa_s = tl.load(r_lut_y + rev_idx_spa_s_idx, mask=rev_idx_spa_s_msk).to(tl.int32)

    if rev_idx_spa_s == -1:
        tl.device_assert(False)
        return

    # Load x block
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * x_c_s)[None, :])
    blk_x_msk = (blk_x_idx >= 0 and
                 blk_x_idx < x_b * x_b_s)
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk)

    # Load sum block
    blk_s_idx = (rev_idx_spa_s * y_b_s +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * y_r_s)[:, None] +
                 (tl.arange(0, 1) * y_c_s)[None, :])
    blk_s_msk = (blk_s_idx >= 0 and
                 blk_s_idx < y_b * y_b_s)
    blk_s = tl.load(y + blk_s_idx, mask=blk_s_msk)

    # Compute exp
    buf = blk_x + tl.broadcast_to(blk_s, (TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE))

    # Store block
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.arange(0, TRITON_BLOCK_SIZE)) * o_c_s)[None, :])
    blk_o_msk = (blk_o_idx >= 0 and
                 blk_o_idx < o_b * o_b_s)
    tl.store(o + blk_o_idx, buf, mask=blk_o_msk)
