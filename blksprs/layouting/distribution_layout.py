import typing

import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import stride, can_use_int32_indexing
from blksprs.utils.validation import validate_dimensions, validate_device, \
    validate_contiguous, validate_sparsity, validate_sparsity_block_size


@torch.amp.custom_fwd(device_type="cuda")
def build_distribution_layout(indices: BlksprsTensor, sparsity_layout_indices: Tensor,
                              dim: int, size_target: torch.Size,
                              sparsity_block_size: int) -> Tensor:
    """Builds the sparsity layout of either the source of a gather or the target of a scatter operation.

    Args:
        indices (BlksprsTensor): The block-sparse indices tensor in compressed form used for the gather or scatter operation.
        sparsity_layout_indices (Tensor): The sparsity layout of the indices block-sparse tensor.
        dim (int): The dimension along which the operation is conducted.
        size_target (torch.Size): The size of the block-sparse target tensor in regular form.
        sparsity_block_size (int): The size of the sparsity blocks.

    Returns:
        Tensor: The sparsity layout of the source or target tensor.

    """
    validate_dimensions(indices)
    validate_contiguous(indices, sparsity_layout_indices)
    validate_device(indices, sparsity_layout_indices)
    validate_sparsity(sparsity_block_size, (indices, sparsity_layout_indices))
    validate_sparsity_block_size(sparsity_block_size, indices, size_target)

    layout_indices_i = torch.nonzero(sparsity_layout_indices).contiguous()

    adjusted_dim = dim % 3

    return build_distribution_layout_operation(indices, layout_indices_i, adjusted_dim, size_target, sparsity_block_size)


@triton_op("blksprs::build_distribution_layout", mutates_args={})
def build_distribution_layout_operation(indices: Tensor, layout_indices_i: Tensor,
                                        adjusted_dim: int, size_target: typing.List[int],
                                        sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size_target[0], size_target[1] // sparsity_block_size,
                             size_target[2] // sparsity_block_size,
                             dtype=torch.bool, device=indices.device)

        i_b, i_r, i_c = indices.size()
        i_b_s, i_r_s, i_c_s = stride(indices)
        lidx_i_r, lidx_i_c = layout_indices_i.size()
        lidx_i_r_s, lidx_i_c_s = stride(layout_indices_i)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        triton_grid = lambda meta: [i_b,
                                    triton.cdiv(i_r, meta["TRITON_BLOCK_SIZE"]),
                                    triton.cdiv(i_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(indices, layout_indices_i, output)

        (wrap_triton(build_distribution_layout_kernel)[triton_grid]
         (indices,
          i_b, i_r, i_c, i_b_s, i_r_s, i_c_s,
          layout_indices_i,
          lidx_i_r, lidx_i_r_s, lidx_i_c_s,
          adjusted_dim,
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
def build_distribution_layout_kernel(i,
                                     i_b, i_r, i_c, i_b_s, i_r_s, i_c_s,
                                     lidx_i,
                                     lidx_i_r, lidx_i_r_s, lidx_i_c_s,
                                     dim,
                                     o,
                                     o_b, o_r, o_c, o_b_s, o_r_s, o_c_s,
                                     sparsity_block_size,
                                     USE_INT64: tl.constexpr,
                                     TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    # Get triton block indices
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch, row, and column index
    spa_bat_i_idx = (pid_blk * lidx_i_r_s + 0 * lidx_i_c_s)
    spa_bat_i_msk = ((spa_bat_i_idx >= 0) &
                     (spa_bat_i_idx < tl.cast(lidx_i_r, index_dtype) * lidx_i_r_s))
    spa_bat_i = tl.cast(tl.load(lidx_i + spa_bat_i_idx, mask=spa_bat_i_msk, other=0), index_dtype)

    spa_row_i_idx = (pid_blk * lidx_i_r_s + 1 * lidx_i_c_s)
    spa_row_i_msk = ((spa_row_i_idx >= 0) &
                     (spa_row_i_idx < tl.cast(lidx_i_r, index_dtype) * lidx_i_r_s))
    spa_row_i = tl.cast(tl.load(lidx_i + spa_row_i_idx, mask=spa_row_i_msk, other=0), index_dtype)

    spa_col_i_idx = (pid_blk * lidx_i_r_s + 2 * lidx_i_c_s)
    spa_col_i_msk = ((spa_col_i_idx >= 0) &
                     (spa_col_i_idx < tl.cast(lidx_i_r, index_dtype) * lidx_i_r_s))
    spa_col_i = tl.cast(tl.load(lidx_i + spa_col_i_idx, mask=spa_col_i_msk, other=0), index_dtype)

    row_offsets = pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    col_offsets = pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)
    blk_i_idx = (pid_blk * i_b_s +
                 (row_offsets * i_r_s)[:, None] +
                 (col_offsets * i_c_s)[None, :])
    blk_i_msk = ((pid_blk < tl.cast(i_b, index_dtype)) &
                 (row_offsets[:, None] < tl.cast(i_r, index_dtype)) &
                 (col_offsets[None, :] < tl.cast(i_c, index_dtype)))
    blk_i = tl.cast(tl.load(i + blk_i_idx, mask=blk_i_msk, other=0), index_dtype)

    dst_bat_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_bat_i, dtype=index_dtype)
    dst_row_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_row_i, dtype=index_dtype)
    dst_col_idx = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), spa_col_i, dtype=index_dtype)
    if dim == 0:
        dst_bat_idx = blk_i
    elif dim == 1:
        dst_row_idx = blk_i // sparsity_block_size
    elif dim == 2:
        dst_col_idx = blk_i // sparsity_block_size

    blk_v = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), 1, dtype=tl.int1)

    blk_o_idx = ((dst_bat_idx * o_b_s) +
                 (dst_row_idx * o_r_s) +
                 (dst_col_idx * o_c_s))
    blk_o_msk = (blk_i_msk &
                 (dst_bat_idx >= 0) &
                 (dst_bat_idx < tl.cast(o_b, index_dtype)) &
                 (dst_row_idx >= 0) &
                 (dst_row_idx < tl.cast(o_r, index_dtype)) &
                 (dst_col_idx >= 0) &
                 (dst_col_idx < tl.cast(o_c, index_dtype)))
    tl.store(o + blk_o_idx, blk_v, mask=blk_o_msk)
