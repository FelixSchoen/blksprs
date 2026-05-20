import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.tools import stride, can_use_int32_indexing


@triton_op("blksprs::flow_pull_forward", mutates_args={})
def flow_pull_forward(x: Tensor, sparsity_layout_o: Tensor,
                      layout_indices: Tensor, packed_indices: Tensor,
                      sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)
        lidx_r, lidx_c = layout_indices.size()
        lidx_r_s, lidx_c_s = stride(layout_indices)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            output,
            sparsity_layout_o,
            layout_indices,
            packed_indices,
        )

        (wrap_triton(flow_pull_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
          layout_indices, lidx_r, lidx_r_s, lidx_c_s,
          packed_indices,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flow"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def flow_pull_kernel(x,
                     x_b, x_b_s, x_r_s, x_c_s,
                     o,
                     o_b, o_b_s, o_r_s, o_c_s,
                     s_l_o_b, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                     lidx, lidx_r, lidx_r_s, lidx_c_s,
                     pidx,
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

    # Load packed index
    packed_idx_idx = (spa_bat * s_l_o_b_s +
                       spa_row * s_l_o_r_s +
                       spa_col * s_l_o_c_s)
    packed_idx_msk = ((packed_idx_idx >= 0) &
                       (packed_idx_idx < tl.cast(s_l_o_b, index_dtype) * s_l_o_b_s))
    packed_idx = tl.cast(tl.load(pidx + packed_idx_idx,
                          mask=packed_idx_msk, other=-1), tl.int32)

    if packed_idx >= 0:
        blk_x_idx = (tl.cast(packed_idx, index_dtype) * x_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_x_msk = ((blk_x_idx >= 0) &
                     (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

        blk_o_idx = (pid_blk * o_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
        blk_o_msk = ((blk_o_idx >= 0) &
                     (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


@triton_op("blksprs::flow_push_forward", mutates_args={})
def flow_push_forward(x: Tensor, sparsity_layout_x: Tensor, layout_indices: Tensor, packed_indices: Tensor,
                      sparsity_block_size: int, n_sparse_blocks: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros(size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
                             dtype=x.dtype, device=x.device)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        lidx_r, lidx_c = layout_indices.size()
        lidx_r_s, lidx_c_s = stride(layout_indices)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        def triton_grid(meta): return [x_b,
                                       triton.cdiv(
                                           x_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = not can_use_int32_indexing(
            x,
            sparsity_layout_x,
            layout_indices,
            packed_indices,
            output,
        )

        (wrap_triton(flow_push_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          layout_indices, lidx_r, lidx_r_s, lidx_c_s,
          packed_indices,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


# noinspection PyUnusedLocal
@triton.autotune(
    configs=get_autotune_configs("flow"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def flow_push_kernel(x,
                     x_b, x_b_s, x_r_s, x_c_s,
                     s_l_x_b, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                     lidx, lidx_r, lidx_r_s, lidx_c_s,
                     pidx,
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

    # Get sparsity index of current input block consisting of its batch, row, and column index
    spa_val_idx = pid_blk * lidx_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Get packed index
    packed_idx_idx = (spa_bat * s_l_x_b_s +
                       spa_row * s_l_x_r_s +
                       spa_col * s_l_x_c_s)
    packed_idx_msk = ((packed_idx_idx >= 0) &
                       (packed_idx_idx < tl.cast(s_l_x_b, index_dtype) * s_l_x_b_s))
    packed_idx = tl.cast(tl.load(pidx + packed_idx_idx,
                          mask=packed_idx_msk, other=-1), tl.int32)

    if packed_idx >= 0:
        blk_x_idx = (pid_blk * x_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
        blk_x_msk = ((blk_x_idx >= 0) &
                     (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
        blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

        blk_o_idx = (tl.cast(packed_idx, index_dtype) * o_b_s +
                     ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                     ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
        blk_o_msk = ((blk_o_idx >= 0) &
                     (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s))
        tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)
