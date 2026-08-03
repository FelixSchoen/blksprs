import torch
import triton
from torch import Tensor
from torch._library import triton_op
from torch._library.triton import wrap_triton
from triton import language as tl

from blksprs.utils.autotuning import get_autotune_configs, prune_autotune_configs
from blksprs.utils.blksprs_tensor import BlksprsTensor
from blksprs.utils.tools import build_layout_indices, stride, build_packed_indices, can_use_int32_indexing, \
    prepare_layout_cache, finalize_layout_cache
from blksprs.utils.validation import validate_contiguous, validate_dimensions, validate_device, \
    validate_sparsity, validate_dtype_int, validate_sparsity_block_size, ensure_contiguous, \
    validate_dimension, validate_indices, validate_sparsity_layout, validate_dtype_supported, \
    validate_distribution_shape


@torch.amp.custom_fwd(device_type="cuda")
def gather(src: BlksprsTensor, sparsity_layout_src: Tensor,
           dim: int,
           idx: BlksprsTensor, sparsity_layout_idx: Tensor,
           sparsity_block_size: int, layout_cache: dict | None = None) -> BlksprsTensor:
    """Gathers values from a compressed block-sparse tensor.

    Args:
        src (BlksprsTensor): The compressed source tensor.
        sparsity_layout_src (Tensor): The sparsity layout of ``src``.
        dim (int): The dimension along which to gather.
        idx (BlksprsTensor): The compressed indices specifying which source values to gather.
        sparsity_layout_idx (Tensor): The sparsity layout of ``idx``.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: The gathered values in compressed form.

    """
    src, idx = ensure_contiguous(src, idx)

    validate_dimensions(src, idx)
    validate_contiguous(src, idx)
    validate_dtype_int(idx)
    validate_dtype_supported(src)
    validate_device(src, idx)
    validate_sparsity(sparsity_block_size,
                      (src, sparsity_layout_src), (idx, sparsity_layout_idx))
    validate_sparsity_block_size(sparsity_block_size, src, idx)

    adjusted_dim = validate_dimension(dim)
    source_size = torch.Size((
        sparsity_layout_src.size(0),
        sparsity_layout_src.size(1) * sparsity_block_size,
        sparsity_layout_src.size(2) * sparsity_block_size,
    ))
    index_size = torch.Size((
        sparsity_layout_idx.size(0),
        sparsity_layout_idx.size(1) * sparsity_block_size,
        sparsity_layout_idx.size(2) * sparsity_block_size,
    ))
    validate_distribution_shape(index_size, source_size, adjusted_dim)
    validate_indices(idx, adjusted_dim, source_size)

    layout_cache = gather_build_layout_cache(layout_cache, sparsity_layout_src, sparsity_layout_idx)

    return BlksprsTensor.wrap(gather_forward(src, sparsity_layout_src, layout_cache["packed_indices_x"],
                                             adjusted_dim, idx, sparsity_layout_idx, layout_cache["layout_indices_i"],
                                             sparsity_block_size))


@triton_op("blksprs::gather_forward", mutates_args={})
def gather_forward(x: Tensor, sparsity_layout_x: Tensor, packed_indices_x: Tensor,
                   dim: int, i: Tensor, _: Tensor, layout_indices_i: Tensor,
                   sparsity_block_size: int) -> Tensor:
    with torch.no_grad():
        output = torch.zeros_like(i, dtype=x.dtype)

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        s_l_x_b, s_l_x_r, s_l_x_c = sparsity_layout_x.size()
        s_l_x_b_s, s_l_x_r_s, s_l_x_c_s = stride(sparsity_layout_x)
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = stride(i)
        lidx_i_r, lidx_i_c = layout_indices_i.size()
        lidx_i_r_s, lidx_i_c_s = stride(layout_indices_i)
        o_b, o_r, o_c = output.size()
        o_b_s, o_r_s, o_c_s = stride(output)

        def triton_grid(meta): return [o_b,
                                       triton.cdiv(
                                           o_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(o_c, meta["TRITON_BLOCK_SIZE"])]

        use_int64 = i.dtype == torch.int64 or not can_use_int32_indexing(
            x,
            sparsity_layout_x,
            packed_indices_x,
            i,
            output,
            layout_indices_i,
        )

        (wrap_triton(gather_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          s_l_x_b, s_l_x_r, s_l_x_c, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
          packed_indices_x,
          dim,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output,
          o_b, o_b_s, o_r_s, o_c_s,
          layout_indices_i, lidx_i_r, lidx_i_r_s, lidx_i_c_s,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output


def gather_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout_x, i, sparsity_layout_i = ctx.saved_tensors
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size

    return scatter_reduce(grad_output, sparsity_layout_i,
                          dim, i,
                          sparsity_layout_x, sparsity_block_size,
                          reduce_op="sum"), None, None, None, None, None, None, None


@triton.autotune(
    configs=get_autotune_configs("distribution"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def gather_kernel(x,
                  x_b, x_b_s, x_r_s, x_c_s,
                  s_l_x_b, s_l_x_r, s_l_x_c, s_l_x_b_s, s_l_x_r_s, s_l_x_c_s,
                  pidx_x,
                  dim,
                  i,
                  i_b, i_b_s, i_r_s, i_c_s,
                  o,
                  o_b, o_b_s, o_r_s, o_c_s,
                  lidx_o, lidx_o_r, lidx_o_r_s, lidx_o_c_s,
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

    # Load index values
    blk_i_idx = ((pid_blk * i_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * i_c_s)[None, :])
    blk_i_msk = ((blk_i_idx >= 0) &
                 (blk_i_idx < tl.cast(i_b, index_dtype) * i_b_s))
    blk_i = tl.cast(tl.load(i + blk_i_idx, mask=blk_i_msk, other=0), index_dtype)

    # Get indices of sparsity blocks and positions within the blocks
    pos_spa_blk_x = blk_i // sparsity_block_size
    pos_spa_int_x = blk_i % sparsity_block_size

    packed_dst_bat_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_bat_o, index_dtype), dtype=index_dtype)
    packed_dst_row_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_row_o, index_dtype), dtype=index_dtype)
    packed_dst_col_x = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_col_o, index_dtype), dtype=index_dtype)
    dst_row_x = (((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    dst_col_x = (((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    if dim == 0:
        packed_dst_bat_x = blk_i
    elif dim == 1:
        packed_dst_row_x = pos_spa_blk_x
        dst_row_x = tl.cast(pos_spa_int_x, index_dtype) * x_r_s
    elif dim == 2:
        packed_dst_col_x = pos_spa_blk_x
        dst_col_x = tl.cast(pos_spa_int_x, index_dtype) * x_c_s

    # Load packed indices for x
    packed_idx_x_idx = ((tl.cast(packed_dst_bat_x, index_dtype) * s_l_x_b_s) +
                         (tl.cast(packed_dst_row_x, index_dtype) * s_l_x_r_s) +
                         (tl.cast(packed_dst_col_x, index_dtype) * s_l_x_c_s))
    packed_idx_x_msk = ((packed_dst_bat_x >= 0) &
                         (tl.cast(packed_dst_bat_x, index_dtype) < tl.cast(s_l_x_b, index_dtype)) &
                         (packed_dst_row_x >= 0) &
                         (tl.cast(packed_dst_row_x, index_dtype) < tl.cast(s_l_x_r, index_dtype)) &
                         (packed_dst_col_x >= 0) &
                         (tl.cast(packed_dst_col_x, index_dtype) < tl.cast(s_l_x_c, index_dtype)))
    packed_idx_x = tl.cast(
        tl.load(pidx_x + packed_idx_x_idx, mask=packed_idx_x_msk, other=-1), index_dtype)

    # Load x values
    blk_x_idx = ((tl.cast(packed_idx_x, index_dtype) * x_b_s) +
                 dst_row_x +
                 dst_col_x)
    blk_x_msk = (((blk_x_idx >= 0) &
                  (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s)) &
                 (packed_idx_x >= 0))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Store output
    blk_o_idx = ((pid_blk * o_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * o_c_s)[None, :])
    blk_o_msk = (((blk_o_idx >= 0) &
                  (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s)) &
                 (packed_idx_x >= 0))
    tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)


def gather_build_layout_cache(layout_cache: dict | None, sparsity_layout_src: Tensor, sparsity_layout_idx: Tensor):
    layout_cache = prepare_layout_cache(
        layout_cache, "gather", sparsity_layout_src, sparsity_layout_idx)

    if "packed_indices_x" not in layout_cache:
        layout_cache["packed_indices_x"] = build_packed_indices(sparsity_layout_src)

    if "layout_indices_i" not in layout_cache:
        layout_indices_i = build_layout_indices(sparsity_layout_idx)
        layout_cache["layout_indices_i"] = layout_indices_i

    validate_contiguous(sparsity_layout_src, layout_cache["packed_indices_x"],
                        sparsity_layout_idx, layout_cache["layout_indices_i"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def gather_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, _, dim, i,
     sparsity_layout_i, _, sparsity_block_size) = inputs

    ctx.save_for_backward(sparsity_layout_x, i, sparsity_layout_i)
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size


gather_forward.register_autograd(
    gather_wrapper_backward, setup_context=gather_setup_context)


def scatter(src: BlksprsTensor, sparsity_layout_src: Tensor,
            dim: int,
            idx: BlksprsTensor,
            sparsity_layout_tgt: Tensor,
            sparsity_block_size: int, layout_cache: dict | None = None) -> BlksprsTensor:
    """Scatters values without reducing overlapping writes.

    This is a convenience wrapper around :func:`scatter_reduce` with ``reduce_op="none"``.

    Warning:
        When multiple source elements map to the same output position, the result is non-deterministic due
        to concurrent writes. Use :func:`scatter_reduce` with ``reduce_op="sum"`` for a defined reduction
        with overlapping indices. Floating-point summation may still be non-deterministic because it uses
        atomic GPU reductions.

        This non-reducing operation does not support a backward pass. Use
        :func:`scatter_reduce` with ``reduce_op="sum"`` when gradients are required.

    Args:
        src (BlksprsTensor): The compressed source tensor.
        sparsity_layout_src (Tensor): The sparsity layout shared by ``src`` and ``idx``.
        dim (int): The dimension along which to scatter.
        idx (BlksprsTensor): The compressed indices tensor, using ``sparsity_layout_src``.
        sparsity_layout_tgt (Tensor): The sparsity layout of the target tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: The scattered tensor in compressed form.

    """
    return scatter_reduce(src, sparsity_layout_src,
                          dim,
                          idx,
                          sparsity_layout_tgt,
                          sparsity_block_size,
                          reduce_op="none", layout_cache=layout_cache)


@torch.amp.custom_fwd(device_type="cuda")
def scatter_reduce(src: BlksprsTensor, sparsity_layout_src: Tensor,
                   dim: int,
                   idx: BlksprsTensor,
                   sparsity_layout_tgt: Tensor,
                   sparsity_block_size: int,
                   reduce_op: str = "sum", layout_cache: dict | None = None) -> BlksprsTensor:
    """Scatters values from a compressed block-sparse tensor.

    Args:
        src (BlksprsTensor): The compressed source tensor.
        sparsity_layout_src (Tensor): The sparsity layout shared by ``src`` and ``idx``.
        dim (int): The dimension along which to scatter.
        idx (BlksprsTensor): The compressed indices specifying target positions.
        sparsity_layout_tgt (Tensor): The sparsity layout of the target tensor.
        sparsity_block_size (int): The size of the sparsity blocks.
        reduce_op (str, optional): The reduction applied to overlapping writes. Supported values are ``"none"`` and
            ``"sum"``; only ``"sum"`` supports a backward pass (default ``"sum"``).
        layout_cache (dict, optional): Reusable layout metadata cache (default ``None``).

    Returns:
        BlksprsTensor: The scattered values in compressed form.

    """
    src, idx = ensure_contiguous(src, idx)

    validate_dimensions(src, idx)
    validate_contiguous(src, idx)
    validate_dtype_int(idx)
    validate_dtype_supported(src)
    validate_device(src, idx, sparsity_layout_tgt)
    validate_sparsity(sparsity_block_size, (src, sparsity_layout_src),
                      (idx, sparsity_layout_src))  # idx shares src's layout
    validate_sparsity_layout(sparsity_layout_tgt)
    validate_sparsity_block_size(sparsity_block_size, src, idx)

    if reduce_op not in ["none", "sum"]:
        raise ValueError(f"Reduction operation '{reduce_op}' is not supported")
    if reduce_op == "sum" and src.dtype not in (
            torch.int32, torch.int64,
            torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Reduction operation 'sum' does not support {src.dtype} source tensors")

    adjusted_dim = validate_dimension(dim)
    target_size = torch.Size((
        sparsity_layout_tgt.size(0),
        sparsity_layout_tgt.size(1) * sparsity_block_size,
        sparsity_layout_tgt.size(2) * sparsity_block_size,
    ))
    source_size = torch.Size((
        sparsity_layout_src.size(0),
        sparsity_layout_src.size(1) * sparsity_block_size,
        sparsity_layout_src.size(2) * sparsity_block_size,
    ))
    validate_distribution_shape(source_size, target_size, adjusted_dim)
    validate_indices(idx, adjusted_dim, target_size)

    layout_cache = scatter_reduce_build_layout_cache(
        layout_cache, sparsity_layout_src, sparsity_layout_tgt)

    return BlksprsTensor.wrap(scatter_reduce_forward(src, sparsity_layout_src, layout_cache["layout_indices_x"],
                                                     adjusted_dim, idx,
                                                     sparsity_layout_tgt, layout_cache["packed_indices_o"],
                                                     sparsity_block_size, layout_cache["n_sparse_blocks"],
                                                     reduce_op))


@triton_op("blksprs::scatter_reduce_forward", mutates_args={})
def scatter_reduce_forward(x: Tensor, _: Tensor, layout_indices_x: Tensor,
                           dim: int, i: Tensor,
                           sparsity_layout_o: Tensor, packed_indices_o: Tensor,
                           sparsity_block_size: int, n_sparse_blocks: int,
                           reduce_op: str) -> Tensor:
    with torch.no_grad():
        accumulator_dtype = (
            torch.float32
            if reduce_op == "sum" and x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )
        output_accumulator = torch.zeros(
            size=(n_sparse_blocks, sparsity_block_size, sparsity_block_size),
            dtype=accumulator_dtype,
            device=x.device,
        )

        x_b, x_r, x_c = x.size()
        x_b_s, x_r_s, x_c_s = stride(x)
        lidx_x_r, lidx_x_c = layout_indices_x.size()
        lidx_x_r_s, lidx_x_c_s = stride(layout_indices_x)
        i_b, i_r, i_c = i.size()
        i_b_s, i_r_s, i_c_s = stride(i)
        o_b, o_r, o_c = output_accumulator.size()
        o_b_s, o_r_s, o_c_s = stride(output_accumulator)
        s_l_o_b, s_l_o_r, s_l_o_c = sparsity_layout_o.size()
        s_l_o_b_s, s_l_o_r_s, s_l_o_c_s = stride(sparsity_layout_o)

        def triton_grid(meta): return [x_b,
                                       triton.cdiv(
                                           x_r, meta["TRITON_BLOCK_SIZE"]),
                                       triton.cdiv(x_c, meta["TRITON_BLOCK_SIZE"])]

        reduce_op_ind = 0
        if reduce_op == "sum":
            reduce_op_ind = 1

        use_int64 = i.dtype == torch.int64 or not can_use_int32_indexing(
            x,
            layout_indices_x,
            i,
            output_accumulator,
            sparsity_layout_o,
            packed_indices_o,
        )

        (wrap_triton(scatter_reduce_kernel)[triton_grid]
         (x,
          x_b, x_b_s, x_r_s, x_c_s,
          layout_indices_x, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
          dim,
          i,
          i_b, i_b_s, i_r_s, i_c_s,
          output_accumulator,
          o_b, o_b_s,
          s_l_o_b, s_l_o_r, s_l_o_c, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
          packed_indices_o,
          reduce_op_ind,
          sparsity_block_size,
          USE_INT64=use_int64))

        return output_accumulator.to(x.dtype)


def scatter_reduce_wrapper_backward(ctx, grad_output):
    grad_output = grad_output.contiguous()
    sparsity_layout_x, i, sparsity_layout_o = ctx.saved_tensors
    dim = ctx.dim
    sparsity_block_size = ctx.sparsity_block_size
    reduce_op = ctx.reduce_op

    if reduce_op == "sum":
        return gather(grad_output, sparsity_layout_o, dim, i, sparsity_layout_x,
                      sparsity_block_size), None, None, None, None, None, None, None, None, None
    else:
        raise ValueError(
            f"Reduction operation '{reduce_op}' does not support backward pass")


@triton.autotune(
    configs=get_autotune_configs("distribution"),
    key=["sparsity_block_size"],
    prune_configs_by={"early_config_prune": prune_autotune_configs},
    reset_to_zero=["o"]
)
@triton.jit
def scatter_reduce_kernel(x,
                          x_b, x_b_s, x_r_s, x_c_s,
                          lidx_x, lidx_x_r, lidx_x_r_s, lidx_x_c_s,
                          dim,
                          i,
                          i_b, i_b_s, i_r_s, i_c_s,
                          o,
                          o_b, o_b_s,
                          s_l_o_b, s_l_o_r, s_l_o_c, s_l_o_b_s, s_l_o_r_s, s_l_o_c_s,
                          pidx_o,
                          reduce_op_ind: tl.constexpr,
                          sparsity_block_size,
                          USE_INT64: tl.constexpr,
                          TRITON_BLOCK_SIZE: tl.constexpr) -> None:
    # Get triton block indices
    index_dtype = tl.int64 if USE_INT64 else tl.int32
    pid_blk = tl.cast(tl.program_id(axis=0), index_dtype)
    pid_row = tl.cast(tl.program_id(axis=1), index_dtype)
    pid_col = tl.cast(tl.program_id(axis=2), index_dtype)

    # Get position of current sparsity block consisting of its batch, row, and column index
    spa_val_idx = pid_blk * lidx_x_r_s + tl.cast(tl.arange(0, 4), index_dtype) * lidx_x_c_s
    spa_val_msk = (tl.arange(0, 4) < 3)
    spa_val = tl.load(lidx_x + spa_val_idx, mask=spa_val_msk, other=0)

    spa_bat_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 0)), index_dtype)
    spa_row_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 1)), index_dtype)
    spa_col_x = tl.cast(tl.sum(spa_val * (tl.arange(0, 4) == 2)), index_dtype)

    # Load x values
    blk_x_idx = ((pid_blk * x_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :])
    blk_x_msk = ((blk_x_idx >= 0) &
                 (blk_x_idx < tl.cast(x_b, index_dtype) * x_b_s))
    blk_x = tl.load(x + blk_x_idx, mask=blk_x_msk, other=0)

    # Load index values
    blk_i_idx = ((pid_blk * i_b_s) +
                 ((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * i_r_s)[:, None] +
                 ((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * i_c_s)[None, :])
    blk_i_msk = ((blk_i_idx >= 0) &
                 (blk_i_idx < tl.cast(i_b, index_dtype) * i_b_s))
    blk_i = tl.cast(tl.load(i + blk_i_idx, mask=blk_i_msk, other=0), index_dtype)

    # Get indices of sparsity blocks and positions within the blocks
    pos_spa_blk_x = blk_i // sparsity_block_size
    pos_spa_int_x = blk_i % sparsity_block_size

    packed_dst_bat_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_bat_x, index_dtype), dtype=index_dtype)
    packed_dst_row_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_row_x, index_dtype), dtype=index_dtype)
    packed_dst_col_o = tl.full((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE), tl.cast(
        spa_col_x, index_dtype), dtype=index_dtype)
    dst_row_o = (((pid_row * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_r_s)[:, None]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    dst_col_o = (((pid_col * TRITON_BLOCK_SIZE + tl.cast(tl.arange(0, TRITON_BLOCK_SIZE), index_dtype)) * x_c_s)[None, :]
                 .broadcast_to((TRITON_BLOCK_SIZE, TRITON_BLOCK_SIZE)))
    if dim == 0:
        packed_dst_bat_o = blk_i
    elif dim == 1:
        packed_dst_row_o = pos_spa_blk_x
        dst_row_o = tl.cast(pos_spa_int_x, index_dtype) * x_r_s
    elif dim == 2:
        packed_dst_col_o = pos_spa_blk_x
        dst_col_o = tl.cast(pos_spa_int_x, index_dtype) * x_c_s

    # Load packed indices for o
    packed_idx_o_idx = ((tl.cast(packed_dst_bat_o, index_dtype) * s_l_o_b_s) +
                         (tl.cast(packed_dst_row_o, index_dtype) * s_l_o_r_s) +
                         (tl.cast(packed_dst_col_o, index_dtype) * s_l_o_c_s))
    packed_idx_o_msk = ((packed_dst_bat_o >= 0) &
                         (tl.cast(packed_dst_bat_o, index_dtype) < tl.cast(s_l_o_b, index_dtype)) &
                         (packed_dst_row_o >= 0) &
                         (tl.cast(packed_dst_row_o, index_dtype) < tl.cast(s_l_o_r, index_dtype)) &
                         (packed_dst_col_o >= 0) &
                         (tl.cast(packed_dst_col_o, index_dtype) < tl.cast(s_l_o_c, index_dtype)))
    packed_idx_o = tl.cast(
        tl.load(pidx_o + packed_idx_o_idx, mask=packed_idx_o_msk, other=-1), index_dtype)

    # Store output
    blk_o_idx = ((tl.cast(packed_idx_o, index_dtype) * o_b_s) +
                 dst_row_o +
                 dst_col_o)
    blk_o_msk = (((blk_o_idx >= 0) &
                  (blk_o_idx < tl.cast(o_b, index_dtype) * o_b_s)) &
                 (packed_idx_o >= 0))

    if reduce_op_ind == 0:
        tl.store(o + blk_o_idx, blk_x, mask=blk_o_msk)
    elif reduce_op_ind == 1:
        tl.atomic_add(o + blk_o_idx, blk_x, mask=blk_o_msk)


def scatter_reduce_build_layout_cache(layout_cache: dict | None, sparsity_layout_src: Tensor, sparsity_layout_tgt: Tensor):
    layout_cache = prepare_layout_cache(
        layout_cache, "scatter_reduce", sparsity_layout_src, sparsity_layout_tgt)

    if "layout_indices_x" not in layout_cache:
        layout_indices_x = build_layout_indices(sparsity_layout_src)
        layout_cache["layout_indices_x"] = layout_indices_x

    if "packed_indices_o" not in layout_cache:
        layout_cache["packed_indices_o"] = build_packed_indices(sparsity_layout_tgt)

    if "n_sparse_blocks" not in layout_cache:
        n_sparse_blocks = torch.sum(sparsity_layout_tgt.to(torch.int)).item()
        layout_cache["n_sparse_blocks"] = n_sparse_blocks

    validate_contiguous(sparsity_layout_src, layout_cache["layout_indices_x"],
                        sparsity_layout_tgt, layout_cache["packed_indices_o"])

    return finalize_layout_cache(layout_cache)


# noinspection PyUnusedLocal
def scatter_reduce_setup_context(ctx, inputs, output):
    (_, sparsity_layout_x, _, dim, i, sparsity_layout_o,
     _, sparsity_block_size, _, reduce_op) = inputs

    ctx.save_for_backward(sparsity_layout_x, i, sparsity_layout_o)
    ctx.dim = dim
    ctx.sparsity_block_size = sparsity_block_size
    ctx.reduce_op = reduce_op


scatter_reduce_forward.register_autograd(
    scatter_reduce_wrapper_backward, setup_context=scatter_reduce_setup_context)
