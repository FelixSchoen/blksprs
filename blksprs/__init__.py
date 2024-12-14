from blksprs.utils.blksprs_tensor import BlksprsTensor


class ops:
    from blksprs.ops.conversion import to_dense, to_sparse, from_blksprs, to_blksprs, adapt_layout
    from blksprs.ops.distribution import gather, scatter, scatter_reduce
    from blksprs.ops.matmul import matmul
    from blksprs.ops.softmax import softmax
    from blksprs.ops.transpose import transpose
    from blksprs.ops.repeat import repeat, repeat_interleave
    from blksprs.ops.partitioning import split, merge

    class misc:
        from blksprs.ops.misc.row_wise import row_wise_sum, row_wise_max, row_wise_add, row_wise_sub
        from blksprs.ops.misc.broadcast_ops import broadcast_add, broadcast_sub
        from blksprs.ops.misc.exp import exp


class layouting:
    from blksprs.layouting.distribution_layout import build_distribution_layout
    from blksprs.layouting.sparsity_layout import build_sparsity_layout, build_sparsity_layout_adaption, \
        build_sparsity_layout_matmul, build_sparsity_layout_matmul_fast
    from blksprs.utils.layout_utils import build_full_sparsity_layout


class utils:
    from blksprs.utils.processing import apply_torch_linear, apply_torch_normalisation, apply_torch_dropout, \
        apply_function_applicable_row_wise
    from blksprs.utils.tools import do_shape_blocksparse, undo_shape_blocksparse
    from blksprs.utils.validation import disable_validation

    class validation:
        from blksprs.utils.validation import disable_validation
        from blksprs.utils.validation import validate_dimensions, validate_contiguous, validate_dtype_float, \
            validate_dtype_int, validate_device, validate_sparsity, validate_sparsity_dense, \
            validate_sparsity_block_size, \
            validate_triton_block_size
