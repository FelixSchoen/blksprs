import blksprs as bs
import torch


def test_readme():
    # Set up parameters (batch size, number of heads, dimensions for matrices (m, k) and (n, k))
    b, h, m, n, k = 2, 4, 64, 64, 16

    # Percentage of blocks that will be sparse in the output for demonstration purposes
    sparsity_percentage = 25

    # Must be a power of two, greater than or equal to 16 for matmul, and divide m, n, and k
    sparsity_block_size = 16

    # Initialise random (dense) tensors
    x = torch.randn(size=(b, h, m, k), device="cuda")
    y = torch.randn(size=(b, h, n, k), device="cuda").transpose(-1, -2).contiguous()

    # Flatten leading dimensions because BLK-SPRS operations accept three-dimensional tensors
    x_dense, x_shape_original = bs.utils.do_shape_blocksparse(x)
    y_dense, _ = bs.utils.do_shape_blocksparse(y)

    # Create sparsity layouts from existing tensors
    sparsity_layout_x = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)
    sparsity_layout_y = bs.layouting.build_sparsity_layout(y_dense, sparsity_block_size)

    # Create random sparsity layout for output tensor
    sparsity_layout_o = _get_random_sparsity_layout(b * h, m, n, sparsity_block_size, sparsity_percentage)

    # Convert tensors to sparse tensors for matrix multiplication
    x_sparse = bs.ops.to_sparse(x_dense, sparsity_layout_x, sparsity_block_size)
    y_sparse = bs.ops.to_sparse(y_dense, sparsity_layout_y, sparsity_block_size)

    # Default torch.compile mode is supported for correctness. Public wrapper
    # validation and cache preparation may graph-break, so benchmark the full workload.
    matmul_compiled = torch.compile(bs.ops.matmul)

    # Perform matrix multiplication
    o_sparse = matmul_compiled(x_sparse, sparsity_layout_x,
                               y_sparse, sparsity_layout_y,
                               sparsity_layout_o, sparsity_block_size)

    # Apply element-wise operation
    o_sparse = torch.add(o_sparse, 1)

    o_dense = bs.ops.to_dense(o_sparse, sparsity_layout_o, sparsity_block_size)

    # Sanity check
    o_torch = torch.matmul(x_dense, y_dense)
    o_torch = torch.add(o_torch, 1)

    # Perform round trip to set sparse blocks to 0
    o_torch_round_trip = bs.ops.to_dense(
        bs.ops.to_sparse(o_torch, sparsity_layout_o, sparsity_block_size),
        sparsity_layout_o, sparsity_block_size, fill_value=0)

    # Assert that the output is correct
    assert torch.allclose(o_dense, o_torch_round_trip, atol=2e-2)  # Note that small numerical differences are expected

    # Assert that the output has the correct sparsity layout
    actual_sparsity_layout_o = bs.layouting.build_sparsity_layout(o_dense, sparsity_block_size)
    assert torch.equal(actual_sparsity_layout_o, sparsity_layout_o)

    # Convert output tensor back to original shape
    o = bs.utils.undo_shape_blocksparse(o_dense, x_shape_original)
    assert o.shape == (b, h, m, n)

    # Other available functions
    bs.ops.transpose(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.softmax(o_sparse, sparsity_layout_o, sparsity_block_size, flag_fused=False)
    bs.ops.softmax_fused(o_sparse, sparsity_layout_o,
                         sparsity_block_size)  # Explicit fused execution; raises above 131,072 padded row elements
    bs.ops.softmax(o_sparse, sparsity_layout_o,
                   sparsity_block_size)  # Fused by default with automatic fallback for oversized rows
    bs.ops.misc.row_wise_sum(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.misc.row_wise_max(o_sparse, sparsity_layout_o, sparsity_block_size)

    # Flash Attention
    seq_len, head_dim = 512, 64
    sparsity_block_size_attn = 64

    q = torch.randn(b, seq_len, h, head_dim, device="cuda")
    k = torch.randn(b, seq_len, h, head_dim, device="cuda")
    v = torch.randn(b, seq_len, h, head_dim, device="cuda")

    # Flash Attention expects (batch * heads, seq_len, head_dim)
    q_dense = q.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()
    k_dense = k.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()
    v_dense = v.transpose(1, 2).reshape(-1, seq_len, head_dim).contiguous()

    n_batches_attn = b * h
    n_seq_blocks = seq_len // sparsity_block_size_attn
    n_head_blocks = head_dim // sparsity_block_size_attn

    sparsity_layout_qkv = torch.ones(
        n_batches_attn, n_seq_blocks, n_head_blocks,
        device="cuda", dtype=torch.bool,
    )
    attention_layout = torch.tril(torch.ones(
        n_batches_attn, n_seq_blocks, n_seq_blocks,
        device="cuda", dtype=torch.bool,
    ))
    sparsity_layout_o = bs.layouting.build_sparsity_layout_matmul(attention_layout, sparsity_layout_qkv)

    q_sparse = bs.ops.to_sparse(q_dense, sparsity_layout_qkv, sparsity_block_size_attn)
    k_sparse = bs.ops.to_sparse(k_dense, sparsity_layout_qkv, sparsity_block_size_attn)
    v_sparse = bs.ops.to_sparse(v_dense, sparsity_layout_qkv, sparsity_block_size_attn)

    # Pre-build reusable layout cache data for repeated calls with the same layouts
    flash_layout_cache = bs.ops.flash_attention_build_layout_cache(
        attention_layout,
        sparsity_layout_q=sparsity_layout_qkv,
        sparsity_layout_k=sparsity_layout_qkv,
        sparsity_layout_v=sparsity_layout_qkv,
        n_seq_blocks_q=n_seq_blocks,
        n_seq_blocks_k=n_seq_blocks,
        n_head_blocks=n_head_blocks,
        sparsity_layout_o=sparsity_layout_o,
    )

    attn_out_sparse = bs.ops.flash_attention(
        q_sparse, sparsity_layout_qkv,
        k_sparse, sparsity_layout_qkv,
        v_sparse, sparsity_layout_qkv,
        attention_layout, sparsity_block_size_attn,
        layout_cache=flash_layout_cache,
        sparsity_layout_o=sparsity_layout_o,
    )
    attn_out_dense = bs.ops.to_dense(attn_out_sparse, sparsity_layout_o, sparsity_block_size_attn)
    attn_out = attn_out_dense.reshape(b, h, seq_len, head_dim).transpose(1, 2).contiguous()

    assert attn_out.shape == (b, seq_len, h, head_dim)


def _get_random_sparsity_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    """Creates a random sparsity layout with the requested percentage of inactive blocks."""
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), device="cuda", dtype=torch.bool)

    num_zero_elements = int(m_s * n_s * (sparsity_percentage / 100))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = False

    return sparsity_layout
