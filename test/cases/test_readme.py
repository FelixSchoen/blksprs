import torch
import blksprs as bs


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

    # Convert tensors to three-dimensional (dense) tensors since Triton can only handle tensors of exactly three dimensions
    x_dense, x_shape_original = bs.utils.do_shape_blocksparse(x)
    y_dense, y_shape_original = bs.utils.do_shape_blocksparse(y)

    # Create sparsity layouts from existing tensors
    sparsity_layout_x = bs.layouting.build_sparsity_layout(x_dense, sparsity_block_size)
    sparsity_layout_y = bs.layouting.build_sparsity_layout(y_dense, sparsity_block_size)

    # Create random sparsity layout for output tensor
    sparsity_layout_o = _get_random_sparsity_layout(b * h, m, n, sparsity_block_size, sparsity_percentage)

    # Convert tensors to sparse tensors for matrix multiplication
    x_sparse = bs.ops.to_sparse(x_dense, sparsity_layout_x, sparsity_block_size)
    y_sparse = bs.ops.to_sparse(y_dense, sparsity_layout_y, sparsity_block_size)

    # As of version 2.0, blksprs supports JIT compilation
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
    assert torch.allclose(actual_sparsity_layout_o.to(torch.int), sparsity_layout_o)

    # Convert output tensor back to original shape
    o = bs.utils.undo_shape_blocksparse(o_dense, x_shape_original)

    # Other available functions
    bs.ops.transpose(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.softmax(o_sparse, sparsity_layout_o, sparsity_block_size, flag_fused=False)
    bs.ops.softmax_fused(o_sparse, sparsity_layout_o,
                         sparsity_block_size)  # Significantly faster version that requires that rows of matrix fit into memory (default if flag is not set)
    bs.ops.misc.row_wise_sum(o_sparse, sparsity_layout_o, sparsity_block_size)
    bs.ops.misc.row_wise_max(o_sparse, sparsity_layout_o, sparsity_block_size)

    # Flash Attention
    seq_len, head_dim = 512, 64
    sparsity_block_size_attn = 128

    q = torch.randn(b, seq_len, h, head_dim, device="cuda")
    k = torch.randn(b, seq_len, h, head_dim, device="cuda")
    v = torch.randn(b, seq_len, h, head_dim, device="cuda")

    n_batches_attn = b * h
    n_seq_blocks = seq_len // sparsity_block_size_attn
    attention_layout = torch.tril(torch.ones(n_batches_attn, n_seq_blocks, n_seq_blocks, device="cuda", dtype=torch.bool))

    lut = bs.ops.flash_attention_build_lut(attention_layout, n_seq_blocks, n_seq_blocks)

    attn_out = bs.ops.flash_attention(q, k, v, attention_layout, sparsity_block_size_attn, lut=lut)

    assert attn_out.shape == (b, seq_len, h, head_dim)



def _get_random_sparsity_layout(b, m, n, sparsity_block_size, sparsity_percentage):
    """Helper function, creates a random sparsity layout for a given shape with a given percentage of blocks marked as sparse.

    """
    m_s = m // sparsity_block_size
    n_s = n // sparsity_block_size

    sparsity_layout = torch.ones(size=(b, m_s, n_s), device="cuda", dtype=torch.int)

    num_zero_elements = int(m_s * n_s * (sparsity_percentage / 100))
    for b_i in range(b):
        indices = torch.randperm(m_s * n_s)[:num_zero_elements]
        sparsity_layout[b_i, indices // n_s, indices % n_s] = 0

    return sparsity_layout


import pytest


@pytest.fixture(scope="session", autouse=True)
def setup():
    torch.manual_seed(0)
    torch.set_printoptions(edgeitems=64, linewidth=10000)
    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"
