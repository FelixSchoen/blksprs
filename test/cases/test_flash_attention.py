"""Comprehensive tests for block-sparse Flash Attention."""

import os

os.environ["BLKSPRS_AUTOTUNE"] = "TEST"

import random

import pytest
import torch
import torch.nn.functional as F

import blksprs as bs

# Device setup
DEVICE = torch.device("cuda:0")

# Test configurations
# (batch, seq_q, seq_k, n_heads, head_dim, block_size, attn_sparsity_pct, use_causal_mask)
FLASH_ATTENTION_TEST_CONFIGS = [
    # Dense attention
    (2, 64, 64, 4, 64, 16, 0, False),      # Dense, no mask
    (2, 64, 64, 4, 64, 16, 0, True),       # Dense, causal mask
    (2, 64, 64, 4, 64, 32, 0, False),      # Different block size

    # Sparse attention
    (2, 128, 128, 4, 64, 16, 0.5, False),  # 50% sparse
    (2, 128, 128, 4, 64, 16, 0.75, False), # 75% sparse
    (2, 128, 128, 4, 64, 32, 0.75, False), # 75% sparse, different block

    # Sparse + causal mask
    (2, 128, 128, 4, 64, 16, 0.5, True),
    (2, 128, 128, 4, 64, 32, 0.75, True),

    # Different head_dim / block_size ratios
    (2, 128, 128, 4, 64, 16, 0.5, False),  # head_dim=64, bs=16 (4 head blocks)
    (2, 128, 128, 4, 64, 32, 0.5, False),  # head_dim=64, bs=32 (2 head blocks)
    (2, 128, 128, 4, 128, 32, 0.5, False), # head_dim=128, bs=32 (4 head blocks)

    # Asymmetric Q/K lengths (cross-attention)
    (2, 256, 128, 4, 64, 32, 0.5, False),
    (2, 128, 256, 4, 64, 32, 0.5, False),

    # Edge cases
    (1, 64, 64, 1, 64, 16, 0, False),      # Single batch, single head
    (2, 32, 32, 4, 32, 16, 0, False),      # Small sequence
    (2, 32, 32, 4, 32, 16, 0, True),       # Small sequence, causal mask

    # Larger configurations
    (2, 256, 256, 4, 64, 32, 0.5, False),
    (2, 256, 256, 4, 64, 32, 0.5, True),
]

# Tolerances
ATOL = 5e-2
RTOL = 5e-2

# Seed
SEED = 0
RANDOM_SEED = True


@pytest.fixture(scope="session", autouse=True)
def setup():
    global SEED
    global RANDOM_SEED

    if RANDOM_SEED:
        seed = random.randint(0, 2 ** 32 - 1)
        SEED = seed
        print("Using randomly generated seed...")
    else:
        seed = SEED
        print("Notice: Not using randomly generated seed!")

    print("Seed:", seed)
    torch.manual_seed(seed)
    torch.set_printoptions(edgeitems=64, linewidth=10000)

    yield

    print("Seed:", seed)


def _get_attention_layout(n_batches: int, n_seq_q: int, n_seq_k: int,
                          sparsity_pct: float, device=DEVICE) -> torch.Tensor:
    """Generate a random attention sparsity layout."""
    attention_layout = torch.ones(n_batches, n_seq_q, n_seq_k,
                                  dtype=torch.bool, device=device)

    num_zero_elements = int(n_seq_q * n_seq_k * sparsity_pct)
    for b in range(n_batches):
        indices = torch.randperm(n_seq_q * n_seq_k)[:num_zero_elements]
        attention_layout[b, indices // n_seq_k, indices % n_seq_k] = False

    return attention_layout


def _build_causal_mask(n_batches: int, seq_q: int, seq_k: int, device=DEVICE) -> torch.Tensor:
    """Build a causal attention mask.
    
    Returns:
        Boolean tensor (n_batches, seq_q, seq_k) where True = masked (upper triangular)
    """
    # Upper triangular (positions to mask out)
    mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool, device=device), diagonal=1)
    # Expand for batches
    return mask.unsqueeze(0).expand(n_batches, -1, -1)


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         attention_layout: torch.Tensor, block_size: int,
                         attention_mask: torch.Tensor = None, scale: float = None) -> torch.Tensor:
    """Compute reference attention using PyTorch.

    Args:
        q, k, v: Input tensors (batch, seq, n_heads, head_dim)
        attention_layout: Block attention mask (batch*n_heads, n_seq_q, n_seq_k)
        block_size: Sparsity block size
        attention_mask: Optional boolean mask (batch*n_heads, seq_q, seq_k), True=masked
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor (batch, seq_q, n_heads, head_dim)
    """
    batch, seq_q, n_heads, head_dim = q.shape
    _, seq_k, _, _ = k.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Reshape for attention: (batch, n_heads, seq, head_dim)
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)

    # Compute attention scores
    attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Apply block sparsity mask
    attn_layout_expanded = attention_layout.reshape(batch, n_heads,
                                                     seq_q // block_size,
                                                     seq_k // block_size)

    for b in range(batch):
        for h in range(n_heads):
            for i in range(seq_q // block_size):
                for j in range(seq_k // block_size):
                    if not attn_layout_expanded[b, h, i, j]:
                        attn_scores[b, h,
                                    i * block_size:(i + 1) * block_size,
                                    j * block_size:(j + 1) * block_size] = float("-inf")

    # Apply attention mask if provided
    if attention_mask is not None:
        attention_mask_expanded = attention_mask.reshape(batch, n_heads, seq_q, seq_k)
        attn_scores = attn_scores.masked_fill(attention_mask_expanded, float("-inf"))

    # Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Handle NaN from all-masked rows
    attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

    # Output
    out = torch.matmul(attn_probs, v_t)

    return out.permute(0, 2, 1, 3)


@pytest.mark.parametrize("config", FLASH_ATTENTION_TEST_CONFIGS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_flash_attention_forward(config: tuple, use_amp: bool):
    """Test flash attention forward pass."""
    batch, seq_q, seq_k, n_heads, head_dim, block_size, attn_sparsity, use_causal_mask = config

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        # Create Q, K, V
        q = torch.randn(batch, seq_q, n_heads, head_dim, device=DEVICE)
        k = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)
        v = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)

        # Create attention layout
        n_batches = batch * n_heads
        n_seq_blocks_q = seq_q // block_size
        n_seq_blocks_k = seq_k // block_size
        attention_layout = _get_attention_layout(n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                                  attn_sparsity)

        # Create attention mask if needed
        attention_mask = None
        if use_causal_mask and seq_q == seq_k:  # Causal only makes sense for self-attention
            attention_mask = _build_causal_mask(n_batches, seq_q, seq_k)

        # Reference implementation
        ref_out = _reference_attention(q, k, v, attention_layout, block_size,
                                        attention_mask=attention_mask)
        ref_dtype = ref_out.dtype

        # Block-sparse implementation
        blksprs_out = bs.ops.flash_attention(
            q, k, v, attention_layout, block_size, attention_mask=attention_mask
        )

        # Compare
        assert torch.allclose(blksprs_out.to(ref_dtype), ref_out, atol=ATOL, rtol=RTOL), \
            f"Forward pass mismatch for config {config}, use_amp={use_amp}"


@pytest.mark.parametrize("config", FLASH_ATTENTION_TEST_CONFIGS)
@pytest.mark.parametrize("use_amp", [True, False])
def test_flash_attention_backward(config: tuple, use_amp: bool):
    """Test flash attention backward pass."""
    batch, seq_q, seq_k, n_heads, head_dim, block_size, attn_sparsity, use_causal_mask = config

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        # Create Q, K, V
        q = torch.randn(batch, seq_q, n_heads, head_dim, device=DEVICE)
        k = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)
        v = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)

        # Create attention layout
        n_batches = batch * n_heads
        n_seq_blocks_q = seq_q // block_size
        n_seq_blocks_k = seq_k // block_size
        attention_layout = _get_attention_layout(n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                                  attn_sparsity)

        # Create attention mask if needed
        attention_mask = None
        if use_causal_mask and seq_q == seq_k:
            attention_mask = _build_causal_mask(n_batches, seq_q, seq_k)

        # Reference implementation with gradients
        q_ref = q.clone().requires_grad_(True)
        k_ref = k.clone().requires_grad_(True)
        v_ref = v.clone().requires_grad_(True)
        ref_out = _reference_attention(q_ref, k_ref, v_ref, attention_layout, block_size,
                                        attention_mask=attention_mask)

        # Block-sparse implementation with gradients
        q_blksprs = q.clone().requires_grad_(True)
        k_blksprs = k.clone().requires_grad_(True)
        v_blksprs = v.clone().requires_grad_(True)
        blksprs_out = bs.ops.flash_attention(
            q_blksprs, k_blksprs, v_blksprs, attention_layout, block_size, attention_mask=attention_mask
        )

        # Forward check
        assert torch.allclose(blksprs_out.to(ref_out.dtype), ref_out, atol=ATOL, rtol=RTOL), \
            f"Forward pass mismatch before backward for config {config}"

        # Backward pass
        target = torch.randn_like(ref_out)
        ref_loss = F.l1_loss(ref_out, target)
        blksprs_loss = F.l1_loss(blksprs_out, target)

        ref_loss.backward()
        blksprs_loss.backward()

        # Compare gradients
        if q_ref.grad is not None and q_blksprs.grad is not None:
            assert torch.allclose(q_blksprs.grad.to(q_ref.grad.dtype), q_ref.grad,
                                  atol=ATOL, rtol=RTOL), \
                f"Q gradient mismatch for config {config}"

        if k_ref.grad is not None and k_blksprs.grad is not None:
            assert torch.allclose(k_blksprs.grad.to(k_ref.grad.dtype), k_ref.grad,
                                  atol=ATOL, rtol=RTOL), \
                f"K gradient mismatch for config {config}"

        if v_ref.grad is not None and v_blksprs.grad is not None:
            assert torch.allclose(v_blksprs.grad.to(v_ref.grad.dtype), v_ref.grad,
                                  atol=ATOL, rtol=RTOL), \
                f"V gradient mismatch for config {config}"


@pytest.mark.parametrize("use_mask", [True, False])
def test_flash_attention_mask_correctness(use_mask: bool):
    """Test that attention masking is applied correctly."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    # Dense attention layout
    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)

    # Create causal mask if needed
    attention_mask = _build_causal_mask(n_batches, seq, seq) if use_mask else None

    # Get output
    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size, attention_mask=attention_mask)

    # Check that output doesn't have NaN
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_flash_attention_sliding_window_mask():
    """Test sliding window attention mask."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16
    window_size = 48  # Attend only to positions within window_size

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)

    # Create sliding window mask: mask out positions more than window_size away
    # True = masked (blocked), False = allowed
    positions_q = torch.arange(seq, device=DEVICE).unsqueeze(1)  # [seq, 1]
    positions_k = torch.arange(seq, device=DEVICE).unsqueeze(0)  # [1, seq]
    distance = torch.abs(positions_q - positions_k)  # [seq, seq]
    sliding_mask = (distance > window_size)  # True where too far apart
    sliding_mask = sliding_mask.unsqueeze(0).expand(n_batches, -1, -1).contiguous()

    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size, attention_mask=sliding_mask)

    # Compare with reference
    ref_out = _reference_attention(q, k, v, attention_layout, block_size, attention_mask=sliding_mask)

    assert not torch.isnan(out).any(), "Output contains NaN"
    assert torch.allclose(out, ref_out, atol=ATOL, rtol=RTOL), "Sliding window mask mismatch"


def test_flash_attention_numerical_stability():
    """Test numerical stability with large values."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16

    # Large values that could cause overflow
    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE) * 10
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE) * 10
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    # Dense attention layout
    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)

    # Should not produce NaN or Inf
    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size)

    assert not torch.isnan(out).any(), "Output contains NaN with large inputs"
    assert not torch.isinf(out).any(), "Output contains Inf with large inputs"


def test_flash_attention_empty_attention():
    """Test handling of empty attention layout."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    # Fully sparse attention layout (no blocks to compute)
    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.zeros(n_batches, n_seq_blocks, n_seq_blocks,
                                   dtype=torch.bool, device=DEVICE)

    # Should return zeros
    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size)

    assert out.shape == (batch, seq, n_heads, head_dim)
    assert torch.allclose(out, torch.zeros_like(out)), "Empty attention should return zeros"


def test_flash_attention_input_validation():
    """Test input validation."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    # Valid attention layout
    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)

    # Wrong attention layout shape
    with pytest.raises(ValueError):
        wrong_layout = torch.ones(n_batches, n_seq_blocks + 1, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)
        bs.ops.flash_attention(q, k, v, wrong_layout, block_size)

    # Invalid block size (not power of 2)
    with pytest.raises(ValueError):
        bs.ops.flash_attention(q, k, v, attention_layout, 15)

    # Block size too small
    with pytest.raises(ValueError):
        bs.ops.flash_attention(q, k, v, attention_layout, 8)

    # Mismatched K/V sequence lengths
    with pytest.raises(ValueError):
        k_wrong = torch.randn(batch, seq + block_size, n_heads, head_dim, device=DEVICE)
        bs.ops.flash_attention(q, k_wrong, v, attention_layout, block_size)


@pytest.mark.parametrize("head_dim,block_size", [(64, 16), (64, 32), (128, 32), (128, 64)])
def test_flash_attention_head_dim_divisibility(head_dim: int, block_size: int):
    """Test different head_dim / block_size ratios."""
    batch, seq, n_heads = 2, 64, 4

    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks, n_seq_blocks,
                                  dtype=torch.bool, device=DEVICE)

    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size)

    # Compare with reference
    ref_out = _reference_attention(q, k, v, attention_layout, block_size)

    assert torch.allclose(out, ref_out, atol=ATOL, rtol=RTOL), \
        f"Mismatch for head_dim={head_dim}, block_size={block_size}"


def test_flash_attention_cross_attention():
    """Test cross-attention with different Q and K/V lengths."""
    batch, seq_q, seq_k, n_heads, head_dim = 2, 128, 64, 4, 64
    block_size = 16

    q = torch.randn(batch, seq_q, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq_k, n_heads, head_dim, device=DEVICE)

    n_batches = batch * n_heads
    n_seq_blocks_q = seq_q // block_size
    n_seq_blocks_k = seq_k // block_size
    attention_layout = torch.ones(n_batches, n_seq_blocks_q, n_seq_blocks_k,
                                  dtype=torch.bool, device=DEVICE)

    out = bs.ops.flash_attention(q, k, v, attention_layout, block_size)

    # Compare with reference
    ref_out = _reference_attention(q, k, v, attention_layout, block_size)

    assert out.shape == (batch, seq_q, n_heads, head_dim)
    assert torch.allclose(out, ref_out, atol=ATOL, rtol=RTOL)


def test_flash_attention_deterministic():
    """Test that output is deterministic."""
    batch, seq, n_heads, head_dim = 2, 64, 4, 64
    block_size = 16

    torch.manual_seed(42)
    q = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    k = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)
    v = torch.randn(batch, seq, n_heads, head_dim, device=DEVICE)

    n_batches = batch * n_heads
    n_seq_blocks = seq // block_size
    attention_layout = _get_attention_layout(n_batches, n_seq_blocks, n_seq_blocks, 0.5)

    # Run twice
    out1 = bs.ops.flash_attention(q, k, v, attention_layout, block_size)
    out2 = bs.ops.flash_attention(q, k, v, attention_layout, block_size)

    assert torch.allclose(out1, out2), "Output should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
