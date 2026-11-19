import pytest
import torch

from blksprs.layouting import (
    build_causal_self_attention_layout,
    build_causal_self_attention_mask,
    build_causal_window_self_attention_layout,
    build_causal_window_self_attention_mask,
)

LARGE_WINDOW_SIZES = (100, 1000, 1024, 4096)


def _effective_permissions(
        layout: torch.Tensor,
        mask: torch.Tensor,
        mask_layout: torch.Tensor,
        sparsity_block_size: int,
) -> torch.Tensor:
    """Expand a block layout and its compact mask into dense attention permissions."""
    n_batches, n_blocks, _ = layout.shape
    blocks = layout[..., None, None].expand(
        -1, -1, -1, sparsity_block_size, sparsity_block_size).clone()
    batch, query_block, key_block = mask_layout.nonzero(as_tuple=True)
    blocks[batch, query_block, key_block] &= ~mask.as_subclass(torch.Tensor)
    n_tokens = n_blocks * sparsity_block_size
    return blocks.transpose(2, 3).reshape(n_batches, n_tokens, n_tokens)


def _reference_permissions(
        lengths: torch.Tensor,
        n_tokens: int,
        window_size: int | None,
) -> torch.Tensor:
    """Dense causal, window, and padding permissions of each sequence."""
    query = torch.arange(n_tokens)[None, :, None]
    key = torch.arange(n_tokens)[None, None, :]
    lengths = lengths[:, None, None]
    allowed = (key <= query) & (query < lengths) & (key < lengths)
    if window_size is not None:
        allowed &= query - key < window_size
    return allowed


@pytest.mark.parametrize("sparsity_block_size", [16, 32, 64])
def test_causal_window_mask_matches_dense_reference(sparsity_block_size: int) -> None:
    block_size = sparsity_block_size
    window_sizes = [None, *range(1, 3 * block_size + 3), *LARGE_WINDOW_SIZES]
    failures = []
    for window_size in window_sizes:
        length_sets = [
            [4 * block_size],
            [4 * block_size - 3, 2 * block_size + 1, 1, 0],
            [3 * block_size + 5, block_size - 1],
        ]
        if window_size is not None:
            length_sets.append([window_size + 2 * block_size + 3, (window_size + 1) // 2])
        for length_set in length_sets:
            lengths = torch.tensor(length_set)
            if window_size is None:
                layout = build_causal_self_attention_layout(lengths, block_size)
                mask, mask_layout = build_causal_self_attention_mask(
                    lengths, layout, block_size)
            else:
                layout = build_causal_window_self_attention_layout(
                    lengths, block_size, window_size)
                mask, mask_layout = build_causal_window_self_attention_mask(
                    lengths, layout, block_size, window_size)

            n_blocks = layout.size(1)
            effective = _effective_permissions(layout, mask, mask_layout, block_size)
            reference = _reference_permissions(lengths, n_blocks * block_size, window_size)
            reference_blocks = reference.view(
                -1, n_blocks, block_size, n_blocks, block_size).any(dim=(2, 4))
            exact = torch.equal(effective, reference)
            # Every active block holds an attended pair, and every mask block masks a pair.
            tight_layout = torch.equal(layout, reference_blocks)
            compact_mask = bool(mask.flatten(1).any(dim=1).all())
            if not (exact and tight_layout and compact_mask):
                failures.append((
                    window_size,
                    length_set,
                    int((effective & ~reference).sum()),
                    int((reference & ~effective).sum()),
                    tight_layout,
                    compact_mask,
                ))

    assert not failures, (
        "(window_size, lengths, extra_pairs, missing_pairs, tight_layout, compact_mask): "
        f"{failures[:10]}"
    )


@pytest.mark.parametrize(
    ("sparsity_block_size", "layout_window_size", "window_size"),
    [(16, None, 17), (16, None, 1), (16, 33, 17), (16, 100, 64), (32, 4096, 1024)],
)
def test_causal_window_mask_rejects_layouts_beyond_the_window(
        sparsity_block_size: int,
        layout_window_size: int | None,
        window_size: int,
) -> None:
    lengths = torch.tensor([2 * window_size + 4 * sparsity_block_size + 5, window_size])
    if layout_window_size is None:
        layout = build_causal_self_attention_layout(lengths, sparsity_block_size)
    else:
        layout = build_causal_window_self_attention_layout(
            lengths, sparsity_block_size, layout_window_size)

    with pytest.raises(ValueError, match="outside the causal window band"):
        build_causal_window_self_attention_mask(
            lengths, layout, sparsity_block_size, window_size)


def test_causal_window_mask_accepts_sparser_layouts() -> None:
    lengths = torch.tensor([200, 77])
    layout = build_causal_window_self_attention_layout(lengths, 16, 17)
    mask, mask_layout = build_causal_window_self_attention_mask(lengths, layout, 16, 100)

    effective = _effective_permissions(layout, mask, mask_layout, 16)
    reference = _reference_permissions(lengths, layout.size(1) * 16, 100)
    dense_layout = layout.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
    assert torch.equal(effective, reference & dense_layout)


def test_causal_layout_tracks_each_valid_prefix() -> None:
    layout = build_causal_self_attention_layout(torch.tensor([1, 17, 32]), 16)

    assert layout.dtype is torch.bool
    assert layout.shape == (3, 2, 2)
    assert torch.equal(layout[0], torch.tensor([[True, False], [False, False]]))
    assert torch.equal(layout[1], torch.tensor([[True, False], [True, True]]))
    assert torch.equal(layout[2], torch.tensor([[True, False], [True, True]]))


def test_causal_mask_is_compact_and_masks_future_and_padding() -> None:
    layout = build_causal_self_attention_layout(torch.tensor([17]), 16)
    mask, mask_layout = build_causal_self_attention_mask(torch.tensor([17]), layout, 16)

    assert torch.equal(mask_layout[0], torch.tensor([[True, False], [True, True]]))
    assert mask.shape == (3, 16, 16)
    # First diagonal block masks only its strictly upper triangle.
    assert not mask[0, 15, 0]
    assert mask[0, 0, 1]
    # The off-diagonal block exists only because its query block is partial.
    assert not mask[1, 0, 0]
    assert mask[1, 1, 0]
    # The partial diagonal block masks future positions and all padding rows.
    assert not mask[2, 0, 0]
    assert mask[2, 0, 1]
    assert mask[2, 1, 0]


def test_causal_window_mask_refines_the_boundary_block() -> None:
    layout = build_causal_window_self_attention_layout(torch.tensor([32]), 16, 17)
    mask, mask_layout = build_causal_window_self_attention_mask(
        torch.tensor([32]), layout, 16, 17)

    assert torch.equal(layout[0], torch.tensor([[True, False], [True, True]]))
    assert torch.equal(mask_layout[0], torch.tensor([[True, False], [True, True]]))
    # Query token 16 may attend back to token 0 (distance 16), but not 17 tokens back.
    assert not mask[1, 0, 0]
    assert mask[1, 15, 0]


@pytest.mark.parametrize("lengths", [torch.tensor([-1]), torch.ones((1, 1), dtype=torch.long)])
def test_causal_layout_rejects_invalid_lengths(lengths: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_causal_self_attention_layout(lengths, 16)
