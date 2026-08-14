import pytest
import torch

from blksprs.layouting import (
    build_causal_self_attention_layout,
    build_causal_self_attention_mask,
    build_causal_window_self_attention_layout,
    build_causal_window_self_attention_mask,
)


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
