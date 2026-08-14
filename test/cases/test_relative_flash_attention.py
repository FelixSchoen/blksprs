import math

import pytest
import torch

import blksprs as bs


DEVICE = torch.device("cuda")
BLOCK_SIZE = 16


def _run_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations: torch.Tensor,
        embedding: torch.Tensor,
) -> torch.Tensor:
    n_blocks = query.size(1) // BLOCK_SIZE
    n_head_blocks = query.size(2) // BLOCK_SIZE
    layout = torch.ones((1, n_blocks, n_head_blocks), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.tril(
        torch.ones((1, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    mask, mask_layout = bs.layouting.build_causal_self_attention_mask(
        torch.tensor([query.size(1)], device=DEVICE), attention_layout, BLOCK_SIZE)
    output = bs.ops.flash_attention_relative_embedding(
        bs.ops.to_sparse(query, layout, BLOCK_SIZE), layout,
        bs.ops.to_sparse(key, layout, BLOCK_SIZE), layout,
        bs.ops.to_sparse(value, layout, BLOCK_SIZE), layout,
        attention_layout,
        BLOCK_SIZE,
        relations,
        relations,
        embedding,
        -4,
        4,
        attention_mask=mask,
        sparsity_layout_mask=mask_layout,
        sparsity_layout_o=layout,
    )
    return bs.ops.to_dense(output, layout, BLOCK_SIZE)


def _run_reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        relations: torch.Tensor,
        embedding: torch.Tensor,
) -> torch.Tensor:
    relation_indices = (relations[:, :, None] - relations[:, None, :]).clamp(-4, 4) + 4
    relative = embedding[0][relation_indices[0]].unsqueeze(0)
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.size(-1))
    scores = scores + torch.einsum("bid,bijd->bij", query, relative)
    causal_mask = torch.triu(
        torch.ones((query.size(1), query.size(1)), dtype=torch.bool, device=DEVICE), diagonal=1)
    return torch.softmax(scores.masked_fill(causal_mask, float("-inf")), dim=-1) @ value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_blocks", [1, 2])
def test_relative_flash_attention_matches_dense_forward_and_gradients(
        head_blocks: int,
) -> None:
    torch.manual_seed(11)
    head_dimension = head_blocks * BLOCK_SIZE
    fused_inputs = [
        torch.randn((1, BLOCK_SIZE, head_dimension), device=DEVICE, dtype=torch.float32, requires_grad=True)
        for _ in range(3)
    ]
    fused_relations = torch.arange(BLOCK_SIZE, device=DEVICE).unsqueeze(0)
    fused_embedding = torch.randn((1, 9, head_dimension), device=DEVICE, dtype=torch.float32, requires_grad=True)
    reference_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in fused_inputs]
    reference_embedding = fused_embedding.detach().clone().requires_grad_(True)

    fused_output = _run_fused(*fused_inputs, fused_relations, fused_embedding)
    reference_output = _run_reference(*reference_inputs, fused_relations, reference_embedding)
    torch.testing.assert_close(fused_output, reference_output, rtol=2e-4, atol=2e-4)

    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(fused_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(fused.grad, reference.grad, rtol=4e-4, atol=4e-4)
    torch.testing.assert_close(fused_embedding.grad, reference_embedding.grad, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_matches_dense_across_attention_blocks() -> None:
    torch.manual_seed(12)
    sequence_length = 2 * BLOCK_SIZE
    query, key, value = [
        torch.randn((1, sequence_length, BLOCK_SIZE), device=DEVICE, dtype=torch.float32)
        for _ in range(3)
    ]
    relations = torch.arange(sequence_length, device=DEVICE).unsqueeze(0)
    embedding = torch.randn((1, 9, BLOCK_SIZE), device=DEVICE, dtype=torch.float32)

    torch.testing.assert_close(
        _run_fused(query, key, value, relations, embedding),
        _run_reference(query, key, value, relations, embedding),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_supports_cuda_autocast() -> None:
    torch.manual_seed(13)
    query, key, value = [
        torch.randn((1, BLOCK_SIZE, BLOCK_SIZE), device=DEVICE, dtype=torch.float32)
        for _ in range(3)
    ]
    relations = torch.arange(BLOCK_SIZE, device=DEVICE).unsqueeze(0)
    embedding = torch.randn((1, 9, BLOCK_SIZE), device=DEVICE, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=True):
        output = _run_fused(query, key, value, relations, embedding)

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
