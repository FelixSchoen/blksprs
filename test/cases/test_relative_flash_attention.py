import importlib
import math

import pytest
import torch

import blksprs as bs


DEVICE = torch.device("cuda")
BLOCK_SIZE = 16
ATOL = 2e-2
RTOL = 1.5e-2
FLASH_ATTENTION_MODULE = importlib.import_module(
    "blksprs.ops.flash_attention")
RELATIVE_FLASH_ATTENTION_MODULE = importlib.import_module(
    "blksprs.ops.relative_flash_attention")


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
        key_relations_are_unique=True,
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
    torch.testing.assert_close(
        fused_output, reference_output, rtol=RTOL, atol=ATOL)

    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(fused_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            fused.grad, reference.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        fused_embedding.grad, reference_embedding.grad, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_matches_dense_across_attention_blocks() -> None:
    torch.manual_seed(12)
    sequence_length = 2 * BLOCK_SIZE
    fused_inputs = [
        torch.randn(
            (1, sequence_length, BLOCK_SIZE),
            device=DEVICE,
            dtype=torch.float32,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in fused_inputs
    ]
    relations = torch.arange(sequence_length, device=DEVICE).unsqueeze(0)
    fused_embedding = torch.randn(
        (1, 9, BLOCK_SIZE),
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_embedding = fused_embedding.detach().clone().requires_grad_(True)

    fused_output = _run_fused(*fused_inputs, relations, fused_embedding)
    reference_output = _run_reference(
        *reference_inputs, relations, reference_embedding)
    torch.testing.assert_close(
        fused_output, reference_output, rtol=RTOL, atol=ATOL)

    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(
            fused_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            fused.grad, reference.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        fused_embedding.grad,
        reference_embedding.grad,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head_blocks", [1, 2])
def test_relative_flash_attention_packs_sparse_query_rows(
        head_blocks: int) -> None:
    """Sparse query batches retain exact outputs and gradients when packed."""
    torch.manual_seed(121)
    batch_size = 2
    sequence_length = 4 * BLOCK_SIZE
    head_dimension = head_blocks * BLOCK_SIZE
    fused_inputs = [
        torch.randn(
            (batch_size, sequence_length, head_dimension),
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in fused_inputs
    ]
    query_layout = torch.zeros(
        (batch_size, 4, head_blocks), dtype=torch.bool, device=DEVICE
    )
    query_layout[0, -1] = True
    query_layout[1, -2:] = True
    if head_blocks > 1:
        query_layout[0, -1, 0] = False
    output_layout = torch.zeros_like(query_layout)
    output_layout[0, -1] = True
    output_layout[1, -2:] = True
    key_value_layout = torch.ones_like(query_layout)
    attention_layout = torch.zeros(
        (batch_size, 4, 4), dtype=torch.bool, device=DEVICE
    )
    attention_layout[0, -1] = True
    attention_layout[1, -2:] = True
    relations = torch.arange(
        sequence_length, device=DEVICE
    ).expand(batch_size, -1).contiguous()
    fused_embedding = torch.randn(
        (batch_size, 9, head_dimension),
        device=DEVICE,
        requires_grad=True,
    )
    reference_embedding = (
        fused_embedding.detach().clone().requires_grad_(True)
    )

    fused_sparse = [
        bs.ops.to_sparse(fused_inputs[0], query_layout, BLOCK_SIZE),
        bs.ops.to_sparse(fused_inputs[1], key_value_layout, BLOCK_SIZE),
        bs.ops.to_sparse(fused_inputs[2], key_value_layout, BLOCK_SIZE),
    ]
    fused_output = bs.ops.to_dense(
        bs.ops.flash_attention_relative_embedding(
            fused_sparse[0], query_layout,
            fused_sparse[1], key_value_layout,
            fused_sparse[2], key_value_layout,
            attention_layout,
            BLOCK_SIZE,
            relations,
            relations,
            fused_embedding,
                -4,
                4,
                key_relations_are_unique=True,
                sparsity_layout_o=output_layout,
            ),
            output_layout,
            BLOCK_SIZE,
        )

    query_feature_mask = query_layout.repeat_interleave(
        BLOCK_SIZE, dim=1
    ).repeat_interleave(BLOCK_SIZE, dim=2)
    sparse_reference_query = reference_inputs[0] * query_feature_mask
    relation_indices = (
        relations[:, :, None] - relations[:, None, :]
    ).clamp(-4, 4) + 4
    relative = reference_embedding[
        torch.arange(batch_size, device=DEVICE)[:, None, None],
        relation_indices,
    ]
    scores = (
        sparse_reference_query @ reference_inputs[1].transpose(-1, -2)
    ) / math.sqrt(head_dimension)
    scores = scores + torch.einsum(
        "bid,bijd->bij", sparse_reference_query, relative
    )
    allowed = attention_layout.repeat_interleave(
        BLOCK_SIZE, dim=-2
    ).repeat_interleave(BLOCK_SIZE, dim=-1)
    active_rows = allowed.any(dim=-1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.where(
        active_rows.unsqueeze(-1), scores, torch.zeros_like(scores)
    )
    reference_output = torch.softmax(scores, dim=-1) @ reference_inputs[2]
    reference_output = reference_output * active_rows.unsqueeze(-1)

    torch.testing.assert_close(
        fused_output, reference_output, rtol=RTOL, atol=ATOL
    )
    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(
            fused_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            fused.grad, reference.grad, rtol=RTOL, atol=ATOL
        )
    torch.testing.assert_close(
        fused_embedding.grad,
        reference_embedding.grad,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_reuses_packed_query_layout_cache() -> None:
    torch.manual_seed(122)
    sequence_length = 4 * BLOCK_SIZE
    query_layout = torch.zeros(
        (1, 4, 1), dtype=torch.bool, device=DEVICE
    )
    query_layout[:, -1] = True
    key_value_layout = torch.ones_like(query_layout)
    attention_layout = torch.zeros(
        (1, 4, 4), dtype=torch.bool, device=DEVICE
    )
    attention_layout[:, -1] = True
    dense_inputs = [
        torch.randn(
            (1, sequence_length, BLOCK_SIZE), device=DEVICE
        )
        for _ in range(3)
    ]
    sparse_inputs = [
        bs.ops.to_sparse(dense_inputs[0], query_layout, BLOCK_SIZE),
        bs.ops.to_sparse(dense_inputs[1], key_value_layout, BLOCK_SIZE),
        bs.ops.to_sparse(dense_inputs[2], key_value_layout, BLOCK_SIZE),
    ]
    relations = torch.arange(
        sequence_length, device=DEVICE
    ).unsqueeze(0)
    embedding = torch.randn((1, 9, BLOCK_SIZE), device=DEVICE)
    layout_cache = {}

    def run() -> torch.Tensor:
        return bs.ops.to_dense(
            bs.ops.flash_attention_relative_embedding(
                sparse_inputs[0], query_layout,
                sparse_inputs[1], key_value_layout,
                sparse_inputs[2], key_value_layout,
                attention_layout,
                BLOCK_SIZE,
                relations,
                relations,
                embedding,
                -4,
                4,
                key_relations_are_unique=True,
                layout_cache=layout_cache,
                sparsity_layout_o=query_layout,
            ),
            query_layout,
            BLOCK_SIZE,
        )

    first = run()
    first_projection = layout_cache["relative_query_projection"]
    first_row_indices = first_projection["row_indices"]
    first_gather_indices = first_projection["gather_indices"]
    first_source_block_indices = first_projection["source_block_indices"]
    first_source_row_indices = first_projection["source_row_indices"]
    second = run()

    torch.testing.assert_close(second, first, rtol=RTOL, atol=ATOL)
    assert layout_cache["relative_query_projection"] is first_projection
    assert first_projection["row_indices"] is first_row_indices
    assert first_projection["gather_indices"] is first_gather_indices
    assert first_projection["source_block_indices"] is first_source_block_indices
    assert first_projection["source_row_indices"] is first_source_row_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("packed_queries", [False, True])
def test_relative_flash_attention_index_widths_match(
        monkeypatch: pytest.MonkeyPatch,
        packed_queries: bool,
) -> None:
    """Projected-table addresses are exact with 32- or 64-bit arithmetic."""
    torch.manual_seed(124)
    batch_size = 2
    n_blocks = 4
    sequence_length = n_blocks * BLOCK_SIZE
    query_layout = torch.ones(
        (batch_size, n_blocks, 1), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.ones(
        (batch_size, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE)
    if packed_queries:
        query_layout[:, :-1] = False
        attention_layout[:, :-1] = False
    key_value_layout = torch.ones_like(query_layout)
    base_inputs = [
        torch.randn(
            (batch_size, sequence_length, BLOCK_SIZE), device=DEVICE)
        for _ in range(3)
    ]
    positions = torch.arange(sequence_length, device=DEVICE)
    relations = torch.stack((
        positions,
        positions // 3,
        positions % 5,
    )).unsqueeze(1).expand(-1, batch_size, -1).contiguous()
    relation_bounds = ((-4, 4), (-2, 2), (0, 0))
    base_embeddings = torch.randn(
        (batch_size, 15, BLOCK_SIZE), device=DEVICE)
    output_gradient = torch.randn_like(base_inputs[0])

    def run() -> tuple[torch.Tensor, tuple[torch.Tensor, ...], bool]:
        inputs = [
            tensor.detach().clone().requires_grad_(True)
            for tensor in base_inputs
        ]
        embeddings = base_embeddings.detach().clone().requires_grad_(True)
        sparse_inputs = [
            bs.ops.to_sparse(inputs[0], query_layout, BLOCK_SIZE),
            bs.ops.to_sparse(inputs[1], key_value_layout, BLOCK_SIZE),
            bs.ops.to_sparse(inputs[2], key_value_layout, BLOCK_SIZE),
        ]
        layout_cache = {}
        output = bs.ops.to_dense(
            bs.ops.flash_attention_relative_embeddings(
                sparse_inputs[0], query_layout,
                sparse_inputs[1], key_value_layout,
                sparse_inputs[2], key_value_layout,
                attention_layout,
                BLOCK_SIZE,
                relations,
                relations,
                embeddings,
                relation_bounds,
                key_relations_are_unique=(True, False, False),
                layout_cache=layout_cache,
                sparsity_layout_o=query_layout,
            ),
            query_layout,
            BLOCK_SIZE,
        )
        (output * output_gradient).sum().backward()
        gradients = tuple(
            tensor.grad.detach().clone()
            for tensor in (*inputs, embeddings)
        )
        projection_is_packed = layout_cache[
            "relative_query_projection"]["packed"]
        return output.detach(), gradients, projection_is_packed

    original_index_check = FLASH_ATTENTION_MODULE.can_use_int32_indexing
    decisions = []

    def record_index_check(*values) -> bool:
        decision = original_index_check(*values)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(
        FLASH_ATTENTION_MODULE,
        "can_use_int32_indexing",
        record_index_check,
    )
    output_32, gradients_32, projection_is_packed = run()
    assert decisions and all(decisions)
    assert projection_is_packed is packed_queries

    monkeypatch.setattr(
        FLASH_ATTENTION_MODULE,
        "can_use_int32_indexing",
        lambda *values: False,
    )
    output_64, gradients_64, _ = run()

    torch.testing.assert_close(
        output_32, output_64, rtol=RTOL, atol=ATOL)
    for gradient_32, gradient_64 in zip(
            gradients_32, gradients_64, strict=True):
        torch.testing.assert_close(
            gradient_32, gradient_64, rtol=RTOL, atol=ATOL)


def _run_attention_with_mask(
        attention_mask: torch.Tensor,
        *,
        use_relative_attention: bool,
) -> torch.Tensor:
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    inputs = [
        bs.BlksprsTensor.wrap(torch.randn(
            (1, BLOCK_SIZE, BLOCK_SIZE), device=DEVICE))
        for _ in range(3)
    ]
    arguments = (
        inputs[0], layout,
        inputs[1], layout,
        inputs[2], layout,
        layout,
        BLOCK_SIZE,
    )
    mask = bs.BlksprsTensor.wrap(attention_mask)
    if use_relative_attention:
        relations = torch.arange(
            BLOCK_SIZE, device=DEVICE).unsqueeze(0)
        return bs.ops.flash_attention_relative_embedding(
            *arguments,
            relations,
            relations,
            torch.randn((1, 3, BLOCK_SIZE), device=DEVICE),
            -1,
            1,
            attention_mask=mask,
            sparsity_layout_mask=layout,
            sparsity_layout_o=layout,
        )
    return bs.ops.flash_attention(
        *arguments,
        attention_mask=mask,
        sparsity_layout_mask=layout,
        sparsity_layout_o=layout,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_relative_attention", [False, True])
def test_boolean_attention_masks_skip_binary_value_scan(
        monkeypatch: pytest.MonkeyPatch,
        use_relative_attention: bool,
) -> None:
    module = (
        RELATIVE_FLASH_ATTENTION_MODULE
        if use_relative_attention
        else FLASH_ATTENTION_MODULE
    )

    def fail_binary_validation(*tensors) -> None:
        raise AssertionError("boolean masks must not be scanned")

    monkeypatch.setattr(module, "validate_binary", fail_binary_validation)
    output = _run_attention_with_mask(
        torch.zeros(
            (1, BLOCK_SIZE, BLOCK_SIZE), dtype=torch.bool, device=DEVICE),
        use_relative_attention=use_relative_attention,
    )
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_relative_attention", [False, True])
def test_floating_attention_masks_retain_binary_value_validation(
        monkeypatch: pytest.MonkeyPatch,
        use_relative_attention: bool,
) -> None:
    module = (
        RELATIVE_FLASH_ATTENTION_MODULE
        if use_relative_attention
        else FLASH_ATTENTION_MODULE
    )
    original_validate_binary = module.validate_binary
    validated_tensors = []

    def record_binary_validation(*tensors) -> None:
        validated_tensors.extend(tensors)
        original_validate_binary(*tensors)

    monkeypatch.setattr(
        module,
        "validate_binary",
        record_binary_validation,
    )
    valid_mask = torch.zeros(
        (1, BLOCK_SIZE, BLOCK_SIZE), device=DEVICE)
    output = _run_attention_with_mask(
        valid_mask,
        use_relative_attention=use_relative_attention,
    )
    assert torch.isfinite(output).all()
    assert len(validated_tensors) == 1
    assert validated_tensors[0].dtype == torch.float32

    invalid_mask = torch.full_like(valid_mask, 2.0)
    with pytest.raises(ValueError, match="0 or 1"):
        _run_attention_with_mask(
            invalid_mask,
            use_relative_attention=use_relative_attention,
        )
    assert len(validated_tensors) == 2
    assert all(tensor.dtype == torch.float32 for tensor in validated_tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_packs_active_rows_without_query_storage() -> None:
    torch.manual_seed(123)
    sequence_length = 4 * BLOCK_SIZE
    dense_inputs = [
        torch.randn(
            (1, sequence_length, BLOCK_SIZE),
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in dense_inputs
    ]
    query_layout = torch.zeros(
        (1, 4, 1), dtype=torch.bool, device=DEVICE
    )
    query_layout[:, -1] = True
    output_layout = torch.zeros_like(query_layout)
    output_layout[:, -2:] = True
    key_value_layout = torch.ones_like(query_layout)
    attention_layout = torch.zeros(
        (1, 4, 4), dtype=torch.bool, device=DEVICE
    )
    attention_layout[:, -2:] = True
    relations = torch.arange(
        sequence_length, device=DEVICE
    ).unsqueeze(0)
    fused_embedding = torch.randn(
        (1, 9, BLOCK_SIZE), device=DEVICE, requires_grad=True
    )
    reference_embedding = (
        fused_embedding.detach().clone().requires_grad_(True)
    )

    fused_output = bs.ops.to_dense(
        bs.ops.flash_attention_relative_embedding(
            bs.ops.to_sparse(dense_inputs[0], query_layout, BLOCK_SIZE),
            query_layout,
            bs.ops.to_sparse(
                dense_inputs[1], key_value_layout, BLOCK_SIZE
            ),
            key_value_layout,
            bs.ops.to_sparse(
                dense_inputs[2], key_value_layout, BLOCK_SIZE
            ),
            key_value_layout,
            attention_layout,
            BLOCK_SIZE,
            relations,
            relations,
            fused_embedding,
            -4,
            4,
            key_relations_are_unique=True,
            sparsity_layout_o=output_layout,
        ),
        output_layout,
        BLOCK_SIZE,
    )

    query_row_mask = query_layout.repeat_interleave(
        BLOCK_SIZE, dim=1
    )
    sparse_query = reference_inputs[0] * query_row_mask
    relation_indices = (
        relations[:, :, None] - relations[:, None, :]
    ).clamp(-4, 4) + 4
    relative = reference_embedding[0][relation_indices[0]].unsqueeze(0)
    scores = (
        sparse_query @ reference_inputs[1].transpose(-1, -2)
    ) / math.sqrt(BLOCK_SIZE)
    scores = scores + torch.einsum(
        "bid,bijd->bij", sparse_query, relative
    )
    allowed = attention_layout.repeat_interleave(
        BLOCK_SIZE, dim=-2
    ).repeat_interleave(BLOCK_SIZE, dim=-1)
    active_rows = allowed.any(dim=-1)
    scores = scores.masked_fill(~allowed, float("-inf"))
    scores = torch.where(
        active_rows.unsqueeze(-1), scores, torch.zeros_like(scores)
    )
    reference_output = torch.softmax(scores, dim=-1) @ reference_inputs[2]
    reference_output = reference_output * active_rows.unsqueeze(-1)

    torch.testing.assert_close(
        fused_output, reference_output, rtol=RTOL, atol=ATOL
    )
    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(
            dense_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            fused.grad, reference.grad, rtol=RTOL, atol=ATOL
        )
    torch.testing.assert_close(
        fused_embedding.grad,
        reference_embedding.grad,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_packs_empty_attention_layout() -> None:
    torch.manual_seed(124)
    layout = torch.ones(
        (1, 2, 1), dtype=torch.bool, device=DEVICE
    )
    attention_layout = torch.zeros(
        (1, 2, 2), dtype=torch.bool, device=DEVICE
    )
    dense_inputs = [
        torch.randn(
            (1, 2 * BLOCK_SIZE, BLOCK_SIZE),
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    sparse_inputs = [
        bs.ops.to_sparse(tensor, layout, BLOCK_SIZE)
        for tensor in dense_inputs
    ]
    relations = torch.arange(
        2 * BLOCK_SIZE, device=DEVICE
    ).unsqueeze(0)
    embedding = torch.randn(
        (1, 9, BLOCK_SIZE), device=DEVICE, requires_grad=True
    )

    output = bs.ops.flash_attention_relative_embedding(
        sparse_inputs[0], layout,
        sparse_inputs[1], layout,
        sparse_inputs[2], layout,
        attention_layout,
        BLOCK_SIZE,
        relations,
        relations,
        embedding,
        -4,
        4,
        key_relations_are_unique=True,
        sparsity_layout_o=layout,
    )
    output.sum().backward()

    assert torch.count_nonzero(output) == 0
    for tensor in (*dense_inputs, embedding):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_relative_flash_attention_supports_empty_batch() -> None:
    layout = torch.zeros(
        (0, 1, 1), dtype=torch.bool, device=DEVICE
    )
    inputs = [
        torch.empty(
            (0, BLOCK_SIZE, BLOCK_SIZE),
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    sparse_inputs = [
        bs.BlksprsTensor.wrap(tensor) for tensor in inputs
    ]
    relations = torch.empty(
        (1, 0, BLOCK_SIZE), dtype=torch.int64, device=DEVICE
    )
    embedding = torch.empty(
        (0, 3, BLOCK_SIZE), device=DEVICE, requires_grad=True
    )

    output = bs.ops.flash_attention_relative_embeddings(
        sparse_inputs[0], layout,
        sparse_inputs[1], layout,
        sparse_inputs[2], layout,
        layout,
        BLOCK_SIZE,
        relations,
        relations,
        embedding,
        ((-1, 1),),
        sparsity_layout_o=layout,
    )
    output.sum().backward()

    assert output.shape == (0, BLOCK_SIZE, BLOCK_SIZE)
    for tensor in (*inputs, embedding):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("block_size", "n_blocks"),
    [(16, 1), (16, 2), (32, 1), (64, 1)],
)
def test_multi_relative_flash_attention_matches_dense_forward_and_gradients(
        block_size: int,
        n_blocks: int,
) -> None:
    torch.manual_seed(14)
    sequence_length = n_blocks * block_size
    bounds = ((-2, 2), (-3, 3), (0, 0))
    query_relations = torch.stack((
        torch.arange(sequence_length, device=DEVICE),
        torch.arange(sequence_length, device=DEVICE) // 2,
        torch.arange(sequence_length, device=DEVICE),
    )).unsqueeze(1)
    key_relations = torch.stack((
        torch.arange(sequence_length, device=DEVICE) // 2,
        torch.arange(sequence_length, device=DEVICE) // 3,
        torch.arange(sequence_length - 1, -1, -1, device=DEVICE),
    )).unsqueeze(1)
    query_relations[:, :, 0] = -1
    key_relations[:, :, 0] = -1
    query_validity = query_relations.ge(0)
    key_validity = key_relations.ge(0)
    fused_inputs = [
        torch.randn(
            (1, sequence_length, block_size),
            device=DEVICE,
            dtype=torch.float32,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    fused_embedding = torch.randn(
        (1, 13, block_size),
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_inputs = [
        tensor.detach().clone().requires_grad_(True)
        for tensor in fused_inputs
    ]
    reference_embedding = fused_embedding.detach().clone().requires_grad_(True)
    layout = torch.ones(
        (1, n_blocks, 1), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.tril(torch.ones(
        (1, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    mask, mask_layout = bs.layouting.build_causal_self_attention_mask(
        torch.tensor([sequence_length], device=DEVICE),
        attention_layout,
        block_size,
    )

    fused_sparse = [
        bs.ops.to_sparse(tensor, layout, block_size)
        for tensor in fused_inputs
    ]
    fused_output = bs.ops.to_dense(
        bs.ops.flash_attention_relative_embeddings(
            fused_sparse[0], layout,
            fused_sparse[1], layout,
            fused_sparse[2], layout,
            attention_layout,
            block_size,
            query_relations,
            key_relations,
            fused_embedding,
            bounds,
            query_relation_validity=query_validity,
            key_relation_validity=key_validity,
            attention_mask=mask,
            sparsity_layout_mask=mask_layout,
            sparsity_layout_o=layout,
        ),
        layout,
        block_size,
    )

    scores = (
        reference_inputs[0] @ reference_inputs[1].transpose(-1, -2)
    ) / math.sqrt(block_size)
    relation_offset = 0
    for relation, (relation_min, relation_max) in enumerate(bounds):
        indices = (
            query_relations[relation, :, :, None]
            - key_relations[relation, :, None, :]
        ).clamp(relation_min, relation_max) - relation_min + relation_offset
        gathered = reference_embedding[
            torch.arange(1, device=DEVICE)[:, None, None],
            indices,
        ]
        pair_validity = (
            query_validity[relation, :, :, None]
            & key_validity[relation, :, None, :]
        )
        scores = scores + torch.einsum(
            "bid,bijd->bij", reference_inputs[0], gathered
        ) * pair_validity
        relation_offset += relation_max - relation_min + 1
    causal_mask = torch.triu(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=DEVICE,
        ),
        diagonal=1,
    )
    reference_output = torch.softmax(
        scores.masked_fill(causal_mask, float("-inf")), dim=-1
    ) @ reference_inputs[2]

    torch.testing.assert_close(
        fused_output, reference_output, rtol=RTOL, atol=ATOL)
    gradient = torch.randn_like(fused_output)
    (fused_output * gradient).sum().backward()
    (reference_output * gradient).sum().backward()
    for fused, reference in zip(
            fused_inputs, reference_inputs, strict=True):
        torch.testing.assert_close(
            fused.grad, reference.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        fused_embedding.grad,
        reference_embedding.grad,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_relative_attention", [False, True])
def test_native_causal_lengths_match_compact_mask_with_padding(
        use_relative_attention: bool,
) -> None:
    torch.manual_seed(15)
    valid_length = BLOCK_SIZE + 5
    padded_length = 2 * BLOCK_SIZE
    feature_layout = torch.ones(
        (1, 2, 1), dtype=torch.bool, device=DEVICE)
    attention_layout = bs.layouting.build_causal_self_attention_layout(
        torch.tensor([valid_length], device=DEVICE), BLOCK_SIZE)
    attention_mask, mask_layout = bs.layouting.build_causal_self_attention_mask(
        torch.tensor([valid_length], device=DEVICE),
        attention_layout,
        BLOCK_SIZE,
    )
    dense_inputs = [
        torch.randn(
            (1, padded_length, BLOCK_SIZE),
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(3)
    ]
    native_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in dense_inputs
    ]
    explicit_sparse = [
        bs.ops.to_sparse(value, feature_layout, BLOCK_SIZE)
        for value in dense_inputs
    ]
    native_sparse = [
        bs.ops.to_sparse(value, feature_layout, BLOCK_SIZE)
        for value in native_inputs
    ]
    relations = torch.arange(
        padded_length, device=DEVICE).unsqueeze(0)
    explicit_embedding = torch.randn(
        (1, 9, BLOCK_SIZE), device=DEVICE, requires_grad=True)
    native_embedding = (
        explicit_embedding.detach().clone().requires_grad_(True))

    def run(
            sparse_inputs: list[torch.Tensor],
            embedding: torch.Tensor,
            *,
            native: bool,
    ) -> torch.Tensor:
        mask_arguments = (
            {"causal_lengths": torch.tensor([valid_length], device=DEVICE)}
            if native
            else {
                "attention_mask": attention_mask,
                "sparsity_layout_mask": mask_layout,
            }
        )
        if use_relative_attention:
            sparse_output = bs.ops.flash_attention_relative_embedding(
                sparse_inputs[0], feature_layout,
                sparse_inputs[1], feature_layout,
                sparse_inputs[2], feature_layout,
                attention_layout,
                BLOCK_SIZE,
                relations,
                relations,
                embedding,
                -4,
                4,
                key_relations_are_unique=True,
                sparsity_layout_o=feature_layout,
                **mask_arguments,
            )
        else:
            sparse_output = bs.ops.flash_attention(
                sparse_inputs[0], feature_layout,
                sparse_inputs[1], feature_layout,
                sparse_inputs[2], feature_layout,
                attention_layout,
                BLOCK_SIZE,
                sparsity_layout_o=feature_layout,
                **mask_arguments,
            )
        return bs.ops.to_dense(
            sparse_output, feature_layout, BLOCK_SIZE)

    explicit_output = run(
        explicit_sparse, explicit_embedding, native=False)
    native_output = run(native_sparse, native_embedding, native=True)
    torch.testing.assert_close(
        native_output, explicit_output, rtol=RTOL, atol=ATOL)

    gradient = torch.randn_like(explicit_output)
    (explicit_output * gradient).sum().backward()
    (native_output * gradient).sum().backward()
    for native, explicit in zip(
            native_inputs, dense_inputs, strict=True):
        torch.testing.assert_close(
            native.grad, explicit.grad, rtol=RTOL, atol=ATOL)
    if use_relative_attention:
        torch.testing.assert_close(
            native_embedding.grad,
            explicit_embedding.grad,
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("causal_lengths", "exception", "message"),
    [
        (torch.tensor([16.0]), TypeError, "integral"),
        (torch.tensor([[16]]), ValueError, "one-dimensional"),
        (torch.tensor([-1]), ValueError, "between zero and 16"),
        (torch.tensor([17]), ValueError, "between zero and 16"),
    ],
)
def test_native_causal_lengths_are_validated(
        causal_lengths: torch.Tensor,
        exception: type[Exception],
        message: str,
) -> None:
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    inputs = [
        bs.ops.to_sparse(
            torch.randn((1, BLOCK_SIZE, BLOCK_SIZE), device=DEVICE),
            layout,
            BLOCK_SIZE,
        )
        for _ in range(3)
    ]
    with pytest.raises(exception, match=message):
        bs.ops.flash_attention(
            inputs[0], layout,
            inputs[1], layout,
            inputs[2], layout,
            layout,
            BLOCK_SIZE,
            causal_lengths=causal_lengths.to(DEVICE),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 64])
def test_relative_flash_attention_supports_maximum_relation_count(
        block_size: int,
) -> None:
    torch.manual_seed(15)
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    attention_mask, mask_layout = (
        bs.layouting.build_causal_self_attention_mask(
            torch.tensor([block_size], device=DEVICE),
            layout,
            block_size,
        )
    )
    inputs = [
        bs.ops.to_sparse(
            torch.randn(
                (1, block_size, block_size),
                device=DEVICE,
                requires_grad=True,
            ),
            layout,
            block_size,
        )
        for _ in range(3)
    ]
    position = torch.arange(block_size, device=DEVICE)
    relations = torch.stack(
        tuple(position // divisor for divisor in range(1, 9)),
    ).unsqueeze(1)
    embeddings = torch.randn(
        (1, 24, block_size),
        device=DEVICE,
        requires_grad=True,
    )

    output = bs.ops.flash_attention_relative_embeddings(
        inputs[0], layout,
        inputs[1], layout,
        inputs[2], layout,
        layout,
        block_size,
        relations,
        relations,
        embeddings,
        ((-1, 1),) * 8,
        attention_mask=attention_mask,
        sparsity_layout_mask=mask_layout,
        sparsity_layout_o=layout,
    )
    output.float().sum().backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(embeddings.grad).all()


@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("packed_queries", [False, True])
def test_relative_flash_attention_training_performance(
        packed_queries: bool,
) -> None:
    batch_size = 8
    sequence_length = 1024
    head_dimension = 64
    block_size = 32
    n_sequence_blocks = sequence_length // block_size
    n_head_blocks = head_dimension // block_size
    query_layout = torch.ones(
        (batch_size, n_sequence_blocks, n_head_blocks),
        dtype=torch.bool,
        device=DEVICE,
    )
    if packed_queries:
        query_layout[:, :-2] = False
    key_value_layout = torch.ones_like(query_layout)
    attention_layout = torch.tril(torch.ones(
        (batch_size, n_sequence_blocks, n_sequence_blocks),
        dtype=torch.bool,
        device=DEVICE,
    ))
    if packed_queries:
        attention_layout[:, :-2] = False
    mask, mask_layout = bs.layouting.build_causal_self_attention_mask(
        torch.full((batch_size,), sequence_length, device=DEVICE),
        attention_layout,
        block_size,
    )
    query = torch.randn(
        (int(query_layout.sum().item()), block_size, block_size),
        device=DEVICE,
        dtype=torch.float16,
        requires_grad=True,
    )
    key_value_inputs = [
        torch.randn(
            (int(key_value_layout.sum().item()),
             block_size, block_size),
            device=DEVICE,
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    inputs = [query, *key_value_inputs]
    positions = torch.arange(
        sequence_length, device=DEVICE, dtype=torch.int32
    ).expand(batch_size, -1).contiguous()
    relations = torch.stack((
        positions,
        positions // 16,
        positions % 16,
    ))
    relation_bounds = ((-512, 512), (-32, 32), (-8, 8))
    n_relation_embeddings = sum(
        maximum - minimum + 1
        for minimum, maximum in relation_bounds
    )
    embeddings = torch.randn(
        (batch_size, n_relation_embeddings, head_dimension),
        device=DEVICE,
        dtype=torch.float16,
        requires_grad=True,
    )
    flash_cache = {}
    relative_cache = {}

    def run_flash():
        return bs.ops.flash_attention(
            inputs[0], query_layout,
            inputs[1], key_value_layout,
            inputs[2], key_value_layout,
            attention_layout,
            block_size,
            attention_mask=mask,
            sparsity_layout_mask=mask_layout,
            sparsity_layout_o=query_layout,
            layout_cache=flash_cache,
        )

    def run_relative():
        return bs.ops.flash_attention_relative_embeddings(
            inputs[0], query_layout,
            inputs[1], key_value_layout,
            inputs[2], key_value_layout,
            attention_layout,
            block_size,
            relations,
            relations,
            embeddings,
            relation_bounds,
            key_relations_are_unique=(True, False, False),
            attention_mask=mask,
            sparsity_layout_mask=mask_layout,
            sparsity_layout_o=query_layout,
            layout_cache=relative_cache,
        )

    def measure(function, gradient_inputs: list[torch.Tensor]) -> float:
        for _ in range(2):
            function().float().sum().backward()
            for tensor in gradient_inputs:
                tensor.grad = None
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function().float().sum().backward()
        end.record()
        end.synchronize()
        for tensor in gradient_inputs:
            tensor.grad = None
        return start.elapsed_time(end)

    flash_time = measure(run_flash, inputs)
    relative_time = measure(run_relative, [*inputs, embeddings])

    assert relative_cache["relative_query_projection"]["packed"] is packed_queries

    assert relative_time <= flash_time * 8, (
        "Relative Flash attention regressed beyond the guarded training-cost "
        f"ratio: relative={relative_time:.3f} ms, flash={flash_time:.3f} ms."
    )
