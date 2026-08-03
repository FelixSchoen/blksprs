import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")

import pytest
import numpy as np
import torch

import blksprs as bs
from blksprs.ops.conversion import to_sparse_build_layout_cache
from blksprs.ops.flash_attention import _select_segmented_index_dtype
from blksprs.utils.benchmarking import benchmark


DEVICE = torch.device("cuda:0")
SPARSITY_BLOCK_SIZE = 16


def _full_layout(rows: int = 1, columns: int = 1) -> torch.Tensor:
    return torch.ones((1, rows, columns), dtype=torch.bool, device=DEVICE)


def _non_contiguous_gradient(blocks: int = 1) -> torch.Tensor:
    base = torch.randn((blocks, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
    gradient = base[:, :, ::2]
    assert not gradient.is_contiguous()
    return gradient


def _simple_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale=None,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    layout = _full_layout()
    bias_sparse = None
    bias_layout = None
    if attention_bias is not None:
        bias_sparse = bs.ops.to_sparse(
            attention_bias, layout, SPARSITY_BLOCK_SIZE)
        bias_layout = layout

    output = bs.ops.flash_attention(
        bs.ops.to_sparse(q, layout, SPARSITY_BLOCK_SIZE), layout,
        bs.ops.to_sparse(k, layout, SPARSITY_BLOCK_SIZE), layout,
        bs.ops.to_sparse(v, layout, SPARSITY_BLOCK_SIZE), layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        scale=scale,
        attention_bias=bias_sparse,
        sparsity_layout_bias=bias_layout,
        sparsity_layout_o=layout,
    )
    return bs.ops.to_dense(output, layout, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize(
    ("dtype", "key_length"),
    [
        (torch.bfloat16, 16_384),
        (torch.float16, 65_536),
    ],
)
def test_flash_attention_long_sequence_forward_preserves_uniform_values(
        dtype: torch.dtype, key_length: int):
    block_size = 32
    head_dimension = 32
    query_length = block_size
    query_layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    key_layout = torch.ones(
        (1, key_length // block_size, 1), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.ones(
        (1, 1, key_length // block_size), dtype=torch.bool, device=DEVICE)
    q = torch.zeros((1, query_length, head_dimension), dtype=dtype, device=DEVICE)
    k = torch.zeros((1, key_length, head_dimension), dtype=dtype, device=DEVICE)
    v = torch.ones((1, key_length, head_dimension), dtype=dtype, device=DEVICE)

    output_sparse = bs.ops.flash_attention(
        bs.ops.to_sparse(q, query_layout, block_size), query_layout,
        bs.ops.to_sparse(k, key_layout, block_size), key_layout,
        bs.ops.to_sparse(v, key_layout, block_size), key_layout,
        attention_layout,
        block_size,
        sparsity_layout_o=query_layout,
    )
    output = bs.ops.to_dense(output_sparse, query_layout, block_size)

    assert torch.equal(output, torch.ones_like(output))


def test_flash_attention_long_sequence_backward_preserves_uniform_gradient():
    block_size = 32
    head_dimension = 32
    query_length = 16_384
    key_length = block_size
    query_layout = torch.ones(
        (1, query_length // block_size, 1), dtype=torch.bool, device=DEVICE)
    key_layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    attention_layout = torch.ones(
        (1, query_length // block_size, 1), dtype=torch.bool, device=DEVICE)
    q = torch.zeros(
        (1, query_length, head_dimension), dtype=torch.bfloat16,
        device=DEVICE, requires_grad=True)
    k = torch.zeros(
        (1, key_length, head_dimension), dtype=torch.bfloat16,
        device=DEVICE, requires_grad=True)
    v = torch.ones(
        (1, key_length, head_dimension), dtype=torch.bfloat16,
        device=DEVICE, requires_grad=True)

    output_sparse = bs.ops.flash_attention(
        bs.ops.to_sparse(q, query_layout, block_size), query_layout,
        bs.ops.to_sparse(k, key_layout, block_size), key_layout,
        bs.ops.to_sparse(v, key_layout, block_size), key_layout,
        attention_layout,
        block_size,
        sparsity_layout_o=query_layout,
    )
    output_sparse.sum().backward()

    expected_gradient = query_length / key_length
    assert v.grad is not None
    assert torch.all(v.grad == expected_gradient)


def test_row_wise_sum_uses_float32_accumulation_for_bfloat16():
    block_size = 16
    columns = 16_384
    n_blocks = columns // block_size
    layout = torch.ones((1, 1, n_blocks), dtype=torch.bool, device=DEVICE)
    source = bs.BlksprsTensor.wrap(torch.ones(
        (n_blocks, block_size, block_size), dtype=torch.bfloat16, device=DEVICE))

    output, output_layout = bs.ops.misc.row_wise_sum(
        source, layout, block_size, flag_slice_only=True)

    assert output.dtype == source.dtype
    assert output_layout.shape == (1, 1, 1)
    assert torch.all(output == columns)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_repeat_backward_uses_float32_accumulation(dtype: torch.dtype):
    repeats = 4_096
    layout = _full_layout()
    source = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=dtype,
        device=DEVICE,
        requires_grad=True,
    )

    output, _ = bs.ops.repeat(
        bs.BlksprsTensor.wrap(source), layout,
        (1, 1, repeats), SPARSITY_BLOCK_SIZE)
    output[..., 0, 0].sum().backward()

    assert source.grad is not None
    assert source.grad[0, 0, 0] == repeats
    assert torch.count_nonzero(source.grad) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scatter_sum_uses_float32_accumulation(dtype: torch.dtype):
    columns = 4_096
    n_blocks = columns // SPARSITY_BLOCK_SIZE
    source_layout = _full_layout(columns=n_blocks)
    target_layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (n_blocks, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=dtype,
        device=DEVICE,
    ))
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        source.shape, dtype=torch.int64, device=DEVICE))

    output = bs.ops.scatter_reduce(
        source, source_layout, 2, indices, target_layout,
        SPARSITY_BLOCK_SIZE, reduce_op="sum")

    assert output.dtype == source.dtype
    assert torch.all(output[..., 0] == columns)
    assert torch.count_nonzero(output[..., 1:]) == 0


@pytest.mark.parametrize("flag_fused", [False, True])
def test_softmax_backward_supports_non_contiguous_output_gradient(flag_fused: bool):
    layout = _full_layout()
    values = torch.randn((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    gradient = _non_contiguous_gradient()

    x_non_contiguous = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output_non_contiguous = bs.ops.softmax(
        x_non_contiguous, layout, SPARSITY_BLOCK_SIZE, flag_fused=flag_fused)
    actual = torch.autograd.grad(output_non_contiguous, x_non_contiguous, gradient)[0]

    x_contiguous = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output_contiguous = bs.ops.softmax(
        x_contiguous, layout, SPARSITY_BLOCK_SIZE, flag_fused=flag_fused)
    expected = torch.autograd.grad(output_contiguous, x_contiguous, gradient.contiguous())[0]

    assert torch.allclose(actual, expected)


def test_repeat_backward_supports_non_contiguous_output_gradient():
    layout = _full_layout()
    values = torch.randn((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    gradient = _non_contiguous_gradient(blocks=2)

    x_non_contiguous = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output_non_contiguous, _ = bs.ops.repeat(
        x_non_contiguous, layout, (1, 1, 2), SPARSITY_BLOCK_SIZE)
    actual = torch.autograd.grad(output_non_contiguous, x_non_contiguous, gradient)[0]

    x_contiguous = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output_contiguous, _ = bs.ops.repeat(
        x_contiguous, layout, (1, 1, 2), SPARSITY_BLOCK_SIZE)
    expected = torch.autograd.grad(output_contiguous, x_contiguous, gradient.contiguous())[0]

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("operation", "expected_layout_shape"),
    [
        (lambda x, layout: bs.ops.repeat(
            x, layout, (0, 1, 1), SPARSITY_BLOCK_SIZE), (0, 1, 1)),
        (lambda x, layout: bs.ops.repeat(
            x, layout, (1, 0, 1), SPARSITY_BLOCK_SIZE), (1, 0, 1)),
        (lambda x, layout: bs.ops.repeat_interleave(
            x, layout, 0, SPARSITY_BLOCK_SIZE), (0, 1, 1)),
    ],
)
def test_repeat_operations_accept_zero_repetitions(operation, expected_layout_shape):
    layout = _full_layout()
    source_base = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    )
    source = bs.BlksprsTensor.wrap(source_base)

    output, output_layout = operation(source, layout)

    assert type(output) is bs.BlksprsTensor
    assert output.shape == (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    assert type(output_layout) is torch.Tensor
    assert output_layout.shape == expected_layout_shape
    output.sum().backward()
    assert torch.equal(source_base.grad, torch.zeros_like(source_base))


@pytest.mark.parametrize("repeats", [(-1, 1, 1), (True, 1, 1), (1.5, 1, 1)])
def test_repeat_rejects_invalid_repetitions(repeats):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises((TypeError, ValueError), match="repeats"):
        bs.ops.repeat(source, layout, repeats, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize("repeats", [-1, True, 1.5])
def test_repeat_interleave_rejects_invalid_repetitions(repeats):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises((TypeError, ValueError), match="repeats"):
        bs.ops.repeat_interleave(source, layout, repeats, SPARSITY_BLOCK_SIZE)


def test_flash_attention_backward_supports_non_contiguous_output_gradient():
    layout = _full_layout()
    attention_layout = _full_layout()
    values = [
        torch.randn((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
        for _ in range(3)
    ]
    gradient = _non_contiguous_gradient()

    inputs_non_contiguous = [
        bs.BlksprsTensor.wrap(value.clone().requires_grad_(True)) for value in values
    ]
    output_non_contiguous = bs.ops.flash_attention(
        inputs_non_contiguous[0], layout,
        inputs_non_contiguous[1], layout,
        inputs_non_contiguous[2], layout,
        attention_layout,
        SPARSITY_BLOCK_SIZE,
    )
    actual = torch.autograd.grad(
        output_non_contiguous, inputs_non_contiguous, gradient)

    inputs_contiguous = [
        bs.BlksprsTensor.wrap(value.clone().requires_grad_(True)) for value in values
    ]
    output_contiguous = bs.ops.flash_attention(
        inputs_contiguous[0], layout,
        inputs_contiguous[1], layout,
        inputs_contiguous[2], layout,
        attention_layout,
        SPARSITY_BLOCK_SIZE,
    )
    expected = torch.autograd.grad(
        output_contiguous, inputs_contiguous, gradient.contiguous())

    for actual_gradient, expected_gradient in zip(actual, expected):
        assert torch.allclose(actual_gradient, expected_gradient)


def test_conversion_cache_invalidates_for_another_or_mutated_layout():
    cache = {}
    dense = torch.zeros((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
    dense[:, :, :SPARSITY_BLOCK_SIZE] = 1
    first_layout = torch.tensor([[[True, False]]], device=DEVICE)
    second_layout = torch.tensor([[[False, True]]], device=DEVICE)

    first = bs.ops.to_sparse(dense, first_layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)
    assert torch.all(first == 1)

    dense.fill_(2)
    second = bs.ops.to_sparse(dense, second_layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)
    assert torch.all(second == 2)

    second_layout.fill_(False)
    second_layout[0, 0, 0] = True
    dense.fill_(3)
    mutated = bs.ops.to_sparse(dense, second_layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)
    assert torch.all(mutated == 3)


def test_conversion_cache_rebuilds_safely_for_inference_tensors():
    cache = {}

    with torch.inference_mode():
        dense = torch.zeros(
            (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
        dense[:, :, :SPARSITY_BLOCK_SIZE] = 1
        layout = torch.tensor([[[True, False]]], device=DEVICE)

        first = bs.ops.to_sparse(
            dense, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)
        first_layout_indices = cache["layout_indices"]

        dense.fill_(0)
        dense[:, :, SPARSITY_BLOCK_SIZE:] = 2
        layout.logical_not_()
        second = bs.ops.to_sparse(
            dense, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)

    assert torch.all(first == 1)
    assert torch.all(second == 2)
    assert cache["layout_indices"] is not first_layout_indices


@pytest.mark.parametrize("operation", ["repeat", "repeat_interleave"])
def test_derived_repeat_layout_mutation_invalidates_cache(operation: str):
    layout = _full_layout(columns=2)
    source = bs.BlksprsTensor.wrap(torch.randn(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    cache = {}

    if operation == "repeat":
        _, returned_layout = bs.ops.repeat(
            source, layout, (1, 1, 1), SPARSITY_BLOCK_SIZE, layout_cache=cache)
    else:
        _, returned_layout = bs.ops.repeat_interleave(
            source, layout, 1, SPARSITY_BLOCK_SIZE, layout_cache=cache)

    returned_layout[..., 1] = False
    stale_layout = returned_layout

    if operation == "repeat":
        output, returned_layout = bs.ops.repeat(
            source, layout, (1, 1, 1), SPARSITY_BLOCK_SIZE, layout_cache=cache)
    else:
        output, returned_layout = bs.ops.repeat_interleave(
            source, layout, 1, SPARSITY_BLOCK_SIZE, layout_cache=cache)

    assert returned_layout is not stale_layout
    assert torch.equal(returned_layout, layout)
    assert output.size(0) == 2
    assert bs.ops.to_dense(output, returned_layout, SPARSITY_BLOCK_SIZE).shape == (
        1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2)


def test_derived_flash_output_layout_mutation_invalidates_cache():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    cache = {}

    bs.ops.flash_attention(
        source, layout,
        source, layout,
        source, layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        layout_cache=cache,
    )
    stale_layout = cache["sparsity_layout_o"]
    stale_layout.zero_()

    output = bs.ops.flash_attention(
        source, layout,
        source, layout,
        source, layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        layout_cache=cache,
    )
    returned_layout = cache["sparsity_layout_o"]

    assert returned_layout is not stale_layout
    assert torch.equal(returned_layout, layout)
    assert output.size(0) == 1
    assert bs.ops.to_dense(output, returned_layout, SPARSITY_BLOCK_SIZE).shape == (
        1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)


def test_transformed_output_layouts_do_not_alias_inputs():
    operations = (
        lambda source, layout: bs.ops.transpose(
            source, layout, SPARSITY_BLOCK_SIZE),
        lambda source, layout: bs.ops.split(
            source, layout, 1, -1, SPARSITY_BLOCK_SIZE),
        lambda source, layout: bs.ops.merge(
            source, layout, 1, -1, SPARSITY_BLOCK_SIZE),
    )

    for operation in operations:
        input_layout = _full_layout()
        source = bs.BlksprsTensor.wrap(torch.randn(
            (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

        _, output_layout = operation(source, input_layout)

        assert output_layout.data_ptr() != input_layout.data_ptr()
        output_layout.zero_()
        assert torch.all(input_layout)


def test_flash_attention_reuses_prebuilt_layout_cache_with_resolved_defaults():
    layout = _full_layout()
    cache = bs.ops.flash_attention_build_layout_cache(
        layout,
        sparsity_layout_q=layout,
        sparsity_layout_k=layout,
        sparsity_layout_v=layout,
        n_seq_blocks_q=1,
        n_seq_blocks_k=1,
        n_head_blocks=1,
    )
    key_indices = cache["key_indices"]
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    bs.ops.flash_attention(
        source, layout,
        source, layout,
        source, layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        layout_cache=cache,
    )

    assert cache["key_indices"] is key_indices


def test_flash_attention_cache_builder_preserves_positional_cache_argument():
    layout = _full_layout()
    cache = {}

    returned_cache = bs.ops.flash_attention_build_layout_cache(
        layout, layout, layout, layout, 1, 1, 1, None, 1, cache)

    assert returned_cache is cache
    assert torch.equal(cache["sparsity_layout_o"], layout)


def test_flash_attention_default_output_layout_follows_attention_and_values():
    qk_layout = torch.tensor([[[True, False]]], device=DEVICE)
    value_layout = torch.tensor([[[False, True]]], device=DEVICE)
    attention_layout = _full_layout()

    q_dense = torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
    k_dense = torch.zeros_like(q_dense)
    v_dense = torch.zeros_like(q_dense)
    q_dense[..., :SPARSITY_BLOCK_SIZE] = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    k_dense[..., :SPARSITY_BLOCK_SIZE] = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    v_dense[..., SPARSITY_BLOCK_SIZE:] = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    q = bs.ops.to_sparse(q_dense, qk_layout, SPARSITY_BLOCK_SIZE).requires_grad_(True)
    k = bs.ops.to_sparse(k_dense, qk_layout, SPARSITY_BLOCK_SIZE).requires_grad_(True)
    v = bs.ops.to_sparse(v_dense, value_layout, SPARSITY_BLOCK_SIZE).requires_grad_(True)

    cache = {}
    output = bs.ops.flash_attention(
        q, qk_layout,
        k, qk_layout,
        v, value_layout,
        attention_layout,
        SPARSITY_BLOCK_SIZE,
        layout_cache=cache,
    )
    output_layout = bs.layouting.build_sparsity_layout_matmul(
        attention_layout, value_layout)
    output_dense = bs.ops.to_dense(
        output, output_layout, SPARSITY_BLOCK_SIZE, fill_value=0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_dense, k_dense, v_dense)

    assert torch.equal(cache["sparsity_layout_o"], output_layout)
    assert torch.allclose(output_dense, expected, atol=2e-3, rtol=2e-3)
    assert torch.count_nonzero(output_dense[..., :SPARSITY_BLOCK_SIZE]) == 0
    assert torch.count_nonzero(output_dense[..., SPARSITY_BLOCK_SIZE:]) > 0

    output.square().sum().backward()
    assert torch.count_nonzero(q.grad) > 0
    assert torch.count_nonzero(k.grad) > 0
    assert torch.count_nonzero(v.grad) > 0


def test_linear_cache_invalidates_for_another_input_layout():
    first_layout = torch.tensor([[[True], [False]]], device=DEVICE)
    second_layout = torch.tensor([[[False], [True]]], device=DEVICE)
    linear = torch.nn.Linear(SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE, bias=False, device=DEVICE)
    cache = {}

    dense = torch.zeros((1, SPARSITY_BLOCK_SIZE * 2, SPARSITY_BLOCK_SIZE), device=DEVICE)
    dense[:, :SPARSITY_BLOCK_SIZE] = 1
    first_sparse = bs.ops.to_sparse(dense, first_layout, SPARSITY_BLOCK_SIZE)
    bs.utils.apply_torch_linear_cached(
        first_sparse, first_layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)

    dense.fill_(0)
    dense[:, SPARSITY_BLOCK_SIZE:] = 2
    second_sparse = bs.ops.to_sparse(dense, second_layout, SPARSITY_BLOCK_SIZE)
    actual, actual_layout = bs.utils.apply_torch_linear_cached(
        second_sparse, second_layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    expected, expected_layout = bs.utils.apply_torch_linear_cached(
        second_sparse, second_layout, SPARSITY_BLOCK_SIZE, linear, layout_cache={})

    assert torch.equal(actual_layout, expected_layout)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
@pytest.mark.parametrize(("initial_state", "updated_state"), [(False, True), (True, False)])
def test_linear_cache_invalidates_after_parameter_gradient_state_changes(
        parameter_name: str, initial_state: bool, updated_state: bool):
    layout = _full_layout()
    dense = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    )
    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    linear = torch.nn.Linear(
        SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE, bias=True, device=DEVICE)
    linear.weight.requires_grad_(parameter_name == "weight" and initial_state)
    linear.bias.requires_grad_(parameter_name == "bias" and initial_state)
    cache = {}

    bs.utils.apply_torch_linear_cached(
        sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    cached_value_name = "w_t_bs" if parameter_name == "weight" else "bias_slice_bs"
    initial_cached_value = cache[cached_value_name]

    parameter = getattr(linear, parameter_name)
    parameter.requires_grad_(updated_state)
    parameter.grad = None
    actual, actual_layout = bs.utils.apply_torch_linear_cached(
        sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    expected, expected_layout = bs.utils.apply_torch_linear_cached(
        sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache={})

    assert cache[cached_value_name] is not initial_cached_value
    assert cache[cached_value_name].requires_grad is updated_state
    assert torch.equal(actual_layout, expected_layout)
    assert torch.allclose(actual, expected)

    actual.sum().backward()
    assert (parameter.grad is not None) is updated_state


@pytest.mark.parametrize("use_bias", [False, True])
def test_linear_cache_rebuilds_after_gradient_disabled_evaluation(use_bias: bool):
    layout = _full_layout()
    dense = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    )
    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    linear = torch.nn.Linear(
        SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE,
        bias=use_bias, device=DEVICE)
    cache = {}

    with torch.no_grad():
        bs.utils.apply_torch_linear_cached(
            sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    evaluation_weight = cache["w_t_bs"]
    evaluation_bias = cache.get("bias_slice_bs")

    output, _ = bs.utils.apply_torch_linear_cached(
        sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    output.sum().backward()

    assert cache["w_t_bs"] is not evaluation_weight
    assert cache["w_t_bs"].requires_grad
    assert dense.grad is not None
    assert linear.weight.grad is not None
    if use_bias:
        assert cache["bias_slice_bs"] is not evaluation_bias
        assert cache["bias_slice_bs"].requires_grad
        assert linear.bias is not None
        assert linear.bias.grad is not None


@pytest.mark.parametrize("cached", [False, True])
def test_sparse_linear_rejects_mixed_dtypes_outside_autocast(cached: bool):
    layout = _full_layout()
    dense = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.float16,
        device=DEVICE,
    )
    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    linear = torch.nn.Linear(
        SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE,
        dtype=torch.float32, device=DEVICE)

    with pytest.raises(ValueError, match="same dtype"):
        if cached:
            bs.utils.apply_torch_linear_cached(
                sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache={})
        else:
            bs.utils.apply_torch_linear(
                sparse, layout, SPARSITY_BLOCK_SIZE, linear)


@pytest.mark.parametrize("cached", [False, True])
def test_sparse_linear_bias_preserves_autocast_output_dtype(cached: bool):
    layout = _full_layout()
    linear = torch.nn.Linear(
        SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE, bias=True, device=DEVICE)
    cache = {}

    # Exercise a reused cache as well as a fresh autograd graph. Cached packed
    # parameters must remain differentiable across training iterations.
    for _ in range(2):
        dense = torch.randn(
            (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE,
            requires_grad=True)
        sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            expected = linear(dense)
            if cached:
                actual, actual_layout = bs.utils.apply_torch_linear_cached(
                    sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
            else:
                actual, actual_layout = bs.utils.apply_torch_linear(
                    sparse, layout, SPARSITY_BLOCK_SIZE, linear)
            actual_dense = bs.ops.to_dense(
                actual, actual_layout, SPARSITY_BLOCK_SIZE)

        gradient = torch.randn_like(expected)
        gradient_inputs = (dense, linear.weight, linear.bias)
        expected_gradients = torch.autograd.grad(
            expected, gradient_inputs, gradient, retain_graph=True)
        actual_gradients = torch.autograd.grad(
            actual_dense, gradient_inputs, gradient)

        assert actual.dtype == expected.dtype
        assert torch.allclose(actual_dense, expected, atol=2e-2, rtol=1e-2)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            assert torch.allclose(actual_gradient, expected_gradient, atol=2e-2, rtol=1e-2)


def test_autocast_does_not_silently_downcast_float64_inputs():
    layout = _full_layout()
    compressed = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.float64,
        device=DEVICE,
    ))

    with torch.amp.autocast(device_type="cuda"):
        with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
            bs.ops.matmul(
                compressed,
                layout,
                compressed,
                layout,
                layout,
                SPARSITY_BLOCK_SIZE,
            )
        with pytest.raises(ValueError, match="float16, bfloat16, or float32"):
            bs.ops.flash_attention(
                compressed,
                layout,
                compressed,
                layout,
                compressed,
                layout,
                layout,
                SPARSITY_BLOCK_SIZE,
            )

        broadcast_output = bs.ops.misc.broadcast_add(
            torch.ones(
                (1, SPARSITY_BLOCK_SIZE),
                dtype=torch.float64,
                device=DEVICE,
            ),
            torch.ones(
                (1, SPARSITY_BLOCK_SIZE),
                dtype=torch.float64,
                device=DEVICE,
            ),
            layout,
            SPARSITY_BLOCK_SIZE,
        )

    assert broadcast_output.dtype == torch.float64


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("batch_size", "row_blocks", "input_features"),
    [
        (1, 2, 0),
        (1, 0, SPARSITY_BLOCK_SIZE),
        (0, 2, SPARSITY_BLOCK_SIZE),
    ],
)
def test_sparse_linear_bias_matches_torch_for_empty_dimensions(
        cached: bool, batch_size: int, row_blocks: int, input_features: int):
    column_blocks = input_features // SPARSITY_BLOCK_SIZE
    layout = torch.ones(
        (batch_size, row_blocks, column_blocks), dtype=torch.bool, device=DEVICE)
    dense = torch.empty(
        (batch_size, row_blocks * SPARSITY_BLOCK_SIZE, input_features),
        device=DEVICE,
        requires_grad=True,
    )
    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    linear = torch.nn.Linear(
        input_features, SPARSITY_BLOCK_SIZE, bias=True, device=DEVICE)
    cache = {}

    expected = linear(dense)
    if cached:
        actual, actual_layout = bs.utils.apply_torch_linear_cached(
            sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    else:
        actual, actual_layout = bs.utils.apply_torch_linear(
            sparse, layout, SPARSITY_BLOCK_SIZE, linear)
    actual_dense = bs.ops.to_dense(
        actual, actual_layout, SPARSITY_BLOCK_SIZE)

    gradient = torch.randn_like(expected)
    gradient_inputs = (dense, linear.weight, linear.bias)
    expected_gradients = torch.autograd.grad(
        expected, gradient_inputs, gradient, retain_graph=True)
    actual_gradients = torch.autograd.grad(
        actual_dense, gradient_inputs, gradient)

    assert type(actual_layout) is torch.Tensor
    assert actual_layout.shape == (batch_size, row_blocks, 1)
    assert torch.all(actual_layout)
    assert torch.allclose(actual_dense, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_gradient, expected_gradient)


@pytest.mark.parametrize("index_value", [-1, SPARSITY_BLOCK_SIZE])
def test_distribution_operations_reject_out_of_bounds_indices(index_value: int):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    indices = bs.BlksprsTensor.wrap(torch.full(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), index_value,
        dtype=torch.int64, device=DEVICE))

    with pytest.raises(IndexError):
        bs.ops.gather(source, layout, 2, indices, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(IndexError):
        bs.ops.scatter_reduce(source, layout, 2, indices, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(IndexError):
        bs.layouting.build_distribution_layout(
            indices, layout, 2,
            torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
            SPARSITY_BLOCK_SIZE,
        )


@pytest.mark.parametrize(
    ("indexed_dimension", "oversized_dimension"),
    [(indexed_dimension, oversized_dimension)
     for indexed_dimension in range(3)
     for oversized_dimension in range(3)
     if oversized_dimension != indexed_dimension],
)
def test_distribution_operations_reject_oversized_non_indexed_dimensions(
        indexed_dimension: int, oversized_dimension: int):
    small_layout = _full_layout()
    oversized_layout_shape = [1, 1, 1]
    oversized_layout_shape[oversized_dimension] = 2
    oversized_layout = torch.ones(
        oversized_layout_shape, dtype=torch.bool, device=DEVICE)
    small_source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    oversized_source = bs.BlksprsTensor.wrap(torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    oversized_indices = bs.BlksprsTensor.wrap(torch.zeros(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=torch.int64, device=DEVICE))

    with pytest.raises(ValueError, match=f"dimension {oversized_dimension}"):
        bs.ops.gather(
            small_source, small_layout, indexed_dimension,
            oversized_indices, oversized_layout,
            SPARSITY_BLOCK_SIZE,
        )

    with pytest.raises(ValueError, match=f"dimension {oversized_dimension}"):
        bs.ops.scatter_reduce(
            oversized_source, oversized_layout, indexed_dimension,
            oversized_indices, small_layout,
            SPARSITY_BLOCK_SIZE,
        )

    with pytest.raises(ValueError, match=f"dimension {oversized_dimension}"):
        bs.layouting.build_distribution_layout(
            oversized_indices, oversized_layout, indexed_dimension,
            torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
            SPARSITY_BLOCK_SIZE,
        )


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64])
def test_distribution_layout_accepts_integer_indices(dtype: torch.dtype):
    layout = _full_layout()
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE))

    actual = bs.layouting.build_distribution_layout(
        indices, layout, 2,
        torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
        SPARSITY_BLOCK_SIZE,
    )

    assert torch.equal(actual, layout)


@pytest.mark.parametrize("dtype", [torch.bool, torch.float16, torch.float32])
def test_distribution_layout_rejects_non_integer_indices(dtype: torch.dtype):
    layout = _full_layout()
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE))

    with pytest.raises(ValueError, match="integer dtype"):
        bs.layouting.build_distribution_layout(
            indices, layout, 2,
            torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
            SPARSITY_BLOCK_SIZE,
        )


@pytest.mark.parametrize(
    ("dim", "target_size"),
    [
        (0, (-1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
        (1, (1, -SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
        (2, (1, SPARSITY_BLOCK_SIZE, -SPARSITY_BLOCK_SIZE)),
    ],
)
def test_distribution_layout_rejects_negative_target_dimensions(
        dim: int, target_size: tuple[int, int, int]):
    layout = torch.zeros((1, 1, 1), dtype=torch.bool, device=DEVICE)
    indices = bs.BlksprsTensor.wrap(torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.int64,
        device=DEVICE,
    ))

    with pytest.raises(ValueError, match="non-negative integers"):
        bs.layouting.build_distribution_layout(
            indices,
            layout,
            dim,
            torch.Size(target_size),
            SPARSITY_BLOCK_SIZE,
        )


@pytest.mark.parametrize("dim", [-4, 3, 4])
def test_distribution_operations_reject_invalid_dimensions(dim: int):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.int64, device=DEVICE))

    with pytest.raises(IndexError):
        bs.ops.gather(source, layout, dim, indices, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(IndexError):
        bs.ops.scatter_reduce(source, layout, dim, indices, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(IndexError):
        bs.layouting.build_distribution_layout(
            indices, layout, dim,
            torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
            SPARSITY_BLOCK_SIZE,
        )


@pytest.mark.parametrize("dim", [-4, 3, 4])
def test_partition_operations_reject_invalid_dimensions(dim: int):
    layout = _full_layout(columns=2)
    source = bs.BlksprsTensor.wrap(torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises(IndexError):
        bs.ops.split(source, layout, 2, dim, SPARSITY_BLOCK_SIZE)
    with pytest.raises(IndexError):
        bs.ops.merge(source, layout, 1, dim, SPARSITY_BLOCK_SIZE)


def test_flash_attention_rejects_non_binary_layouts():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    invalid_layout = torch.full((1, 1, 1), 0.5, device=DEVICE)

    with pytest.raises(ValueError, match="0 or 1"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            invalid_layout, SPARSITY_BLOCK_SIZE,
        )
    with pytest.raises(ValueError, match="0 or 1"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            layout, SPARSITY_BLOCK_SIZE, sparsity_layout_o=invalid_layout,
        )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_sparsity_layouts_are_rejected(dtype: torch.dtype):
    layout = torch.ones((1, 1, 1), dtype=dtype, device=DEVICE)
    source = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    with pytest.raises(ValueError, match="complex dtype"):
        bs.ops.to_sparse(source, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="complex dtype"):
        bs.layouting.build_sparsity_layout_matmul(layout, layout)


def test_flash_attention_requires_optional_tensor_layout_pairs():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises(ValueError, match="provided together"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            layout, SPARSITY_BLOCK_SIZE, sparsity_layout_mask=layout,
        )
    with pytest.raises(ValueError, match="provided together"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            layout, SPARSITY_BLOCK_SIZE, sparsity_layout_bias=layout,
        )


def test_flash_attention_rejects_non_binary_mask_values():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    invalid_mask = bs.BlksprsTensor.wrap(torch.full(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), 2.0, device=DEVICE))

    with pytest.raises(ValueError, match="0 or 1"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            layout, SPARSITY_BLOCK_SIZE,
            attention_mask=invalid_mask,
            sparsity_layout_mask=layout,
        )


def test_flash_attention_accepts_boolean_mask_values():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    attention_mask = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=torch.bool, device=DEVICE))

    output = bs.ops.flash_attention(
        source, layout, source, layout, source, layout,
        layout, SPARSITY_BLOCK_SIZE,
        attention_mask=attention_mask,
        sparsity_layout_mask=layout,
    )

    assert output.shape == source.shape


@pytest.mark.parametrize("sparsity_block_size", [128, 256])
def test_flash_attention_rejects_unsupported_block_sizes(sparsity_block_size: int):
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, sparsity_block_size, sparsity_block_size), device=DEVICE))

    with pytest.raises(ValueError, match="at most 64"):
        bs.ops.flash_attention(
            source, layout, source, layout, source, layout,
            layout, sparsity_block_size,
        )


def test_flash_attention_segmented_metadata_uses_smallest_safe_dtype():
    layout = _full_layout()
    cache = bs.ops.flash_attention_build_layout_cache(layout)

    assert cache["key_indices"].dtype == torch.int32
    assert cache["key_offsets"].dtype == torch.int32
    assert cache["query_indices"].dtype == torch.int32
    assert cache["query_offsets"].dtype == torch.int32
    assert _select_segmented_index_dtype(2_147_483_647, 2_147_483_648) == torch.int32
    assert _select_segmented_index_dtype(2_147_483_648, 1) == torch.int64
    assert _select_segmented_index_dtype(1, 2_147_483_649) == torch.int64


def test_flash_attention_segmented_metadata_counts_numeric_layouts_exactly():
    n_columns = 65_536
    layout = torch.ones((1, 1, n_columns), dtype=torch.float16, device=DEVICE)

    cache = bs.ops.flash_attention_build_layout_cache(layout)

    assert cache["key_indices"].numel() == n_columns
    assert cache["key_offsets"][-1].item() == n_columns


def test_large_numeric_layout_block_counts_are_exact():
    n_columns = 65_536
    layout = torch.ones((1, 1, n_columns), dtype=torch.float16, device=DEVICE)

    conversion_cache = to_sparse_build_layout_cache({}, layout)
    flash_cache = bs.ops.flash_attention_build_layout_cache(
        _full_layout(),
        sparsity_layout_q=layout,
        n_head_blocks=n_columns,
        sparsity_layout_o=layout,
        n_head_blocks_v=n_columns,
    )

    assert conversion_cache["n_sparse_blocks"] == n_columns
    assert flash_cache["n_sparse_blocks_q"] == n_columns
    assert flash_cache["n_sparse_blocks_o"] == n_columns


def test_adapt_layout_rejects_a_different_logical_shape():
    source_layout = _full_layout(columns=2)
    target_layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises(ValueError, match="Target sparsity layout"):
        bs.ops.adapt_layout(
            source,
            source_layout,
            SPARSITY_BLOCK_SIZE,
            SPARSITY_BLOCK_SIZE,
            sparsity_layout_to=target_layout,
        )


@pytest.mark.parametrize("layout_shape", [(1, 3, 4), (1, 4, 3), (1, 3, 3)])
@pytest.mark.parametrize("provide_target_layout", [False, True])
def test_adapt_layout_backward_handles_non_divisible_coarsening(
        layout_shape: tuple[int, int, int], provide_target_layout: bool):
    source_block_size = 16
    target_block_size = 32
    source_layout = torch.ones(layout_shape, dtype=torch.bool, device=DEVICE)
    source_base = torch.randn(
        (source_layout.numel(), source_block_size, source_block_size),
        device=DEVICE,
        requires_grad=True,
    )
    source = bs.BlksprsTensor.wrap(source_base)
    target_layout = None
    if provide_target_layout:
        target_layout = bs.layouting.build_sparsity_layout_adaption(
            source, source_layout, source_block_size, target_block_size)

    output, output_layout = bs.ops.adapt_layout(
        source,
        source_layout,
        source_block_size,
        target_block_size,
        sparsity_layout_to=target_layout,
    )

    assert output_layout.shape == (
        layout_shape[0],
        (layout_shape[1] + 1) // 2,
        (layout_shape[2] + 1) // 2,
    )
    output.sum().backward()
    assert torch.equal(source_base.grad, torch.ones_like(source_base))


@pytest.mark.parametrize("layout_shape", [(0, 3, 3), (1, 0, 3), (1, 3, 0)])
def test_adapt_layout_backward_handles_empty_non_divisible_coarsening(
        layout_shape: tuple[int, int, int]):
    source_block_size = 16
    target_block_size = 32
    source_layout = torch.zeros(layout_shape, dtype=torch.bool, device=DEVICE)
    source_base = torch.empty(
        (0, source_block_size, source_block_size),
        device=DEVICE,
        requires_grad=True,
    )

    output, _ = bs.ops.adapt_layout(
        bs.BlksprsTensor.wrap(source_base),
        source_layout,
        source_block_size,
        target_block_size,
    )

    output.sum().backward()
    assert source_base.grad is not None
    assert source_base.grad.shape == source_base.shape


def test_dense_conversion_rejects_a_different_layout_batch_size():
    layout = _full_layout()
    dense = torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    with pytest.raises(ValueError, match="batch dimension"):
        bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_row_wise_max_preserves_dtype(dtype: torch.dtype):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE))

    output, _ = bs.ops.misc.row_wise_max(source, layout, SPARSITY_BLOCK_SIZE)

    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_row_wise_max_preserves_negative_infinity(dtype: torch.dtype):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.full(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        float("-inf"), dtype=dtype, device=DEVICE))

    output, _ = bs.ops.misc.row_wise_max(source, layout, SPARSITY_BLOCK_SIZE)

    assert output.dtype == dtype
    assert torch.all(torch.isneginf(output))


@pytest.mark.parametrize("flag_fused", [False, True])
@pytest.mark.parametrize("layout_shape", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_softmax_accepts_empty_layouts(flag_fused: bool, layout_shape: tuple[int, int, int]):
    layout = torch.zeros(layout_shape, dtype=torch.bool, device=DEVICE)
    source = bs.BlksprsTensor.wrap(torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    output = bs.ops.softmax(
        source, layout, SPARSITY_BLOCK_SIZE, flag_fused=flag_fused)

    assert type(output) is bs.BlksprsTensor
    assert output.shape == source.shape


@pytest.mark.parametrize(
    "builder",
    [
        bs.layouting.build_sparsity_layout_matmul,
        bs.layouting.build_sparsity_layout_matmul_fast,
        bs.layouting.build_sparsity_layout_matmul_outer,
    ],
)
def test_layout_matmul_helpers_accept_empty_inner_dimensions(builder):
    left = torch.empty((1, 2, 0), dtype=torch.bool, device=DEVICE)
    right = torch.empty((1, 0, 3), dtype=torch.bool, device=DEVICE)

    output = builder(left, right)

    assert output.shape == (1, 2, 3)
    assert not torch.any(output)


@pytest.mark.parametrize("operation", [bs.ops.misc.row_wise_sum, bs.ops.misc.row_wise_max])
@pytest.mark.parametrize("flag_slice_only", [False, True])
def test_row_wise_reductions_accept_empty_logical_columns(operation, flag_slice_only: bool):
    layout = torch.empty((1, 2, 0), dtype=torch.bool, device=DEVICE)
    source = bs.BlksprsTensor.wrap(torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    output, output_layout = operation(
        source, layout, SPARSITY_BLOCK_SIZE, flag_slice_only=flag_slice_only)

    expected_type = torch.Tensor if flag_slice_only else bs.BlksprsTensor
    assert type(output) is expected_type
    assert output.shape == (
        0,
        SPARSITY_BLOCK_SIZE,
        1 if flag_slice_only else SPARSITY_BLOCK_SIZE,
    )
    assert output_layout.shape == (1, 2, 1)
    assert not torch.any(output_layout)


@pytest.mark.parametrize("operation", [bs.ops.misc.row_wise_sum, bs.ops.misc.row_wise_max])
def test_slice_only_row_wise_reductions_return_auxiliary_base_tensors(operation):
    layout = _full_layout(columns=2)
    source = bs.BlksprsTensor.wrap(torch.randn(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    output, output_layout = operation(
        source, layout, SPARSITY_BLOCK_SIZE, flag_slice_only=True)
    arithmetic_output = bs.ops.misc.row_wise_add(
        source, layout, output, SPARSITY_BLOCK_SIZE)

    assert type(output) is torch.Tensor
    assert output.shape == (1, SPARSITY_BLOCK_SIZE, 1)
    assert output_layout.shape == (1, 1, 1)
    assert type(arithmetic_output) is bs.BlksprsTensor


def test_flash_attention_default_scale_accepts_zero_qk_head_dimension():
    qk_layout = torch.empty((1, 1, 0), dtype=torch.bool, device=DEVICE)
    value_layout = _full_layout()
    attention_layout = _full_layout()
    q = bs.BlksprsTensor.wrap(torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    k = bs.BlksprsTensor.wrap(torch.empty_like(q))
    v = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    output = bs.ops.flash_attention(
        q, qk_layout,
        k, qk_layout,
        v, value_layout,
        attention_layout,
        SPARSITY_BLOCK_SIZE,
    )
    reference = bs.ops.flash_attention(
        q, qk_layout,
        k, qk_layout,
        v, value_layout,
        attention_layout,
        SPARSITY_BLOCK_SIZE,
        scale=1.0,
    )

    assert torch.all(torch.isfinite(output))
    assert torch.equal(output, reference)


def test_row_wise_add_rejects_mixed_dtypes():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.float16, device=DEVICE))
    row_values = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, 1), dtype=torch.float32, device=DEVICE)

    with pytest.raises(ValueError, match="same dtype"):
        bs.ops.misc.row_wise_add(source, layout, row_values, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize("operation", [bs.ops.misc.row_wise_add, bs.ops.misc.row_wise_sub])
def test_row_wise_arithmetic_accepts_empty_logical_columns(operation):
    layout = torch.empty((1, 2, 0), dtype=torch.bool, device=DEVICE)
    source = bs.BlksprsTensor.wrap(torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    row_values = torch.empty(
        (0, SPARSITY_BLOCK_SIZE, 1), device=DEVICE)

    output = operation(source, layout, row_values, SPARSITY_BLOCK_SIZE)

    assert type(output) is bs.BlksprsTensor
    assert output.shape == source.shape


def test_single_block_to_dense_returns_base_tensor_and_preserves_autograd():
    layout = _full_layout()
    source_base = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE, requires_grad=True)
    source = bs.BlksprsTensor.wrap(source_base)

    output = bs.ops.to_dense(source, layout, SPARSITY_BLOCK_SIZE)

    assert type(output) is torch.Tensor
    assert output.data_ptr() == source.data_ptr()
    output.sum().backward()
    assert torch.equal(source_base.grad, torch.ones_like(source_base))


@pytest.mark.parametrize(
    "builder",
    [
        bs.layouting.build_sparsity_layout_matmul,
        bs.layouting.build_sparsity_layout_matmul_fast,
        bs.layouting.build_sparsity_layout_matmul_outer,
    ],
)
def test_layout_matmul_helpers_reject_incompatible_or_invalid_layouts(builder):
    valid = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    other_batch = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    other_inner = torch.ones((1, 2, 1), dtype=torch.bool, device=DEVICE)
    invalid = torch.full((1, 1, 1), 0.5, device=DEVICE)

    with pytest.raises(ValueError, match="Batch dimensions"):
        builder(valid, other_batch)
    with pytest.raises(ValueError, match="Inner dimensions"):
        builder(valid, other_inner)
    with pytest.raises(ValueError, match="0 or 1"):
        builder(invalid, valid)


def test_generic_layout_kernels_reject_unsupported_dtypes():
    layout = _full_layout()
    dense = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.complex64, device=DEVICE)
    compressed = bs.BlksprsTensor.wrap(dense)

    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.layouting.build_sparsity_layout(dense, SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.layouting.build_sparsity_layout_adaption(
            compressed, layout, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.transpose(compressed, layout, SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.repeat(compressed, layout, (1, 1, 1), SPARSITY_BLOCK_SIZE)
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.repeat_interleave(compressed, layout, 1, SPARSITY_BLOCK_SIZE)

    split_layout = torch.ones((1, 1, 2), dtype=torch.bool, device=DEVICE)
    split_source = bs.BlksprsTensor.wrap(torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.complex64, device=DEVICE))
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.split(split_source, split_layout, 2, 2, SPARSITY_BLOCK_SIZE)

    merge_layout = torch.ones((2, 1, 1), dtype=torch.bool, device=DEVICE)
    merge_source = bs.BlksprsTensor.wrap(torch.ones(
        (2, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.complex64, device=DEVICE))
    with pytest.raises(ValueError, match="dtype is not supported"):
        bs.ops.merge(merge_source, merge_layout, 2, 2, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_generic_data_movement_preserves_supported_dtypes(dtype: torch.dtype):
    layout = torch.ones((1, 1, 2), dtype=torch.bool, device=DEVICE)
    dense = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2),
        dtype=dtype, device=DEVICE)

    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    round_trip = bs.ops.to_dense(sparse, layout, SPARSITY_BLOCK_SIZE)
    transposed, transposed_layout = bs.ops.transpose(
        sparse, layout, SPARSITY_BLOCK_SIZE)
    repeated, repeated_layout = bs.ops.repeat(
        sparse, layout, (1, 1, 2), SPARSITY_BLOCK_SIZE)
    interleaved, interleaved_layout = bs.ops.repeat_interleave(
        sparse, layout, 2, SPARSITY_BLOCK_SIZE)
    split, split_layout = bs.ops.split(
        sparse, layout, 2, 2, SPARSITY_BLOCK_SIZE)
    merged, merged_layout = bs.ops.merge(
        split, split_layout, 2, 2, SPARSITY_BLOCK_SIZE)
    dense_indices = torch.arange(
        SPARSITY_BLOCK_SIZE * 2, dtype=torch.int64, device=DEVICE,
    ).view(1, 1, -1).expand_as(dense).contiguous()
    sparse_indices = bs.ops.to_sparse(
        dense_indices, layout, SPARSITY_BLOCK_SIZE)
    gathered = bs.ops.gather(
        sparse, layout, 2, sparse_indices, layout, SPARSITY_BLOCK_SIZE)
    scattered = bs.ops.scatter(
        sparse, layout, 2, sparse_indices, layout, SPARSITY_BLOCK_SIZE)
    adapted_layout = bs.layouting.build_sparsity_layout_adaption(
        sparse, layout, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)

    assert sparse.dtype == dtype
    assert round_trip.dtype == dtype
    assert torch.equal(round_trip, dense)
    assert transposed.dtype == dtype
    assert transposed_layout.shape == (1, 2, 1)
    assert repeated.dtype == dtype
    assert repeated_layout.shape == (1, 1, 4)
    assert interleaved.dtype == dtype
    assert interleaved_layout.shape == (2, 1, 2)
    assert merged.dtype == dtype
    assert torch.equal(merged, sparse)
    assert torch.equal(merged_layout, layout)
    assert gathered.dtype == dtype
    assert torch.equal(gathered, sparse)
    assert scattered.dtype == dtype
    assert torch.equal(scattered, sparse)
    assert torch.equal(adapted_layout, layout)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16])
def test_scatter_without_reduction_supports_compact_dtypes(dtype: torch.dtype):
    layout = _full_layout()
    values = torch.arange(
        SPARSITY_BLOCK_SIZE ** 2, dtype=torch.int64, device=DEVICE,
    ).reshape(1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    if dtype == torch.bool:
        values = values.remainder(2).bool()
    else:
        values = values.to(dtype)
    source = bs.BlksprsTensor.wrap(values)
    indices = bs.BlksprsTensor.wrap(torch.arange(
        SPARSITY_BLOCK_SIZE, dtype=torch.int64, device=DEVICE,
    ).view(1, 1, -1).expand_as(values).contiguous())

    output = bs.ops.scatter(
        source, layout, 2, indices, layout, SPARSITY_BLOCK_SIZE)

    assert output.dtype == dtype
    assert torch.equal(output, source)


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16])
def test_scatter_sum_rejects_compact_dtypes(dtype: torch.dtype):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE))
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.int64, device=DEVICE))

    with pytest.raises(ValueError, match="does not support"):
        bs.ops.scatter_reduce(
            source, layout, 2, indices, layout, SPARSITY_BLOCK_SIZE,
            reduce_op="sum",
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_scatter_sum_supports_documented_dtypes(dtype: torch.dtype):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE))
    indices = bs.BlksprsTensor.wrap(torch.arange(
        SPARSITY_BLOCK_SIZE, dtype=torch.int64, device=DEVICE,
    ).view(1, 1, -1).expand_as(source).contiguous())

    output = bs.ops.scatter_reduce(
        source, layout, 2, indices, layout, SPARSITY_BLOCK_SIZE,
        reduce_op="sum",
    )

    assert output.dtype == dtype
    assert torch.equal(output, source)


def test_broadcast_rejects_mixed_dtypes():
    layout = _full_layout()
    x = torch.randn((1, SPARSITY_BLOCK_SIZE), dtype=torch.float16, device=DEVICE)
    y = torch.randn((1, SPARSITY_BLOCK_SIZE), dtype=torch.float32, device=DEVICE)

    with pytest.raises(ValueError, match="same dtype"):
        bs.ops.misc.broadcast_add(x, y, layout, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (bs.ops.misc.broadcast_add, torch.add),
        (bs.ops.misc.broadcast_sub, torch.sub),
    ],
)
def test_broadcast_operations_support_rectangular_outputs(operation, reference):
    layout = torch.ones((1, 1, 2), dtype=torch.bool, device=DEVICE)
    x = torch.arange(
        SPARSITY_BLOCK_SIZE, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    y = torch.arange(
        SPARSITY_BLOCK_SIZE * 2, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    output = operation(x, y, layout, SPARSITY_BLOCK_SIZE)
    actual = bs.ops.to_dense(output, layout, SPARSITY_BLOCK_SIZE)
    expected = reference(x.unsqueeze(-1), y.unsqueeze(-2))

    assert actual.shape == (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_broadcast_add_supports_documented_dtypes(dtype: torch.dtype):
    layout = _full_layout()
    x = torch.ones((1, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE)
    y = torch.zeros((1, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE)

    output = bs.ops.misc.broadcast_add(x, y, layout, SPARSITY_BLOCK_SIZE)

    assert output.dtype == dtype
    assert torch.all(output == 1)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_broadcast_sub_supports_documented_dtypes(dtype: torch.dtype):
    layout = _full_layout()
    x = torch.ones((1, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE)
    y = torch.zeros((1, SPARSITY_BLOCK_SIZE), dtype=dtype, device=DEVICE)

    output = bs.ops.misc.broadcast_sub(x, y, layout, SPARSITY_BLOCK_SIZE)

    assert output.dtype == dtype
    assert torch.all(output == 1)


def test_broadcast_sub_rejects_boolean_operands_clearly():
    layout = _full_layout()
    x = torch.ones((1, SPARSITY_BLOCK_SIZE), dtype=torch.bool, device=DEVICE)
    y = torch.zeros((1, SPARSITY_BLOCK_SIZE), dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="does not support bool"):
        bs.ops.misc.broadcast_sub(x, y, layout, SPARSITY_BLOCK_SIZE)


def test_public_layout_builders_return_base_tensors():
    layout = _full_layout()
    dense = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    source = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    indices = bs.BlksprsTensor.wrap(torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        dtype=torch.int64,
        device=DEVICE,
    ))

    dense_layout = bs.layouting.build_sparsity_layout(
        bs.BlksprsTensor.wrap(dense), SPARSITY_BLOCK_SIZE)
    adapted_layout = bs.layouting.build_sparsity_layout_adaption(
        source, layout, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    distribution_layout = bs.layouting.build_distribution_layout(
        indices,
        layout,
        2,
        torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
        SPARSITY_BLOCK_SIZE,
    )
    _, operation_layout = bs.ops.adapt_layout(
        source, layout, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    matmul_layouts = [
        builder(bs.BlksprsTensor.wrap(layout), bs.BlksprsTensor.wrap(layout))
        for builder in (
            bs.layouting.build_sparsity_layout_matmul,
            bs.layouting.build_sparsity_layout_matmul_fast,
            bs.layouting.build_sparsity_layout_matmul_outer,
        )
    ]

    assert type(dense_layout) is torch.Tensor
    assert type(adapted_layout) is torch.Tensor
    assert type(distribution_layout) is torch.Tensor
    assert type(operation_layout) is torch.Tensor
    assert all(type(matmul_layout) is torch.Tensor for matmul_layout in matmul_layouts)


def test_public_layout_builders_convert_non_contiguous_data_operands():
    layout = _full_layout()
    dense_storage = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
    dense = dense_storage[:, :, ::2]
    index_storage = torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2),
        dtype=torch.int64,
        device=DEVICE,
    )
    indices = bs.BlksprsTensor.wrap(index_storage[:, :, ::2])
    sparse = bs.ops.to_sparse(
        torch.ones_like(dense), layout, SPARSITY_BLOCK_SIZE).transpose(1, 2)

    assert not dense.is_contiguous()
    assert not indices.is_contiguous()
    assert not sparse.is_contiguous()

    dense_layout = bs.layouting.build_sparsity_layout(
        dense, SPARSITY_BLOCK_SIZE)
    distribution_layout = bs.layouting.build_distribution_layout(
        indices,
        layout,
        2,
        torch.Size((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)),
        SPARSITY_BLOCK_SIZE,
    )
    adapted_layout = bs.layouting.build_sparsity_layout_adaption(
        sparse, layout, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)

    assert torch.equal(dense_layout, layout)
    assert torch.equal(distribution_layout, layout)
    assert torch.equal(adapted_layout, layout)


def test_layout_transformations_return_base_tensors_for_subclassed_layouts():
    layout = _full_layout(columns=2)
    subclassed_layout = bs.BlksprsTensor.wrap(layout)
    dense = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2), device=DEVICE)
    source = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)

    repeat_cache = {}
    _, repeated_layout = bs.ops.repeat(
        source, subclassed_layout, (1, 1, 1), SPARSITY_BLOCK_SIZE,
        layout_cache=repeat_cache,
    )
    _, interleaved_layout = bs.ops.repeat_interleave(
        source, subclassed_layout, 1, SPARSITY_BLOCK_SIZE)
    _, transposed_layout = bs.ops.transpose(
        source, subclassed_layout, SPARSITY_BLOCK_SIZE)
    split_source, split_layout = bs.ops.split(
        source, subclassed_layout, 2, 2, SPARSITY_BLOCK_SIZE)
    _, merged_layout = bs.ops.merge(
        split_source, bs.BlksprsTensor.wrap(split_layout), 2, 2,
        SPARSITY_BLOCK_SIZE,
    )
    _, sum_layout = bs.ops.misc.row_wise_sum(
        source, subclassed_layout, SPARSITY_BLOCK_SIZE)
    _, max_layout = bs.ops.misc.row_wise_max(
        source, subclassed_layout, SPARSITY_BLOCK_SIZE)

    output_layouts = (
        repeated_layout,
        interleaved_layout,
        transposed_layout,
        split_layout,
        merged_layout,
        sum_layout,
        max_layout,
    )
    assert all(type(output_layout) is torch.Tensor for output_layout in output_layouts)
    assert type(repeat_cache["sparsity_layout_o"]) is torch.Tensor
    assert type(repeat_cache["layout_indices"]) is torch.Tensor
    assert type(repeat_cache["packed_indices"]) is torch.Tensor


def test_adapt_layout_returns_a_base_tensor_for_a_subclassed_target_layout():
    layout = _full_layout()
    dense = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    source = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    target_layout = bs.BlksprsTensor.wrap(layout.clone())

    _, output_layout = bs.ops.adapt_layout(
        source,
        layout,
        SPARSITY_BLOCK_SIZE,
        SPARSITY_BLOCK_SIZE,
        sparsity_layout_to=target_layout,
    )

    assert type(output_layout) is torch.Tensor


def test_public_layout_caches_expose_base_tensor_metadata():
    layout = bs.BlksprsTensor.wrap(_full_layout())
    row_striped_cache = {}

    assert bs.ops.is_row_striped_layout(layout, layout_cache=row_striped_cache)
    flash_cache = bs.ops.flash_attention_build_layout_cache(
        layout,
        sparsity_layout_q=layout,
        sparsity_layout_k=layout,
        sparsity_layout_v=layout,
        sparsity_layout_o=layout,
        sparsity_layout_mask=layout,
        sparsity_layout_bias=layout,
    )

    assert type(row_striped_cache["active_row_mask"]) is torch.Tensor
    assert type(row_striped_cache["active_row_flat_indices"]) is torch.Tensor
    assert all(
        type(value) is torch.Tensor
        for name, value in flash_cache.items()
        if isinstance(value, torch.Tensor) and not name.startswith("_blksprs_")
    )


def test_default_softmax_falls_back_for_oversized_fused_rows_and_reuses_cache():
    n_blocks = 8193
    layout = _full_layout(columns=n_blocks)
    source = bs.BlksprsTensor.wrap(torch.zeros(
        (n_blocks, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    cache = {}

    actual = bs.ops.softmax(
        source, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)
    expected = bs.ops.softmax(
        source, layout, SPARSITY_BLOCK_SIZE, flag_fused=False)
    cached_tensors = {
        name: value
        for name, value in cache.items()
        if isinstance(value, torch.Tensor) and not name.startswith("_blksprs_")
    }

    repeated = bs.ops.softmax(
        source, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)

    assert torch.allclose(actual, expected)
    assert torch.equal(repeated, actual)
    assert all(cache[name] is value for name, value in cached_tensors.items())
    assert {
        "layout_indices",
        "max_blocks_line",
        "packed_indices_rws",
        "packed_indices_sorted",
    }.issubset(cache)
    assert all(type(value) is torch.Tensor for value in cached_tensors.values())

    with pytest.raises(ValueError, match="131072"):
        bs.ops.softmax_fused(
            source, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)


def test_to_blksprs_forwards_layout_cache():
    layout = _full_layout()
    dense = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    cache = {}

    output = bs.ops.to_blksprs(
        dense, layout, SPARSITY_BLOCK_SIZE, layout_cache=cache)

    assert type(output) is bs.BlksprsTensor
    assert cache["n_sparse_blocks"] == 1


def test_cuda_operations_reject_cpu_layouts_at_public_boundary():
    layout = torch.ones((1, 1, 1), dtype=torch.bool)
    dense = torch.randn((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    with pytest.raises(ValueError, match="same device"):
        bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)

    cuda_layout = layout.to(DEVICE)
    source = bs.BlksprsTensor.wrap(dense)
    indices = bs.BlksprsTensor.wrap(torch.zeros_like(dense, dtype=torch.int64))
    with pytest.raises(ValueError, match="same device"):
        bs.ops.scatter_reduce(
            source, cuda_layout, 2, indices, layout, SPARSITY_BLOCK_SIZE)


@pytest.mark.parametrize("shape", [(1, 16), (1, 1, 16, 16)])
def test_build_sparsity_layout_full_rejects_non_three_dimensional_tensors(shape):
    tensor = torch.zeros(shape, device=DEVICE)

    with pytest.raises(ValueError, match="3 dimensions"):
        bs.layouting.build_sparsity_layout_full(tensor, SPARSITY_BLOCK_SIZE)


def test_build_sparsity_layout_full_rejects_cpu_tensors():
    tensor = torch.zeros((1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE))

    with pytest.raises(ValueError, match="on GPU"):
        bs.layouting.build_sparsity_layout_full(tensor, SPARSITY_BLOCK_SIZE)


def test_is_row_striped_layout_validates_public_layout_contract():
    with pytest.raises(ValueError, match="on GPU"):
        bs.ops.is_row_striped_layout(torch.ones((1, 1, 1), dtype=torch.bool))

    with pytest.raises(ValueError, match="3 dimensions"):
        bs.ops.is_row_striped_layout(torch.ones((1, 1), dtype=torch.bool, device=DEVICE))

    with pytest.raises(ValueError, match="either 0 or 1"):
        bs.ops.is_row_striped_layout(torch.full((1, 1, 1), 2, device=DEVICE))


@pytest.mark.parametrize("shape", [(), (16,)])
def test_do_shape_blocksparse_rejects_tensors_with_fewer_than_two_dimensions(shape):
    tensor = torch.zeros(shape)

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        bs.utils.do_shape_blocksparse(tensor)


def test_shape_blocksparse_helpers_validate_and_round_trip_shapes():
    matrix = torch.zeros((3, 5))
    shaped, original_shape = bs.utils.do_shape_blocksparse(matrix)

    assert shaped.shape == (1, 3, 5)
    assert bs.utils.undo_shape_blocksparse(shaped, original_shape).shape == matrix.shape

    with pytest.raises(ValueError, match="exactly 3 dimensions"):
        bs.utils.undo_shape_blocksparse(torch.zeros((3, 5)), original_shape)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        bs.utils.undo_shape_blocksparse(shaped, (5,))
    with pytest.raises(ValueError, match="batch dimension"):
        bs.utils.undo_shape_blocksparse(torch.zeros((2, 3, 5)), original_shape)


def test_shape_blocksparse_helpers_support_empty_leading_dimensions():
    tensor = torch.zeros((2, 0, 3, 5))

    shaped, original_shape = bs.utils.do_shape_blocksparse(tensor)
    restored = bs.utils.undo_shape_blocksparse(shaped, original_shape)

    assert shaped.shape == (0, 3, 5)
    assert restored.shape == tensor.shape


@pytest.mark.parametrize("value", [True, 1.5])
def test_positive_integer_validation_rejects_non_integers(value):
    with pytest.raises(TypeError, match="must be an integer"):
        bs.utils.validation.validate_positive_integer(value, "value")

    with pytest.raises(TypeError, match="must contain integers"):
        bs.utils.validation.validate_positive_integer_tuple((1, value, 1), 3, "values")


def test_import_does_not_change_torch_dynamo_configuration():
    repository_root = Path(__file__).parents[2]
    code = """
import torch
before = torch._dynamo.config.capture_scalar_outputs
import blksprs
assert torch._dynamo.config.capture_scalar_outputs == before
"""
    subprocess.run([sys.executable, "-c", code], cwd=repository_root, check=True)


@pytest.mark.benchmark
def test_int64_distribution_indices_above_int32_range():
    torch.cuda.synchronize(DEVICE)
    torch.cuda.empty_cache()
    free_memory, total_memory = torch.cuda.mem_get_info(DEVICE)
    if total_memory < 20 * 1024 ** 3:
        pytest.skip("Requires at least 20 GiB of GPU memory")
    if free_memory < 20 * 1024 ** 3:
        pytest.skip("Requires at least 20 GiB of free GPU memory")

    index_value = 2_147_483_648
    n_column_blocks = index_value // SPARSITY_BLOCK_SIZE + 1
    small_layout = _full_layout()
    large_layout = torch.zeros((1, 1, n_column_blocks), dtype=torch.bool, device=DEVICE)
    large_layout[0, 0, index_value // SPARSITY_BLOCK_SIZE] = True
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    indices = bs.BlksprsTensor.wrap(torch.full(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), index_value,
        dtype=torch.int64, device=DEVICE))

    gathered = bs.ops.gather(
        source, large_layout, 2, indices, small_layout, SPARSITY_BLOCK_SIZE)
    assert torch.all(gathered == 1)

    scattered = bs.ops.scatter_reduce(
        source, small_layout, 2, indices, large_layout, SPARSITY_BLOCK_SIZE)
    expected_scattered = torch.zeros_like(scattered)
    expected_scattered[:, :, 0] = SPARSITY_BLOCK_SIZE
    assert torch.equal(scattered, expected_scattered)

    distribution_layout = bs.layouting.build_distribution_layout(
        indices,
        small_layout,
        2,
        torch.Size((1, SPARSITY_BLOCK_SIZE, n_column_blocks * SPARSITY_BLOCK_SIZE)),
        SPARSITY_BLOCK_SIZE,
    )
    assert int(distribution_layout.sum()) == 1
    assert distribution_layout[0, 0, -1]


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_layout_builders_and_adaption_preserve_nan_blocks(dtype: torch.dtype):
    dense = torch.zeros(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2),
        dtype=dtype,
        device=DEVICE,
    )
    dense[0, 0, SPARSITY_BLOCK_SIZE] = float("nan")
    expected_layout = torch.tensor([[[False, True]]], device=DEVICE)

    dense_layout = bs.layouting.build_sparsity_layout(
        dense, SPARSITY_BLOCK_SIZE)

    source_layout = _full_layout(columns=2)
    source = bs.ops.to_sparse(dense, source_layout, SPARSITY_BLOCK_SIZE)
    adapted_layout = bs.layouting.build_sparsity_layout_adaption(
        source,
        source_layout,
        SPARSITY_BLOCK_SIZE,
        SPARSITY_BLOCK_SIZE,
    )
    adapted, returned_layout = bs.ops.adapt_layout(
        source,
        source_layout,
        SPARSITY_BLOCK_SIZE,
        SPARSITY_BLOCK_SIZE,
    )

    assert torch.equal(dense_layout, expected_layout)
    assert torch.equal(adapted_layout, expected_layout)
    assert torch.equal(returned_layout, expected_layout)
    assert adapted.shape == (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE)
    assert torch.isnan(adapted[0, 0, 0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_row_wise_max_propagates_nan_across_active_blocks(dtype: torch.dtype):
    layout = _full_layout(columns=2)
    dense = torch.arange(
        SPARSITY_BLOCK_SIZE * SPARSITY_BLOCK_SIZE * 2,
        dtype=torch.float32,
        device=DEVICE,
    ).reshape(1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE * 2).to(dtype)
    dense[0, 0, 0] = float("nan")
    dense[0, 1, SPARSITY_BLOCK_SIZE] = float("nan")
    source = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)

    output, output_layout = bs.ops.misc.row_wise_max(
        source, layout, SPARSITY_BLOCK_SIZE)
    actual = bs.ops.to_dense(
        output, output_layout, SPARSITY_BLOCK_SIZE)[..., 0]
    expected = torch.max(dense, dim=-1).values

    assert output.dtype == dtype
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(
        torch.nan_to_num(actual, nan=0.0),
        torch.nan_to_num(expected, nan=0.0),
    )


@pytest.mark.parametrize("non_finite_input", ["q", "k", "bias"])
def test_flash_attention_propagates_non_finite_inputs(non_finite_input: str):
    q = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bias = torch.zeros_like(q) if non_finite_input == "bias" else None

    if non_finite_input == "q":
        q[0, 0, 0] = float("nan")
    elif non_finite_input == "k":
        k[0, 0, 0] = float("nan")
    else:
        assert bias is not None
        bias[0, 0, 0] = float("nan")

    actual = _simple_flash_attention(q, k, v, attention_bias=bias)
    scores = torch.bmm(q, k.transpose(-2, -1)) / SPARSITY_BLOCK_SIZE ** 0.5
    if bias is not None:
        scores = scores + bias
    expected = torch.bmm(torch.softmax(scores, dim=-1), v)

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_flash_attention_propagates_non_finite_scale(scale: float):
    q = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)
    k = torch.ones_like(q)
    v = torch.randn_like(q)

    actual = _simple_flash_attention(q, k, v, scale=scale)
    scores = torch.bmm(q, k.transpose(-2, -1)) * scale
    expected = torch.bmm(torch.softmax(scores, dim=-1), v)

    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.any(torch.isnan(actual))


def test_flash_attention_backward_propagates_nan_bias():
    layout = _full_layout()
    q = bs.BlksprsTensor.wrap(torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    ))
    k = bs.BlksprsTensor.wrap(torch.randn_like(q, requires_grad=True))
    v = bs.BlksprsTensor.wrap(torch.randn_like(q, requires_grad=True))
    bias = bs.BlksprsTensor.wrap(torch.zeros_like(q, requires_grad=True))
    with torch.no_grad():
        bias[0, 0, 0] = float("nan")

    output = bs.ops.flash_attention(
        q, layout,
        k, layout,
        v, layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        attention_bias=bias,
        sparsity_layout_bias=layout,
        sparsity_layout_o=layout,
    )
    gradients = torch.autograd.grad(output.sum(), (q, k, v, bias))

    assert all(torch.any(torch.isnan(gradient)) for gradient in gradients)


def test_flash_attention_mask_overrides_non_finite_bias():
    layout = _full_layout()
    q = torch.randn(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bias = torch.zeros_like(q, requires_grad=True)
    mask = torch.zeros_like(q, dtype=torch.bool)
    mask[..., 0] = True
    with torch.no_grad():
        bias[..., 0] = float("inf")

    output_sparse = bs.ops.flash_attention(
        bs.ops.to_sparse(q, layout, SPARSITY_BLOCK_SIZE), layout,
        bs.ops.to_sparse(k, layout, SPARSITY_BLOCK_SIZE), layout,
        bs.ops.to_sparse(v, layout, SPARSITY_BLOCK_SIZE), layout,
        layout,
        SPARSITY_BLOCK_SIZE,
        attention_mask=bs.ops.to_sparse(
            mask, layout, SPARSITY_BLOCK_SIZE),
        sparsity_layout_mask=layout,
        attention_bias=bs.ops.to_sparse(
            bias, layout, SPARSITY_BLOCK_SIZE),
        sparsity_layout_bias=layout,
        sparsity_layout_o=layout,
    )
    actual = bs.ops.to_dense(
        output_sparse, layout, SPARSITY_BLOCK_SIZE)

    scores = torch.bmm(q, k.transpose(-2, -1)) / SPARSITY_BLOCK_SIZE ** 0.5
    probabilities = torch.softmax(
        (scores + bias).masked_fill(mask, float("-inf")), dim=-1)
    expected = torch.bmm(probabilities, v)

    assert torch.all(torch.isfinite(actual))
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)

    actual.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))
    assert k.grad is not None and torch.all(torch.isfinite(k.grad))
    assert v.grad is not None and torch.all(torch.isfinite(v.grad))
    assert bias.grad is not None and torch.all(torch.isfinite(bias.grad))
    assert torch.count_nonzero(bias.grad.masked_select(mask)) == 0


@pytest.mark.parametrize("layout_shape", [(1, 1, 1), (1, 1, 0), (0, 1, 1)])
def test_empty_row_striped_conversions_preserve_autograd(
        layout_shape: tuple[int, int, int]):
    layout = torch.zeros(layout_shape, dtype=torch.bool, device=DEVICE)
    dense = torch.randn(
        (
            layout_shape[0],
            layout_shape[1] * SPARSITY_BLOCK_SIZE,
            layout_shape[2] * SPARSITY_BLOCK_SIZE,
        ),
        device=DEVICE,
        requires_grad=True,
    )

    sparse = bs.ops.to_sparse_row_striped(
        dense, layout, SPARSITY_BLOCK_SIZE)

    assert sparse.requires_grad
    sparse.sum().backward()
    assert dense.grad is not None
    assert torch.equal(dense.grad, torch.zeros_like(dense))

    sparse_base = torch.empty(
        (0, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE),
        device=DEVICE,
        requires_grad=True,
    )
    round_trip = bs.ops.to_dense_row_striped(
        bs.BlksprsTensor.wrap(sparse_base),
        layout,
        SPARSITY_BLOCK_SIZE,
    )

    assert round_trip.requires_grad
    round_trip.sum().backward()
    assert sparse_base.grad is not None
    assert torch.equal(sparse_base.grad, torch.zeros_like(sparse_base))


@pytest.mark.parametrize("scale", [torch.tensor(1.0), "1.0", 1.0 + 0.0j])
def test_flash_attention_rejects_non_real_scale(scale):
    values = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    with pytest.raises(TypeError, match="scale must be a real number"):
        _simple_flash_attention(values, values, values, scale=scale)


def test_flash_attention_accepts_numpy_real_scale():
    values = torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE)

    output = _simple_flash_attention(
        values, values, values, scale=np.float32(0.25))

    assert torch.allclose(output, values)


def test_row_wise_processing_validates_input_before_packing_layout():
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))
    malformed_layout = torch.ones(
        (1, 1), dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="exactly 3 dimensions"):
        bs.utils.apply_function_applicable_row_wise(
            source, malformed_layout, SPARSITY_BLOCK_SIZE, lambda tensor: tensor)


def test_row_wise_processing_rejects_non_callable():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises(TypeError, match="must be callable"):
        bs.utils.apply_function_applicable_row_wise(
            source, layout, SPARSITY_BLOCK_SIZE, None)


@pytest.mark.parametrize(
    ("function", "exception", "message"),
    [
        (lambda tensor: None, TypeError, "must return a Tensor"),
        (lambda tensor: tensor[..., :-1], ValueError, "output shape"),
        (lambda tensor: tensor.cpu(), ValueError, "on GPU"),
        (lambda tensor: tensor.to(torch.complex64), ValueError, "dtype is not supported"),
    ],
)
def test_row_wise_processing_validates_callable_output(function, exception, message: str):
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    with pytest.raises(exception, match=message):
        bs.utils.apply_function_applicable_row_wise(
            source, layout, SPARSITY_BLOCK_SIZE, function)


def test_row_wise_processing_allows_supported_dtype_changes():
    layout = _full_layout()
    source = bs.BlksprsTensor.wrap(torch.ones(
        (1, SPARSITY_BLOCK_SIZE, SPARSITY_BLOCK_SIZE), device=DEVICE))

    output = bs.utils.apply_function_applicable_row_wise(
        source,
        layout,
        SPARSITY_BLOCK_SIZE,
        lambda tensor: tensor.to(torch.int32),
    )

    assert output.dtype == torch.int32
    assert torch.all(output == 1)


def test_benchmark_rejects_matrix_and_block_size_length_mismatch():
    with pytest.raises(ValueError, match="Matrix sizes and sparsity block sizes"):
        benchmark(["method"], lambda *_: {}, [16, 32], [16], lambda: None)


def test_benchmark_rejects_label_and_function_length_mismatch():
    with pytest.raises(ValueError, match="Method labels and benchmark functions"):
        benchmark(["method", "extra"], lambda *_: {}, [16], [16], lambda: None)
