import copy
import os
import threading

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")

import pytest
import torch

import blksprs as bs
from blksprs.utils import validation

DEVICE = torch.device("cuda:0")
SPARSITY_BLOCK_SIZE = 16
IN_FEATURES = 3 * SPARSITY_BLOCK_SIZE
OUT_FEATURES = 2 * SPARSITY_BLOCK_SIZE

ATOL = 2e-2
RTOL = 1.5e-2


def _sparse_layout() -> torch.Tensor:
    # Two batches of three row blocks over three feature blocks, with an empty and two partially active row blocks.
    layout = torch.ones((2, 3, IN_FEATURES // SPARSITY_BLOCK_SIZE), dtype=torch.bool, device=DEVICE)
    layout[0, 1] = False
    layout[1, 0, 1] = False
    layout[1, 2, 0] = False
    return layout


def _inputs(layout: torch.Tensor, requires_grad: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns a compressed input and its dense equivalent, which is zero outside the layout, as separate leaves.
    dense = torch.randn(
        (layout.size(0), layout.size(1) * SPARSITY_BLOCK_SIZE, layout.size(2) * SPARSITY_BLOCK_SIZE), device=DEVICE)
    sparse = bs.ops.to_sparse(dense, layout, SPARSITY_BLOCK_SIZE)
    dense = bs.ops.to_dense(sparse, layout, SPARSITY_BLOCK_SIZE)
    return sparse.detach().requires_grad_(requires_grad), dense.detach().requires_grad_(requires_grad)


def _linear(use_bias: bool = True) -> torch.nn.Linear:
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=use_bias, device=DEVICE)


def _apply_cached(sparse: torch.Tensor, layout: torch.Tensor, linear: torch.nn.Linear, cache: dict) -> torch.Tensor:
    output, output_layout = bs.utils.apply_torch_linear_cached(
        sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
    return bs.ops.to_dense(output, output_layout, SPARSITY_BLOCK_SIZE)


def _apply_uncached(sparse: torch.Tensor, layout: torch.Tensor, linear: torch.nn.Linear) -> torch.Tensor:
    output, output_layout = bs.utils.apply_torch_linear(sparse, layout, SPARSITY_BLOCK_SIZE, linear)
    return bs.ops.to_dense(output, output_layout, SPARSITY_BLOCK_SIZE)


def _assert_parameter_gradients_close(linear: torch.nn.Linear, reference: torch.nn.Linear) -> None:
    torch.testing.assert_close(linear.weight.grad, reference.weight.grad, atol=ATOL, rtol=RTOL)
    if reference.bias is not None:
        torch.testing.assert_close(linear.bias.grad, reference.bias.grad, atol=ATOL, rtol=RTOL)


def _cached_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _cached_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _cached_tensors(item)


# Cached linear projection

@pytest.mark.parametrize("use_bias", [False, True])
def test_cached_linear_accumulates_gradients_over_micro_batches(use_bias: bool):
    layout = _sparse_layout()
    linear = _linear(use_bias)
    reference = copy.deepcopy(linear)
    cache = {}

    for _ in range(3):
        sparse, dense = _inputs(layout)
        actual = _apply_cached(sparse, layout, linear, cache)
        expected = reference(dense)
        gradient = torch.randn_like(expected)
        actual.backward(gradient)
        expected.backward(gradient)

        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            sparse.grad, bs.ops.to_sparse(dense.grad, layout, SPARSITY_BLOCK_SIZE), atol=ATOL, rtol=RTOL)

    _assert_parameter_gradients_close(linear, reference)


@pytest.mark.parametrize("use_bias", [False, True])
def test_cached_linear_supports_non_reentrant_checkpoint(use_bias: bool):
    layout = _sparse_layout()
    linear = _linear(use_bias)
    reference = copy.deepcopy(linear)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    cache = {}

    def block(sparse: torch.Tensor) -> torch.Tensor:
        return _apply_cached(sparse, layout, linear, cache)

    for _ in range(3):
        sparse, dense = _inputs(layout)
        actual = torch.utils.checkpoint.checkpoint(block, sparse, use_reentrant=False)
        expected = reference(dense)
        gradient = torch.randn_like(expected)
        actual.backward(gradient)
        expected.backward(gradient)

        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(
            sparse.grad, bs.ops.to_sparse(dense.grad, layout, SPARSITY_BLOCK_SIZE), atol=ATOL, rtol=RTOL)
        _assert_parameter_gradients_close(linear, reference)

        optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.step()
        reference_optimizer.zero_grad()


@pytest.mark.parametrize(("optimizer_class", "lr"), [
    (torch.optim.AdamW, 0.1),
    (torch.optim.Adam, 0.1),
    (torch.optim.SGD, 0.01),
])
def test_cached_linear_uses_current_weights_after_fused_optimizer_steps(optimizer_class, lr: float):
    layout = _sparse_layout()
    linear = _linear()
    # Fused optimizers update the parameters without incrementing their version counters.
    optimizer = optimizer_class(linear.parameters(), lr=lr, fused=True)
    cache = {}
    probe, probe_dense = _inputs(layout, requires_grad=False)
    previous = None

    for _ in range(3):
        sparse, dense = _inputs(layout)
        actual = _apply_cached(sparse, layout, linear, cache)
        torch.testing.assert_close(actual, linear(dense), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual, _apply_uncached(sparse, layout, linear))

        actual.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        current = _apply_cached(probe, layout, linear, cache)
        torch.testing.assert_close(current, _apply_uncached(probe, layout, linear))
        torch.testing.assert_close(current, linear(probe_dense), atol=ATOL, rtol=RTOL)
        if previous is not None:
            assert not torch.allclose(current, previous, atol=ATOL, rtol=RTOL)
        previous = current.detach()


@pytest.mark.parametrize("use_bias", [False, True])
def test_cached_linear_without_gradients_uses_current_weights(use_bias: bool):
    layout = _sparse_layout()
    linear = _linear(use_bias)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.1, fused=True)
    cache = {}
    probe, probe_dense = _inputs(layout, requires_grad=False)

    def evaluate() -> torch.Tensor:
        with torch.no_grad():
            actual = _apply_cached(probe, layout, linear, cache)
            torch.testing.assert_close(actual, linear(probe_dense), atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(actual, _apply_uncached(probe, layout, linear))
        return actual

    initial = evaluate()

    sparse, _ = _inputs(layout)
    _apply_cached(sparse, layout, linear, cache).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    stepped = evaluate()
    assert not torch.allclose(stepped, initial, atol=ATOL, rtol=RTOL)

    # In-place updates through ``.data`` do not increment the version counters of the parameters either.
    with torch.no_grad():
        linear.weight.data.mul_(-1.0)
        if use_bias:
            linear.bias.data.add_(1.0)
    updated = evaluate()
    assert not torch.allclose(updated, stepped, atol=ATOL, rtol=RTOL)


def test_cached_linear_cache_holds_only_non_differentiable_metadata():
    layout = _sparse_layout()
    linear = _linear()
    cache = {}

    sparse, _ = _inputs(layout)
    _apply_cached(sparse, layout, linear, cache).sum().backward()

    tensors = list(_cached_tensors(cache))
    assert tensors
    for tensor in tensors:
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
        assert not tensor.is_floating_point()


@pytest.mark.parametrize("use_bias", [False, True])
def test_cached_linear_warm_calls_do_not_synchronise(use_bias: bool):
    layout = _sparse_layout()
    linear = _linear(use_bias)
    # A regular optimizer step increments the parameter versions, which must not invalidate the layout metadata.
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    cache = {}
    sparse, _ = _inputs(layout)

    bs.utils.disable_validation()
    try:
        output, _ = bs.utils.apply_torch_linear_cached(
            sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
        output.square().mean().backward()
        optimizer.step()
        torch.cuda.synchronize()

        torch.cuda.set_sync_debug_mode("error")
        try:
            bs.utils.apply_torch_linear_cached(sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
            with torch.no_grad():
                bs.utils.apply_torch_linear_cached(sparse, layout, SPARSITY_BLOCK_SIZE, linear, layout_cache=cache)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    finally:
        bs.utils.enable_validation()


# Validation switches

def _record_switch(monkeypatch, name: str) -> list[tuple[int, bool, bool]]:
    # Records the thread, autograd phase, and result of every query of a switch.
    calls = []
    query = getattr(validation, name)

    def recording_query():
        result = query()
        calls.append((threading.get_ident(), torch._C._current_graph_task_id() != -1, result))
        return result

    monkeypatch.setattr(validation, name, recording_query)
    return calls


def _train_step() -> None:
    layout = _sparse_layout()
    sparse, _ = _inputs(layout)
    _apply_cached(sparse, layout, _linear(), {}).square().mean().backward()


@pytest.mark.parametrize("disabled", [False, True])
def test_validation_switch_reaches_autograd_backward_threads(monkeypatch, disabled: bool):
    calls = _record_switch(monkeypatch, "_check_skip_validation")

    if disabled:
        bs.utils.disable_validation()
    try:
        _train_step()
    finally:
        bs.utils.enable_validation()

    main_thread = threading.get_ident()
    forward_results = {skip for thread, in_backward, skip in calls if not in_backward}
    backward_calls = [(thread, skip) for thread, in_backward, skip in calls if in_backward]

    assert forward_results == {disabled}
    assert any(thread != main_thread for thread, _ in backward_calls)
    assert {skip for _, skip in backward_calls} == {disabled}


@pytest.mark.parametrize("disabled", [False, True])
def test_contiguous_switch_reaches_autograd_backward_threads(monkeypatch, disabled: bool):
    calls = _record_switch(monkeypatch, "_check_contiguous")

    if disabled:
        bs.utils.disable_contiguous()
    try:
        _train_step()
    finally:
        bs.utils.enable_contiguous()

    main_thread = threading.get_ident()
    backward_calls = [(thread, contiguous) for thread, in_backward, contiguous in calls if in_backward]

    assert any(thread != main_thread for thread, _ in backward_calls)
    assert {contiguous for _, contiguous in backward_calls} == {not disabled}


def test_validation_switches_are_process_global():
    observed = {}

    def observe(key: str) -> None:
        observed[key] = (validation._check_skip_validation(), validation._check_contiguous())

    def observe_in_thread(key: str) -> None:
        thread = threading.Thread(target=observe, args=(key,))
        thread.start()
        thread.join()

    bs.utils.disable_validation()
    bs.utils.disable_contiguous()
    try:
        observe_in_thread("disabled")
    finally:
        bs.utils.enable_contiguous()
        bs.utils.enable_validation()
    observe_in_thread("enabled")

    assert observed == {"disabled": (True, False), "enabled": (False, True)}
