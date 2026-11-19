import importlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")

import pytest
import torch

import blksprs as bs

matmul_module = importlib.import_module("blksprs.ops.matmul")

DEVICE = torch.device("cuda:0")
# CUDA limits the second and third grid axes to 65,535 programs.
MAX_SECONDARY_GRID_PROGRAMS = 65_535


def _compress(dense: torch.Tensor, layout: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    """Selects the active blocks of a dense tensor in the row-major order of the compressed form."""
    b, m, n = dense.shape
    blocks = dense.reshape(b, m // sparsity_block_size, sparsity_block_size,
                           n // sparsity_block_size, sparsity_block_size).transpose(2, 3)
    return blocks[layout]


def _reference_softmax(x: torch.Tensor, layout: torch.Tensor, sparsity_block_size: int) -> torch.Tensor:
    """Computes the softmax over the active blocks of each row of a compressed tensor with PyTorch."""
    b, r, c = layout.shape
    indices = layout.nonzero()
    rows = indices[:, 0] * r + indices[:, 1]
    lines = torch.full((b * r, sparsity_block_size, c, sparsity_block_size), float("-inf"),
                       dtype=x.dtype, device=x.device)
    lines[rows, :, indices[:, 2], :] = x
    lines = torch.softmax(lines.reshape(b * r, sparsity_block_size, -1), dim=-1)
    return lines.reshape(b * r, sparsity_block_size, c, sparsity_block_size)[rows, :, indices[:, 2], :]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).abs().max() / expected.abs().max()).item()


_SOFTMAX_OPERATIONS = {
    "default": lambda x, layout, sbs: bs.ops.softmax(x, layout, sbs),
    "regular": lambda x, layout, sbs: bs.ops.softmax(x, layout, sbs, flag_fused=False),
    "fused": lambda x, layout, sbs: bs.ops.softmax_fused(x, layout, sbs),
}


@pytest.mark.parametrize("operation", ["default", "fused"])
def test_fused_softmax_supports_more_block_rows_than_secondary_grid_axes(operation: str):
    sparsity_block_size = 16
    n_rows = MAX_SECONDARY_GRID_PROGRAMS + 65
    layout = torch.ones((1, n_rows, 1), dtype=torch.bool, device=DEVICE)
    values = torch.randn((n_rows, sparsity_block_size, sparsity_block_size), device=DEVICE)
    gradient = torch.randn_like(values)

    x = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output = _SOFTMAX_OPERATIONS[operation](x, layout, sparsity_block_size)
    actual_gradient = torch.autograd.grad(output, x, gradient)[0]

    x_reference = values.clone().requires_grad_(True)
    expected = torch.softmax(x_reference, dim=-1)
    expected_gradient = torch.autograd.grad(expected, x_reference, gradient)[0]

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


def test_fused_softmax_supports_many_block_rows_in_sparse_batched_layouts():
    torch.manual_seed(0)
    sparsity_block_size = 16
    layout = torch.rand((2, MAX_SECONDARY_GRID_PROGRAMS + 65, 2), device=DEVICE) < 0.5
    values = torch.randn((int(layout.sum()), sparsity_block_size, sparsity_block_size), device=DEVICE)
    gradient = torch.randn_like(values)

    x = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    output = bs.ops.softmax_fused(x, layout, sparsity_block_size)
    actual_gradient = torch.autograd.grad(output, x, gradient)[0]

    x_reference = values.clone().requires_grad_(True)
    expected = _reference_softmax(x_reference, layout, sparsity_block_size)
    expected_gradient = torch.autograd.grad(expected, x_reference, gradient)[0]

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("operation", ["default", "regular", "fused"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_softmax_computes_half_precision_inputs_in_float32_under_autocast(operation: str, dtype: torch.dtype):
    torch.manual_seed(0)
    sparsity_block_size = 16
    layout = torch.rand((2, 4, 4), device=DEVICE) < 0.6
    layout[0, 1] = False
    values = torch.randn((int(layout.sum()), sparsity_block_size, sparsity_block_size), dtype=dtype, device=DEVICE)
    gradient = torch.randn(values.shape, device=DEVICE)

    x = bs.BlksprsTensor.wrap(values.clone().requires_grad_(True))
    with torch.autocast(device_type="cuda", dtype=dtype):
        output = _SOFTMAX_OPERATIONS[operation](x, layout, sparsity_block_size)
        stock_dtype = torch.softmax(values, dim=-1).dtype
    actual_gradient = torch.autograd.grad(output, x, gradient)[0]

    x_reference = values.clone().requires_grad_(True)
    expected = _reference_softmax(x_reference.float(), layout, sparsity_block_size)
    expected_gradient = torch.autograd.grad(expected, x_reference, gradient)[0]

    assert stock_dtype == torch.float32
    assert output.dtype == torch.float32
    assert actual_gradient.dtype == dtype
    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_gradient.float(), expected_gradient.float(), atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("operation", ["default", "regular", "fused"])
def test_softmax_rejects_half_precision_inputs_outside_autocast(operation: str):
    sparsity_block_size = 16
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    x = torch.randn((1, sparsity_block_size, sparsity_block_size), dtype=torch.float16, device=DEVICE)

    with pytest.raises(ValueError, match="float32"):
        _SOFTMAX_OPERATIONS[operation](bs.BlksprsTensor.wrap(x), layout, sparsity_block_size)


@pytest.mark.parametrize("fill", ["random", "true"])
def test_broadcast_add_of_booleans_is_logical_or(fill: str):
    torch.manual_seed(0)
    sparsity_block_size = 16
    layout = torch.tensor([[[True, False, True], [False, True, True]]], device=DEVICE)
    if fill == "random":
        x = torch.rand((1, 2 * sparsity_block_size), device=DEVICE) < 0.5
        y = torch.rand((1, 3 * sparsity_block_size), device=DEVICE) < 0.5
        x[0, 0] = y[0, 0] = True
    else:
        x = torch.ones((1, 2 * sparsity_block_size), dtype=torch.bool, device=DEVICE)
        y = torch.ones((1, 3 * sparsity_block_size), dtype=torch.bool, device=DEVICE)

    output = bs.ops.misc.broadcast_add(x, y, layout, sparsity_block_size)
    expected = _compress(x[:, :, None] + y[:, None, :], layout, sparsity_block_size)

    assert expected.dtype == torch.bool
    assert output.dtype == torch.bool
    assert torch.equal(output, expected)


@pytest.mark.parametrize(("x_dtype", "y_dtype"), [
    (torch.bool, torch.bool),
    (torch.bool, torch.int32),
    (torch.int32, torch.bool),
])
def test_broadcast_sub_rejects_booleans_like_torch(x_dtype: torch.dtype, y_dtype: torch.dtype):
    sparsity_block_size = 16
    layout = torch.ones((1, 1, 1), dtype=torch.bool, device=DEVICE)
    x = torch.ones((1, sparsity_block_size), dtype=x_dtype, device=DEVICE)
    y = torch.zeros((1, sparsity_block_size), dtype=y_dtype, device=DEVICE)

    with pytest.raises(RuntimeError, match="bool"):
        torch.sub(x[:, :, None], y[:, None, :])
    with pytest.raises(ValueError, match="does not support bool"):
        bs.ops.misc.broadcast_sub(x, y, layout, sparsity_block_size)


@pytest.fixture
def restore_tf32_flag():
    previous = torch.backends.cuda.matmul.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _float32_matmul_errors() -> tuple[float, float, float, float]:
    """Returns the relative errors of the blksprs product, its gradients, and the torch product against float64."""
    torch.manual_seed(0)
    sparsity_block_size = 64
    b, m, k, n = 2, 128, 512, 128
    layout_x = torch.ones((b, m // sparsity_block_size, k // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    layout_y = torch.ones((b, k // sparsity_block_size, n // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    layout_o = torch.ones((b, m // sparsity_block_size, n // sparsity_block_size), dtype=torch.bool, device=DEVICE)
    x_dense = torch.randn((b, m, k), device=DEVICE)
    y_dense = torch.randn((b, k, n), device=DEVICE)
    gradient_dense = torch.randn((b, m, n), device=DEVICE)

    x = bs.BlksprsTensor.wrap(_compress(x_dense, layout_x, sparsity_block_size).requires_grad_(True))
    y = bs.BlksprsTensor.wrap(_compress(y_dense, layout_y, sparsity_block_size).requires_grad_(True))
    output = bs.ops.matmul(x, layout_x, y, layout_y, layout_o, sparsity_block_size)
    gradient_x, gradient_y = torch.autograd.grad(
        output, (x, y), _compress(gradient_dense, layout_o, sparsity_block_size))

    x_double, y_double, gradient_double = x_dense.double(), y_dense.double(), gradient_dense.double()
    expected = _compress(x_double @ y_double, layout_o, sparsity_block_size)
    expected_gradient_x = _compress(gradient_double @ y_double.transpose(-1, -2), layout_x, sparsity_block_size)
    expected_gradient_y = _compress(x_double.transpose(-1, -2) @ gradient_double, layout_y, sparsity_block_size)

    return (_relative_error(output, expected),
            _relative_error(gradient_x, expected_gradient_x),
            _relative_error(gradient_y, expected_gradient_y),
            _relative_error(_compress(x_dense @ y_dense, layout_o, sparsity_block_size), expected))


def test_float32_matmul_uses_full_precision_by_default(restore_tf32_flag):
    torch.backends.cuda.matmul.allow_tf32 = False

    *errors, stock_error = _float32_matmul_errors()

    assert stock_error < 1e-5
    assert all(error < 1e-5 for error in errors)


def test_float32_matmul_uses_tf32_when_pytorch_allows_it(restore_tf32_flag):
    torch.backends.cuda.matmul.allow_tf32 = True

    *errors, stock_error = _float32_matmul_errors()

    assert all(error < 5e-3 for error in errors)
    if torch.cuda.get_device_capability(DEVICE) >= (8, 0):
        # Both products round their operands to the 10-bit mantissa of TF32.
        assert stock_error > 1e-5
        assert all(error > 1e-5 for error in errors)


@pytest.mark.parametrize("allow_tf32", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_matmul_precision_is_independent_of_tf32_flag(restore_tf32_flag, allow_tf32: bool,
                                                                     dtype: torch.dtype):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    assert matmul_module._dot_input_precision(torch.empty(0, dtype=dtype, device=DEVICE)) == "tf32"


@pytest.mark.skipif(not hasattr(torch.backends, "fp32_precision"), reason="PyTorch lacks the precision API")
def test_float32_matmul_precision_follows_pytorch_precision_api():
    # The precision API makes later reads of the legacy flag fail, so it is exercised in a separate process.
    repository_root = Path(__file__).parents[2]
    code = """
import importlib
import torch
matmul = importlib.import_module("blksprs.ops.matmul")
float32 = torch.empty(0)
assert matmul._dot_input_precision(float32) == "ieee"
torch.backends.fp32_precision = "tf32"
assert matmul._dot_input_precision(float32) == "tf32"
torch.backends.cuda.matmul.fp32_precision = "ieee"
assert matmul._dot_input_precision(float32) == "ieee"
torch.backends.cuda.matmul.fp32_precision = "tf32"
assert matmul._dot_input_precision(float32) == "tf32"
"""
    subprocess.run([sys.executable, "-c", code], cwd=repository_root, check=True)
