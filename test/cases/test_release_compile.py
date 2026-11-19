import importlib
import inspect
import os

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")

import pytest
import torch
import triton
from torch._dynamo.utils import counters

import blksprs as bs
from blksprs.ops import distribution, flow
from blksprs.ops.misc import row_wise
from blksprs.utils.public_op import on_operand_device
from blksprs.utils.tools import as_base_tensor, build_layout_indices, build_packed_indices

DEVICE = torch.device("cuda:0")
# The sparsity block sizes and layout shapes of the successive calls of a compiled function. Every call changes the
# shapes, so that it recompiles, and the last call also changes the sparsity block size and thereby the kernel
# configurations.
VARIANTS = [(16, (2, 3, 4)), (16, (1, 4, 2)), (32, (3, 2, 3))]
RTOL = 1e-4
ATOL = 1e-4

# Flash Attention calls share one static configuration, so that the compiled kernel variants are reused.
FLASH_SPARSITY_BLOCK_SIZE = 16
FLASH_HEAD_DIMENSION = 32
FLASH_BATCHES = 2
FLASH_RELATION_BOUNDS = ((0, 64),)


@pytest.fixture(autouse=True)
def _fresh_compilation():
    """Starts every test without compiled code and compiles every graph anew instead of loading it from a cache.

    Inductor benchmarks the autotuned kernels of a newly compiled graph on its first call, on the live buffers of the
    call. Cached graphs and autotuning results would skip this benchmark and hide kernels that are not safe to
    benchmark repeatedly.

    """
    torch._dynamo.reset()
    counters.clear()
    with (torch._inductor.config.patch(fx_graph_cache=False, fx_graph_remote_cache=False, autotune_local_cache=False,
                                       autotune_remote_cache=False),
          torch._functorch.config.patch(enable_autograd_cache=False)):
        yield
    torch._dynamo.reset()


# Inputs

def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _layout(shape: tuple[int, int, int], generator: torch.Generator, density: float = 0.6) -> torch.Tensor:
    """Returns a random layout whose first and last blocks are active."""
    layout = torch.rand(shape, generator=generator, device=DEVICE) < density
    layout.view(-1)[0] = True
    layout.view(-1)[-1] = True
    return layout


def _dense(layout: torch.Tensor, sparsity_block_size: int, generator: torch.Generator,
           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    b, r, c = layout.shape
    return torch.randn((b, r * sparsity_block_size, c * sparsity_block_size), generator=generator, device=DEVICE,
                       dtype=dtype)


def _compressed(layout: torch.Tensor, sparsity_block_size: int, generator: torch.Generator,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return bs.ops.to_sparse(_dense(layout, sparsity_block_size, generator, dtype), layout,
                            sparsity_block_size).detach()


def _indices(layout: torch.Tensor, sparsity_block_size: int, generator: torch.Generator,
             unique: bool = False) -> torch.Tensor:
    """Returns compressed column indices for gather and scatter along the last dimension.

    Unique indices permute the columns of every row, so that no two values are scattered to the same position.

    """
    b, r, c = layout.shape
    shape = (b, r * sparsity_block_size, c * sparsity_block_size)
    if unique:
        dense = torch.rand(shape, generator=generator, device=DEVICE).argsort(dim=-1)
    else:
        dense = torch.randint(0, c * sparsity_block_size, shape, generator=generator, device=DEVICE)
    return bs.ops.to_sparse(dense, layout, sparsity_block_size)


def _cotangent(output: torch.Tensor) -> torch.Tensor:
    """Returns a fixed cotangent that depends only on the shape of an output."""
    generator = _generator(output.numel())
    return torch.randn(output.shape, generator=generator, device=DEVICE).to(output.dtype)


# Evaluation

def _flatten(value) -> list:
    if isinstance(value, (tuple, list)):
        return [item for element in value for item in _flatten(element)]
    return [value]


def _evaluate(function, arguments: tuple, differentiable: tuple[int, ...] = (),
              parameters: tuple[torch.Tensor, ...] = ()) -> tuple[list, list]:
    """Runs ``function`` and its backward with fixed cotangents and returns the values and the gradients."""
    inputs = [argument.detach().clone().requires_grad_(True) if index in differentiable else argument
              for index, argument in enumerate(arguments)]
    for parameter in parameters:
        parameter.grad = None

    values = _flatten(function(*inputs))

    outputs = [value for value in values if isinstance(value, torch.Tensor) and value.requires_grad]
    if outputs:
        torch.autograd.backward(outputs, [_cotangent(output) for output in outputs])

    gradients = [inputs[index].grad for index in differentiable] + [parameter.grad for parameter in parameters]
    return [value.detach() if isinstance(value, torch.Tensor) else value for value in values], gradients


def _assert_matches(actual: list, expected: list, exact: bool) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        if not isinstance(expected_value, torch.Tensor):
            assert actual_value == expected_value
        elif exact:
            assert actual_value.dtype == expected_value.dtype
            assert torch.equal(actual_value, expected_value)
        else:
            torch.testing.assert_close(actual_value, expected_value, rtol=RTOL, atol=ATOL, check_dtype=True)


def _assert_compiled_matches_eager(function, make_arguments, differentiable: tuple[int, ...] = (),
                                   exact_values: bool = True, exact_gradients: bool = True) -> None:
    """Compiles ``function`` and compares every call of a new variant, which recompiles, with eager execution.

    ``make_arguments`` returns the arguments and the parameters of a variant, given its sparsity block size, layout
    shape, and a generator.

    """
    compiled = torch.compile(function)

    for seed, (sparsity_block_size, shape) in enumerate(VARIANTS):
        arguments, parameters = make_arguments(sparsity_block_size, shape, _generator(seed))

        expected, expected_gradients = _evaluate(function, arguments, differentiable, parameters)
        actual, actual_gradients = _evaluate(compiled, arguments, differentiable, parameters)

        _assert_matches(actual, expected, exact_values)
        _assert_matches(actual_gradients, expected_gradients, exact_gradients)

    assert counters["stats"]["unique_graphs"] > 0


# Compiled public operations

def _conversions(dense, layout, sparsity_block_size):
    x = bs.ops.to_sparse(dense, layout, sparsity_block_size)
    y = bs.ops.to_blksprs(dense, layout, sparsity_block_size)
    z = bs.ops.to_sparse_shaped(dense, layout, sparsity_block_size)
    return (x,
            bs.ops.to_dense(x * 2, layout, sparsity_block_size),
            bs.ops.from_blksprs(y, layout, sparsity_block_size, fill_value=-1.0),
            bs.ops.to_dense_shaped(z, layout, sparsity_block_size))


def test_compiled_conversions_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_dense(layout, sparsity_block_size, generator), layout, sparsity_block_size), ()

    # The dense gradient sums the contributions of several conversions, whose order may differ.
    _assert_compiled_matches_eager(_conversions, make_arguments, differentiable=(0,), exact_gradients=False)


def _row_striped_conversions(dense, layout, row_striped_layout, sparsity_block_size):
    x = bs.ops.to_sparse_row_striped(dense, row_striped_layout, sparsity_block_size)
    return (bs.ops.is_row_striped_layout(layout),
            bs.ops.is_row_striped_layout(row_striped_layout),
            x,
            bs.ops.to_dense_row_striped(x * 2, row_striped_layout, sparsity_block_size))


def test_compiled_row_striped_conversions_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        row_striped_layout = layout.any(dim=-1, keepdim=True).expand_as(layout).contiguous()
        layout[0, 0, 0], layout[0, 0, -1] = True, False
        return (_dense(layout, sparsity_block_size, generator), layout, row_striped_layout,
                sparsity_block_size), ()

    _assert_compiled_matches_eager(_row_striped_conversions, make_arguments, differentiable=(0,),
                                   exact_gradients=False)


def _adaption(x, layout, sparsity_block_size_from, sparsity_block_size_to):
    y, layout_y = bs.ops.adapt_layout(x, layout, sparsity_block_size_from, sparsity_block_size_to)
    return y, layout_y, bs.layouting.build_sparsity_layout_adaption(x, layout, sparsity_block_size_from,
                                                                    sparsity_block_size_to)


def test_compiled_adapt_layout_matches_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size,
                32 if sparsity_block_size == 16 else 16), ()

    _assert_compiled_matches_eager(_adaption, make_arguments, differentiable=(0,))


def _matmul(x, layout_x, y, layout_y, sparsity_block_size):
    layout_o = bs.layouting.build_sparsity_layout_matmul_fast(layout_x, layout_y)
    return bs.ops.matmul(x, layout_x, y, layout_y, layout_o, sparsity_block_size), layout_o


def test_compiled_matmul_matches_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout_x = _layout(shape, generator)
        layout_y = _layout((shape[0], shape[2], shape[1] + 1), generator)
        return (_compressed(layout_x, sparsity_block_size, generator), layout_x,
                _compressed(layout_y, sparsity_block_size, generator), layout_y, sparsity_block_size), ()

    _assert_compiled_matches_eager(_matmul, make_arguments, differentiable=(0, 2), exact_values=False,
                                   exact_gradients=False)


def _transpose(x, layout, sparsity_block_size):
    return bs.ops.transpose(x, layout, sparsity_block_size)


def test_compiled_transpose_matches_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size), ()

    _assert_compiled_matches_eager(_transpose, make_arguments, differentiable=(0,))


def _partitioning(x, layout, partitions, sparsity_block_size):
    y, layout_y = bs.ops.split(x, layout, partitions, 2, sparsity_block_size)
    z, layout_z = bs.ops.merge(y * 2, layout_y, partitions, 2, sparsity_block_size)
    return y, layout_y, z, layout_z


def test_compiled_split_and_merge_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, shape[2], sparsity_block_size), ()

    _assert_compiled_matches_eager(_partitioning, make_arguments, differentiable=(0,), exact_gradients=False)


def _repeat(x, layout, sparsity_block_size):
    y, layout_y = bs.ops.repeat(x, layout, (2, 1, 2), sparsity_block_size)
    z, layout_z = bs.ops.repeat_interleave(x, layout, 3, sparsity_block_size)
    return y, layout_y, z, layout_z


def test_compiled_repeat_matches_eager():
    """The backward of the repetitions accumulates the gradients of the copies with atomic additions."""

    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size), ()

    _assert_compiled_matches_eager(_repeat, make_arguments, differentiable=(0,), exact_gradients=False)


def _gather(source, layout, indices, sparsity_block_size):
    return bs.ops.gather(source, layout, 2, indices, layout, sparsity_block_size)


def test_compiled_gather_matches_eager():
    """The backward of the gather scatters the gradients with atomic additions."""

    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout,
                _indices(layout, sparsity_block_size, generator), sparsity_block_size), ()

    _assert_compiled_matches_eager(_gather, make_arguments, differentiable=(0,), exact_gradients=False)


def _scatter(source, layout, indices, unique_indices, layout_target, sparsity_block_size):
    summed = bs.ops.scatter_reduce(source, layout, 2, indices, layout_target, sparsity_block_size, reduce_op="sum")
    # The non-reducing scatter has no backward.
    scattered = bs.ops.scatter(source.detach(), layout, 2, unique_indices, layout_target, sparsity_block_size)
    return summed, scattered


def test_compiled_scatter_matches_eager():
    """The summing scatter accumulates overlapping values with atomic additions."""

    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout,
                _indices(layout, sparsity_block_size, generator),
                _indices(layout, sparsity_block_size, generator, unique=True),
                _layout(shape, generator), sparsity_block_size), ()

    _assert_compiled_matches_eager(_scatter, make_arguments, differentiable=(0,), exact_values=False)


def _softmax(x, layout, sparsity_block_size):
    return (bs.ops.softmax(x, layout, sparsity_block_size),
            bs.ops.softmax(x, layout, sparsity_block_size, flag_fused=False),
            bs.ops.softmax_fused(x, layout, sparsity_block_size))


def test_compiled_softmax_matches_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size), ()

    _assert_compiled_matches_eager(_softmax, make_arguments, differentiable=(0,), exact_values=False,
                                   exact_gradients=False)


def _row_wise(x, layout, sparsity_block_size):
    sums, layout_sums = bs.ops.misc.row_wise_sum(x, layout, sparsity_block_size)
    maxima, _ = bs.ops.misc.row_wise_max(x, layout, sparsity_block_size, flag_slice_only=True)
    return (sums, layout_sums, maxima,
            bs.ops.misc.row_wise_add(x, layout, maxima, sparsity_block_size),
            bs.ops.misc.row_wise_sub(x, layout, sums, sparsity_block_size))


def test_compiled_row_wise_operations_match_eager():
    """The row-wise sum accumulates the partial sums of the blocks of a row with atomic additions."""

    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        return (_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size), ()

    _assert_compiled_matches_eager(_row_wise, make_arguments, exact_values=False)


def _broadcast(x, y, layout, sparsity_block_size):
    return (bs.ops.misc.broadcast_add(x, y, layout, sparsity_block_size),
            bs.ops.misc.broadcast_sub(x, y, layout, sparsity_block_size))


def test_compiled_broadcast_operations_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        b, r, c = shape
        return (torch.randn((b, r * sparsity_block_size), generator=generator, device=DEVICE),
                torch.randn((b, c * sparsity_block_size), generator=generator, device=DEVICE),
                layout, sparsity_block_size), ()

    _assert_compiled_matches_eager(_broadcast, make_arguments)


def _layouts(dense, layout_x, layout_y, indices, size_target, valid_lengths, window_size, sparsity_block_size):
    causal_layout = bs.layouting.build_causal_self_attention_layout(valid_lengths, sparsity_block_size)
    window_layout = bs.layouting.build_causal_window_self_attention_layout(valid_lengths, sparsity_block_size,
                                                                           window_size)
    return (bs.layouting.build_sparsity_layout(dense, sparsity_block_size),
            bs.layouting.build_sparsity_layout_full(dense, sparsity_block_size),
            bs.layouting.build_sparsity_layout_matmul(layout_x, layout_y),
            bs.layouting.build_sparsity_layout_matmul_fast(layout_x, layout_y),
            bs.layouting.build_sparsity_layout_matmul_outer(layout_x, layout_y),
            bs.layouting.build_distribution_layout(indices, layout_x, 2, size_target, sparsity_block_size),
            causal_layout,
            bs.layouting.build_causal_self_attention_mask(valid_lengths, causal_layout, sparsity_block_size),
            window_layout,
            bs.layouting.build_causal_window_self_attention_mask(valid_lengths, window_layout, sparsity_block_size,
                                                                 window_size))


def test_compiled_layout_builders_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout_x = _layout(shape, generator)
        layout_y = _layout((shape[0], shape[2], shape[1]), generator)
        dense = bs.ops.to_dense(_compressed(layout_x, sparsity_block_size, generator), layout_x, sparsity_block_size)
        size_target = torch.Size((shape[0], shape[1] * sparsity_block_size, shape[2] * sparsity_block_size))
        valid_lengths = torch.randint(1, 4 * sparsity_block_size, (shape[0],), generator=generator, device=DEVICE)
        return (dense, layout_x, layout_y, _indices(layout_x, sparsity_block_size, generator), size_target,
                valid_lengths, sparsity_block_size + 3, sparsity_block_size), ()

    _assert_compiled_matches_eager(_layouts, make_arguments)


def _linear(x, layout, sparsity_block_size, linear, linear_without_bias, layout_cache):
    y, layout_y = bs.utils.apply_torch_linear(x, layout, sparsity_block_size, linear)
    z, layout_z = bs.utils.apply_torch_linear_cached(x, layout, sparsity_block_size, linear_without_bias,
                                                     layout_cache=layout_cache)
    return y, layout_y, z, layout_z


def test_compiled_linear_matches_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        in_features = shape[2] * sparsity_block_size
        linear = torch.nn.Linear(in_features, 2 * sparsity_block_size, device=DEVICE)
        linear_without_bias = torch.nn.Linear(in_features, 3 * sparsity_block_size, bias=False, device=DEVICE)
        return ((_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size, linear,
                 linear_without_bias, {}),
                (linear.weight, linear.bias, linear_without_bias.weight))

    _assert_compiled_matches_eager(_linear, make_arguments, differentiable=(0,), exact_values=False,
                                   exact_gradients=False)


def _row_wise_modules(x, layout, sparsity_block_size, normalisation, dropout):
    return (bs.utils.apply_torch_normalisation(x, layout, sparsity_block_size, normalisation),
            bs.utils.apply_torch_dropout(x, layout, sparsity_block_size, dropout),
            bs.utils.apply_function_applicable_row_wise(x, layout, sparsity_block_size, torch.nn.functional.gelu))


def test_compiled_row_wise_modules_match_eager():
    def make_arguments(sparsity_block_size, shape, generator):
        layout = _layout(shape, generator)
        normalisation = torch.nn.LayerNorm(shape[2] * sparsity_block_size, device=DEVICE)
        # Dropout is compared in evaluation mode, since compiled code draws other random numbers.
        dropout = torch.nn.Dropout(0.5).eval()
        return ((_compressed(layout, sparsity_block_size, generator), layout, sparsity_block_size, normalisation,
                 dropout), (normalisation.weight, normalisation.bias))

    _assert_compiled_matches_eager(_row_wise_modules, make_arguments, differentiable=(0,), exact_values=False,
                                   exact_gradients=False)


# Accumulating kernels

def test_accumulating_kernels_are_not_autotuned():
    """Autotuned kernels that ``torch.compile`` captures must not accumulate into their outputs.

    ``torch.compile`` rebuilds autotuners with pruned configurations without ``reset_to_zero`` and ``restore_value``,
    so that Inductor benchmarks them on their live outputs.

    """
    # Some modules share their names with operations that the packages export, so they are imported by their paths
    modules = [importlib.import_module(f"blksprs.{name}") for name in (
        "ops.conversion", "ops.distribution", "ops.flow", "ops.matmul", "ops.partitioning", "ops.repeat", "ops.softmax",
        "ops.transpose", "ops.misc.broadcast_ops", "ops.misc.row_wise", "layouting.distribution_layout",
        "layouting.sparsity_layout")]
    kernels = {name: value for module in modules for name, value in vars(module).items()
               if isinstance(value, (triton.runtime.Autotuner, triton.runtime.JITFunction))}

    assert isinstance(kernels["matmul_kernel"], triton.runtime.Autotuner)
    for name in ("row_wise_sum_kernel", "flow_push_kernel", "scatter_reduce_kernel"):
        assert isinstance(kernels[name], triton.runtime.JITFunction), name
    for name, kernel in kernels.items():
        if isinstance(kernel, triton.runtime.Autotuner):
            assert "atomic_add" not in kernel.fn.src, name


def test_compiled_accumulating_operators_match_eager_on_first_calls():
    """Compiles the accumulating Triton operators into single graphs and checks the first call after every compilation.

    The backward of the repetitions, which pushes gradients onto their source blocks, runs eagerly within compiled code,
    so that its operator is compiled directly.

    """

    def accumulate(x, layout, layout_indices, layout_rows, packed_indices_rows, n_rows, gradient, layout_repeated,
                   layout_indices_repeated, packed_indices_repeated, indices, layout_target, packed_indices_target,
                   n_target_blocks, sparsity_block_size):
        return (row_wise.row_wise_sum_forward(x, layout_indices, layout_rows, packed_indices_rows,
                                              sparsity_block_size, n_rows, False),
                flow.flow_push_forward(gradient, layout_repeated, layout_indices_repeated, packed_indices_repeated,
                                       sparsity_block_size, x.size(0)),
                distribution.scatter_reduce_forward(x, layout, layout_indices, 2, indices, layout_target,
                                                    packed_indices_target, sparsity_block_size, n_target_blocks,
                                                    "sum"))

    compiled = torch.compile(accumulate, fullgraph=True)

    for seed, (sparsity_block_size, shape) in enumerate(VARIANTS):
        generator = _generator(seed)
        layout = _layout(shape, generator)
        x = as_base_tensor(_compressed(layout, sparsity_block_size, generator))
        layout_rows = layout.any(dim=-1, keepdim=True)
        layout_repeated = layout.repeat(2, 1, 2)
        packed_indices_repeated = build_packed_indices(layout).reshape(layout.shape).repeat(2, 1, 2).reshape(-1)
        gradient = as_base_tensor(_compressed(layout_repeated, sparsity_block_size, generator))
        layout_target = _layout(shape, generator)
        arguments = (x, layout, build_layout_indices(layout), layout_rows, build_packed_indices(layout_rows),
                     int(layout_rows.sum()), gradient, layout_repeated, build_layout_indices(layout_repeated),
                     packed_indices_repeated.contiguous(),
                     as_base_tensor(_indices(layout, sparsity_block_size, generator)),
                     layout_target, build_packed_indices(layout_target), int(layout_target.sum()),
                     sparsity_block_size)

        expected = accumulate(*arguments)
        actual = compiled(*arguments)

        _assert_matches(list(actual), list(expected), exact=False)


# Flash Attention

class _Attention(torch.nn.Module):
    """Projects dense inputs and attends with Flash Attention, optionally with a learned relative term."""

    def __init__(self, relative: bool):
        super().__init__()
        self.projection = torch.nn.Linear(FLASH_HEAD_DIMENSION, 3 * FLASH_HEAD_DIMENSION, device=DEVICE,
                                          dtype=torch.float16)
        self.relative = relative
        n_relations = FLASH_RELATION_BOUNDS[0][1] - FLASH_RELATION_BOUNDS[0][0] + 1
        self.embeddings = torch.nn.Parameter(
            (torch.randn((FLASH_BATCHES, n_relations, FLASH_HEAD_DIMENSION), generator=_generator(0),
                         device=DEVICE) * 0.3).half()) if relative else None

    def forward(self, x, layout, attention_layout, causal_lengths, positions):
        sparsity_block_size = FLASH_SPARSITY_BLOCK_SIZE
        q, k, v = (bs.ops.to_sparse(part.contiguous(), layout, sparsity_block_size)
                   for part in self.projection(x).chunk(3, dim=-1))
        if self.relative:
            output = bs.ops.flash_attention_relative_embeddings(
                q, layout, k, layout, v, layout, attention_layout, sparsity_block_size, positions, positions,
                self.embeddings, FLASH_RELATION_BOUNDS, key_relations_are_unique=(True,),
                causal_lengths=causal_lengths, sparsity_layout_o=layout)
        else:
            layout_cache = bs.ops.flash_attention_build_layout_cache(
                attention_layout, layout, layout, layout, sparsity_layout_o=layout,
                sparsity_block_size=sparsity_block_size)
            output = bs.ops.flash_attention(q, layout, k, layout, v, layout, attention_layout, sparsity_block_size,
                                            causal_lengths=causal_lengths, sparsity_layout_o=layout,
                                            layout_cache=layout_cache)
        return bs.ops.to_dense(output, layout, sparsity_block_size)


def _flash_inputs(length: int, generator: torch.Generator) -> tuple:
    n_blocks = length // FLASH_SPARSITY_BLOCK_SIZE
    layout = torch.ones((FLASH_BATCHES, n_blocks, FLASH_HEAD_DIMENSION // FLASH_SPARSITY_BLOCK_SIZE),
                        dtype=torch.bool, device=DEVICE)
    attention_layout = torch.tril(torch.ones((FLASH_BATCHES, n_blocks, n_blocks), dtype=torch.bool, device=DEVICE))
    x = torch.randn((FLASH_BATCHES, length, FLASH_HEAD_DIMENSION), generator=generator, device=DEVICE).half()
    causal_lengths = torch.tensor([length, length - 5], device=DEVICE)
    positions = torch.arange(length, device=DEVICE).repeat(1, FLASH_BATCHES, 1)
    return x, layout, attention_layout, causal_lengths, positions


@pytest.mark.parametrize("relative", [False, True])
def test_compiled_flash_attention_model_matches_eager(relative: bool):
    model = _Attention(relative)
    compiled = torch.compile(model)
    parameters = tuple(model.parameters())

    for seed, length in enumerate((64, 96, 128)):
        arguments = _flash_inputs(length, _generator(seed))

        expected, expected_gradients = _evaluate(model, arguments, (0,), parameters)
        actual, actual_gradients = _evaluate(compiled, arguments, (0,), parameters)

        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(actual_gradients, expected_gradients, rtol=5e-3, atol=5e-3)


# Gradients

def test_compiled_backward_accepts_gradients_of_eager_operations():
    """The registered backward functions return base tensors, which compiled backward graphs expect."""
    sparsity_block_size = 16
    generator = _generator(0)
    layout = _layout((2, 3, 4), generator)
    dense = _dense(layout, sparsity_block_size, generator).requires_grad_(True)
    expected_dense = dense.detach().clone().requires_grad_(True)

    def attend(values):
        return bs.ops.softmax(bs.ops.to_sparse(values, layout, sparsity_block_size), layout, sparsity_block_size)

    actual = bs.ops.to_dense(torch.compile(attend)(dense), layout, sparsity_block_size)
    expected = bs.ops.to_dense(attend(expected_dense), layout, sparsity_block_size)
    actual.backward(_cotangent(actual))
    expected.backward(_cotangent(expected))

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(dense.grad, expected_dense.grad, rtol=RTOL, atol=ATOL)


_DIFFERENTIABLE_OPERATIONS = {
    "to_dense": lambda x, layout, sbs, indices: bs.ops.to_dense(x, layout, sbs),
    "adapt_layout": lambda x, layout, sbs, indices: bs.ops.adapt_layout(x, layout, sbs, 2 * sbs)[0],
    "matmul": lambda x, layout, sbs, indices: bs.ops.matmul(
        x, layout, bs.ops.transpose(x, layout, sbs)[0], layout.transpose(-1, -2).contiguous(),
        torch.ones((layout.size(0), layout.size(1), layout.size(1)), dtype=torch.bool, device=DEVICE), sbs),
    "transpose": lambda x, layout, sbs, indices: bs.ops.transpose(x, layout, sbs)[0],
    "split": lambda x, layout, sbs, indices: bs.ops.split(x, layout, layout.size(2), 2, sbs)[0],
    "merge": lambda x, layout, sbs, indices: bs.ops.merge(x, layout, layout.size(0), 2, sbs)[0],
    "repeat": lambda x, layout, sbs, indices: bs.ops.repeat(x, layout, (2, 1, 1), sbs)[0],
    "repeat_interleave": lambda x, layout, sbs, indices: bs.ops.repeat_interleave(x, layout, 2, sbs)[0],
    "gather": lambda x, layout, sbs, indices: bs.ops.gather(x, layout, 2, indices, layout, sbs),
    "scatter_reduce": lambda x, layout, sbs, indices: bs.ops.scatter_reduce(x, layout, 2, indices, layout, sbs),
    "softmax": lambda x, layout, sbs, indices: bs.ops.softmax(x, layout, sbs),
    "softmax_regular": lambda x, layout, sbs, indices: bs.ops.softmax(x, layout, sbs, flag_fused=False),
}


@pytest.mark.parametrize("operation", list(_DIFFERENTIABLE_OPERATIONS))
@pytest.mark.parametrize("subclass", [False, True])
def test_gradients_are_base_tensors(operation: str, subclass: bool):
    """The gradients that the operations pass on are base tensors, also for operands of the ``BlksprsTensor`` subclass.

    A hook observes the gradient as it reaches the operand. ``torch.autograd.grad`` would instead convert its results to
    the subclass of the operands.

    """
    sparsity_block_size = 16
    generator = _generator(0)
    layout = _layout((2, 2, 2), generator)
    x = _compressed(layout, sparsity_block_size, generator)
    x = (x if subclass else as_base_tensor(x)).requires_grad_(True)
    gradient_types = []
    x.register_hook(lambda gradient: gradient_types.append(type(gradient)))

    output = _DIFFERENTIABLE_OPERATIONS[operation](x, layout, sparsity_block_size,
                                                   _indices(layout, sparsity_block_size, generator))
    output.backward(_cotangent(output))

    assert gradient_types == [torch.Tensor]


# Layout caches

def test_compiled_calls_observe_in_place_layout_mutation():
    sparsity_block_size = 16

    def convert(x, layout, layout_cache):
        return bs.ops.to_dense(x, layout, sparsity_block_size, layout_cache=layout_cache)

    def multiply(x, layout_x, y, layout_y, layout_o, layout_cache):
        return bs.ops.matmul(x, layout_x, y, layout_y, layout_o, sparsity_block_size, layout_cache=layout_cache)

    compiled_convert = torch.compile(convert)
    compiled_multiply = torch.compile(multiply)
    generator = _generator(0)
    dense = torch.randn((1, 4 * sparsity_block_size, 4 * sparsity_block_size), generator=generator, device=DEVICE)
    layout = torch.ones((1, 4, 4), dtype=torch.bool, device=DEVICE)
    layout_y = torch.ones((1, 4, 4), dtype=torch.bool, device=DEVICE)
    layout_o = torch.ones((1, 4, 4), dtype=torch.bool, device=DEVICE)
    convert_cache, multiply_cache = {}, {}

    for step in range(4):
        if step > 0:
            # Indexed assignment and copies update the layouts in place
            layout[0, step - 1, step] = False
            layout_o.copy_(_layout((1, 4, 4), generator))
        x = bs.ops.to_sparse(dense, layout, sparsity_block_size)
        y = bs.ops.to_sparse(dense, layout_y, sparsity_block_size)

        actual = compiled_convert(x, layout, convert_cache)
        expected = convert(x, layout, {})
        assert torch.equal(actual, expected)

        actual = compiled_multiply(x, layout, y, layout_y, layout_o, multiply_cache)
        expected = multiply(x, layout, y, layout_y, layout_o, {})
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_compiled_cached_linear_uses_parameters_updated_in_place():
    sparsity_block_size = 16
    generator = _generator(0)
    layout = _layout((2, 2, 2), generator)
    x = _compressed(layout, sparsity_block_size, generator)
    linear = torch.nn.Linear(2 * sparsity_block_size, 3 * sparsity_block_size, device=DEVICE)
    layout_cache = {}

    def project(values):
        return bs.utils.apply_torch_linear_cached(values, layout, sparsity_block_size, linear,
                                                  layout_cache=layout_cache)[0]

    compiled = torch.compile(project)
    with torch.no_grad():
        for _ in range(3):
            actual = compiled(x)
            expected = bs.utils.apply_torch_linear(x, layout, sparsity_block_size, linear)[0]
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            linear.weight.add_(1.0)


@pytest.mark.parametrize("cached", [False, True])
def test_recompilations_are_bounded_for_new_layout_objects(cached: bool):
    """New layout objects of an unchanged shape do not recompile once the compiled code is dynamic."""
    sparsity_block_size = 16
    generator = _generator(0)
    dense = torch.randn((2, 4 * sparsity_block_size, 4 * sparsity_block_size), generator=generator, device=DEVICE)
    layout_cache = {} if cached else None

    def convert(x, layout, layout_y, layout_o):
        y = bs.ops.transpose(x, layout, sparsity_block_size)[0]
        z = bs.ops.matmul(x, layout, y, layout_y, layout_o, sparsity_block_size)
        return bs.ops.to_dense(x, layout, sparsity_block_size, layout_cache=layout_cache), z

    compiled = torch.compile(convert)

    def call() -> None:
        # Every call uses new layout objects with between two and fifteen active blocks
        layout = _layout((2, 4, 4), generator, density=0.5)
        layout.view(-1)[1] = False
        layout_y = layout.transpose(-1, -2).contiguous()
        layout_o = _layout((2, 4, 4), generator, density=0.5)
        x = bs.ops.to_sparse(dense, layout, sparsity_block_size)
        actual = compiled(x, layout, layout_y, layout_o)
        expected = convert(x, layout, layout_y, layout_o)
        assert torch.equal(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)

    for _ in range(4):
        call()
    graphs = counters["stats"]["unique_graphs"]

    for _ in range(8):
        call()

    assert counters["stats"]["unique_graphs"] == graphs


# Public operation calls

def _flash_operands(generator: torch.Generator) -> dict:
    x, layout, attention_layout, causal_lengths, positions = _flash_inputs(64, generator)
    q, k, v = (bs.ops.to_sparse(torch.randn(x.shape, generator=generator, device=DEVICE).half(), layout,
                                FLASH_SPARSITY_BLOCK_SIZE) for _ in range(3))
    return dict(q=q, sparsity_layout_q=layout, k=k, sparsity_layout_k=layout, v=v, sparsity_layout_v=layout,
                attention_layout=attention_layout, sparsity_block_size=FLASH_SPARSITY_BLOCK_SIZE,
                causal_lengths=causal_lengths, sparsity_layout_o=layout), positions


def _public_calls() -> dict:
    """Returns every public operation with a function that builds keyword arguments for a valid call."""
    sbs = 16

    def inputs(generator):
        layout = _layout((2, 3, 4), generator)
        return layout, _compressed(layout, sbs, generator)

    def basic(generator):
        layout, x = inputs(generator)
        return dict(x=x, sparsity_layout=layout, sparsity_block_size=sbs)

    def dense(generator):
        layout, x = inputs(generator)
        return dict(x=bs.ops.to_dense(x, layout, sbs), sparsity_layout=layout, sparsity_block_size=sbs)

    def row_striped(generator):
        layout, x = inputs(generator)
        layout = layout.any(dim=-1, keepdim=True).expand_as(layout).contiguous()
        return dict(x=_dense(layout, sbs, generator), sparsity_layout=layout, sparsity_block_size=sbs)

    def row_striped_compressed(generator):
        arguments = row_striped(generator)
        arguments["x"] = bs.ops.to_sparse_row_striped(**arguments)
        return arguments

    def matmul_call(generator):
        layout, x = inputs(generator)
        layout_y = _layout((2, 4, 2), generator)
        return dict(x=x, sparsity_layout_x=layout, y=_compressed(layout_y, sbs, generator), sparsity_layout_y=layout_y,
                    sparsity_layout_output=_layout((2, 3, 2), generator), sparsity_block_size=sbs)

    def distribution_call(generator, name_source, name_layout, name_target, reduce_op=None):
        layout, x = inputs(generator)
        arguments = {name_source: x, name_layout: layout, "dim": 2,
                     "idx": _indices(layout, sbs, generator, unique=reduce_op is None),
                     name_target: layout, "sparsity_block_size": sbs}
        if reduce_op is not None:
            arguments["reduce_op"] = reduce_op
        return arguments

    def row_wise_operand(generator):
        layout, x = inputs(generator)
        y, _ = bs.ops.misc.row_wise_sum(x, layout, sbs, flag_slice_only=True)
        return dict(x=x, sparsity_layout_x=layout, y=y, sparsity_block_size=sbs)

    def broadcast(generator):
        layout, _ = inputs(generator)
        return dict(x=torch.randn((2, 3 * sbs), generator=generator, device=DEVICE),
                    y=torch.randn((2, 4 * sbs), generator=generator, device=DEVICE),
                    sparsity_layout_output=layout, sparsity_block_size=sbs)

    def layouts(generator):
        layout, _ = inputs(generator)
        return dict(sparsity_layout_x=layout, sparsity_layout_y=_layout((2, 4, 2), generator))

    def causal(generator, window: bool, mask: bool):
        valid_lengths = torch.tensor([3 * sbs + 5, sbs], device=DEVICE)
        arguments: dict = dict(valid_lengths=valid_lengths)
        if mask:
            arguments["attention_layout"] = (
                bs.layouting.build_causal_window_self_attention_layout(valid_lengths, sbs, sbs + 3) if window else
                bs.layouting.build_causal_self_attention_layout(valid_lengths, sbs))
        arguments["sparsity_block_size"] = sbs
        if window:
            arguments["window_size"] = sbs + 3
        return arguments

    def module_call(module_name, make_module):
        def build(generator):
            layout, x = inputs(generator)
            return {"x": x, "sparsity_layout": layout, "sparsity_block_size": sbs, module_name: make_module()}
        return build

    def flash(generator):
        return _flash_operands(generator)[0]

    def relative(generator, multiple: bool):
        arguments, positions = _flash_operands(generator)
        n_relations = FLASH_RELATION_BOUNDS[0][1] - FLASH_RELATION_BOUNDS[0][0] + 1
        embeddings = torch.randn((FLASH_BATCHES, n_relations, FLASH_HEAD_DIMENSION), generator=generator,
                                 device=DEVICE).half()
        if multiple:
            arguments.update(query_relations=positions, key_relations=positions, relative_embeddings=embeddings,
                             relation_bounds=FLASH_RELATION_BOUNDS, key_relations_are_unique=(True,))
        else:
            arguments.update(query_relations=positions[0], key_relations=positions[0], relative_embedding=embeddings,
                             relation_min=FLASH_RELATION_BOUNDS[0][0], relation_max=FLASH_RELATION_BOUNDS[0][1],
                             key_relations_are_unique=True)
        return arguments

    def flash_layout_cache(generator):
        arguments, _ = _flash_operands(generator)
        return dict(attention_layout=arguments["attention_layout"], sparsity_layout_q=arguments["sparsity_layout_q"],
                    sparsity_layout_k=arguments["sparsity_layout_k"], sparsity_layout_v=arguments["sparsity_layout_v"],
                    sparsity_block_size=FLASH_SPARSITY_BLOCK_SIZE)

    def adapt(generator):
        layout, x = inputs(generator)
        return dict(x=x, sparsity_layout_from=layout, sparsity_block_size_from=sbs, sparsity_block_size_to=2 * sbs)

    def distribution_layout_call(generator):
        layout, _ = inputs(generator)
        return dict(indices=_indices(layout, sbs, generator), sparsity_layout_indices=layout, dim=2,
                    size_target=torch.Size((2, 3 * sbs, 4 * sbs)), sparsity_block_size=sbs)

    def repeat_call(generator, repeats):
        layout, x = inputs(generator)
        return dict(x=x, sparsity_layout_x=layout, repeats=repeats, sparsity_block_size=sbs)

    def partition(generator, partitions):
        arguments = basic(generator)
        arguments.update(partitions=partitions, dim=2)
        return arguments

    def merge_call(generator):
        arguments = basic(generator)
        arguments.update(partitions=2, dim=2)
        return arguments

    def basic_layout_only(generator):
        layout, _ = inputs(generator)
        return dict(sparsity_layout=layout)

    def dense_only(generator):
        arguments = dense(generator)
        del arguments["sparsity_layout"]
        return arguments

    return {
        "ops.adapt_layout": (bs.ops.adapt_layout, adapt),
        "ops.flash_attention": (bs.ops.flash_attention, flash),
        "ops.flash_attention_build_layout_cache": (bs.ops.flash_attention_build_layout_cache, flash_layout_cache),
        "ops.flash_attention_relative_embedding": (bs.ops.flash_attention_relative_embedding,
                                                   lambda generator: relative(generator, False)),
        "ops.flash_attention_relative_embeddings": (bs.ops.flash_attention_relative_embeddings,
                                                    lambda generator: relative(generator, True)),
        "ops.flash_attention_warm_up": (bs.ops.flash_attention_warm_up, lambda generator: dict(
            sparsity_block_size=FLASH_SPARSITY_BLOCK_SIZE, head_dimension=FLASH_HEAD_DIMENSION, dtype=torch.float16,
            shapes=[(FLASH_BATCHES, 64)], backward=False)),
        "ops.from_blksprs": (bs.ops.from_blksprs, basic),
        "ops.gather": (bs.ops.gather, lambda generator: distribution_call(
            generator, "src", "sparsity_layout_src", "sparsity_layout_idx")),
        "ops.is_row_striped_layout": (bs.ops.is_row_striped_layout, basic_layout_only),
        "ops.matmul": (bs.ops.matmul, matmul_call),
        "ops.merge": (bs.ops.merge, merge_call),
        "ops.repeat": (bs.ops.repeat, lambda generator: repeat_call(generator, (2, 1, 2))),
        "ops.repeat_interleave": (bs.ops.repeat_interleave, lambda generator: repeat_call(generator, 2)),
        "ops.scatter": (bs.ops.scatter, lambda generator: distribution_call(
            generator, "src", "sparsity_layout_src", "sparsity_layout_tgt")),
        "ops.scatter_reduce": (bs.ops.scatter_reduce, lambda generator: distribution_call(
            generator, "src", "sparsity_layout_src", "sparsity_layout_tgt", reduce_op="sum")),
        "ops.softmax": (bs.ops.softmax, basic),
        "ops.softmax_fused": (bs.ops.softmax_fused, basic),
        "ops.split": (bs.ops.split, lambda generator: partition(generator, 2)),
        "ops.to_blksprs": (bs.ops.to_blksprs, dense),
        "ops.to_dense": (bs.ops.to_dense, basic),
        "ops.to_dense_row_striped": (bs.ops.to_dense_row_striped, row_striped_compressed),
        "ops.to_dense_shaped": (bs.ops.to_dense_shaped, basic),
        "ops.to_sparse": (bs.ops.to_sparse, dense),
        "ops.to_sparse_row_striped": (bs.ops.to_sparse_row_striped, row_striped),
        "ops.to_sparse_shaped": (bs.ops.to_sparse_shaped, dense),
        "ops.transpose": (bs.ops.transpose, basic),
        "ops.misc.broadcast_add": (bs.ops.misc.broadcast_add, broadcast),
        "ops.misc.broadcast_sub": (bs.ops.misc.broadcast_sub, broadcast),
        "ops.misc.row_wise_add": (bs.ops.misc.row_wise_add, row_wise_operand),
        "ops.misc.row_wise_max": (bs.ops.misc.row_wise_max, basic),
        "ops.misc.row_wise_sub": (bs.ops.misc.row_wise_sub, row_wise_operand),
        "ops.misc.row_wise_sum": (bs.ops.misc.row_wise_sum, basic),
        "layouting.build_distribution_layout": (bs.layouting.build_distribution_layout, distribution_layout_call),
        "layouting.build_causal_self_attention_layout": (
            bs.layouting.build_causal_self_attention_layout,
            lambda generator: causal(generator, window=False, mask=False)),
        "layouting.build_causal_self_attention_mask": (
            bs.layouting.build_causal_self_attention_mask,
            lambda generator: causal(generator, window=False, mask=True)),
        "layouting.build_causal_window_self_attention_layout": (
            bs.layouting.build_causal_window_self_attention_layout,
            lambda generator: causal(generator, window=True, mask=False)),
        "layouting.build_causal_window_self_attention_mask": (
            bs.layouting.build_causal_window_self_attention_mask,
            lambda generator: causal(generator, window=True, mask=True)),
        "layouting.build_sparsity_layout": (bs.layouting.build_sparsity_layout, dense_only),
        "layouting.build_sparsity_layout_adaption": (bs.layouting.build_sparsity_layout_adaption, adapt),
        "layouting.build_sparsity_layout_full": (bs.layouting.build_sparsity_layout_full, dense_only),
        "layouting.build_sparsity_layout_matmul": (bs.layouting.build_sparsity_layout_matmul, layouts),
        "layouting.build_sparsity_layout_matmul_fast": (bs.layouting.build_sparsity_layout_matmul_fast, layouts),
        "layouting.build_sparsity_layout_matmul_outer": (bs.layouting.build_sparsity_layout_matmul_outer, layouts),
        "utils.apply_function_applicable_row_wise": (bs.utils.apply_function_applicable_row_wise, module_call(
            "function", lambda: torch.nn.functional.gelu)),
        "utils.apply_torch_dropout": (bs.utils.apply_torch_dropout, module_call(
            "dropout", lambda: torch.nn.Dropout(0.5).eval())),
        "utils.apply_torch_linear": (bs.utils.apply_torch_linear, module_call(
            "linear", lambda: torch.nn.Linear(4 * sbs, 2 * sbs, device=DEVICE))),
        "utils.apply_torch_linear_cached": (bs.utils.apply_torch_linear_cached, module_call(
            "linear", lambda: torch.nn.Linear(4 * sbs, 2 * sbs, device=DEVICE))),
        "utils.apply_torch_normalisation": (bs.utils.apply_torch_normalisation, module_call(
            "normalisation", lambda: torch.nn.LayerNorm(4 * sbs, device=DEVICE))),
    }


def test_public_calls_cover_every_public_operation():
    namespaces = {"ops": bs.ops, "ops.misc": bs.ops.misc, "layouting": bs.layouting, "utils": bs.utils}
    utilities = {"utils.disable_contiguous", "utils.disable_validation", "utils.do_shape_blocksparse",
                 "utils.enable_contiguous", "utils.enable_validation", "utils.undo_shape_blocksparse"}
    public = {f"{prefix}.{name}" for prefix, namespace in namespaces.items() for name in namespace.__all__
              if inspect.isfunction(getattr(namespace, name))}

    assert public - utilities == set(_public_calls())


@pytest.mark.parametrize("name", list(_public_calls()))
def test_public_operations_accept_keyword_arguments_and_leave_operands_unchanged(name: str):
    function, make_arguments = _public_calls()[name]
    arguments = make_arguments(_generator(0))
    bound = inspect.signature(function).bind(**arguments)
    tensors = [value for value in arguments.values() if isinstance(value, torch.Tensor)]

    positional = _flatten(function(*bound.args, **bound.kwargs))
    keyword = _flatten(function(**arguments))

    if "row_wise_sum" in name or "scatter_reduce" in name:
        _assert_matches(keyword, positional, exact=False)
    elif name != "ops.flash_attention_build_layout_cache":
        _assert_matches(keyword, positional, exact=True)
    for tensor in tensors:
        assert vars(tensor) == {}


# Devices

def test_operand_device_guard_uses_the_first_cuda_tensor_argument(monkeypatch: pytest.MonkeyPatch):
    """Simulates a second device that is current, so that the guard switches to the device of the operands."""
    requested = []

    class RecordingDevice:
        def __init__(self, index):
            requested.append(index)

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            return False

    @on_operand_device
    def operation(*arguments, **keywords):
        return len(arguments) + len(keywords)

    cpu = torch.zeros(1)
    cuda = bs.BlksprsTensor(torch.zeros(1, device=DEVICE))
    other_device = DEVICE.index + 1

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "device", RecordingDevice)
        patch.setattr(torch.cuda, "device_count", lambda: other_device + 1)
        patch.setattr(torch.cuda, "current_device", lambda: other_device)

        assert operation(cpu, 1, value=cuda) == 3
        assert operation(cpu, value=cpu) == 2
        assert requested == [DEVICE.index]

        # Operands on the current device need no guard
        patch.setattr(torch.cuda, "current_device", lambda: DEVICE.index)
        assert operation(cuda) == 1
        assert requested == [DEVICE.index]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("compiled", [False, True])
def test_operations_run_on_the_device_of_their_operands(compiled: bool):
    """Operands on another device than the current one are processed on their device, eagerly and compiled."""
    sparsity_block_size = 16
    other = torch.device("cuda:1")

    def pipeline(dense, layout):
        x = bs.ops.to_sparse(dense, layout, sparsity_block_size)
        y = bs.ops.matmul(x, layout, bs.ops.transpose(x, layout, sparsity_block_size)[0],
                          layout.transpose(-1, -2).contiguous(), torch.ones_like(layout), sparsity_block_size)
        s = bs.ops.softmax(y, torch.ones_like(layout), sparsity_block_size)
        sums, _ = bs.ops.misc.row_wise_sum(s, torch.ones_like(layout), sparsity_block_size)
        return bs.ops.to_dense(s, torch.ones_like(layout), sparsity_block_size), sums

    function = torch.compile(pipeline) if compiled else pipeline
    generator = _generator(0)
    layout = torch.ones((2, 3, 3), dtype=torch.bool, device=DEVICE)
    dense = _dense(layout, sparsity_block_size, generator)
    expected = pipeline(dense, layout)

    with torch.cuda.device(DEVICE):
        actual = function(dense.to(other), layout.to(other))

    for actual_value, expected_value in zip(actual, expected):
        assert actual_value.device == other
        torch.testing.assert_close(actual_value.to(DEVICE), expected_value, rtol=RTOL, atol=ATOL)

    arguments, _ = _flash_operands(generator)
    expected_attention = bs.ops.flash_attention(**arguments)
    moved = {name: value.to(other) if isinstance(value, torch.Tensor) else value for name, value in arguments.items()}
    with torch.cuda.device(DEVICE):
        actual_attention = bs.ops.flash_attention(**moved)
    assert actual_attention.device == other
    torch.testing.assert_close(actual_attention.to(DEVICE), expected_attention, rtol=1e-3, atol=1e-3)
