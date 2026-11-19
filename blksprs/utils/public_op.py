import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

import torch
from torch import Tensor

P = ParamSpec("P")
R = TypeVar("R")


def public_op(fn: Callable[P, R]) -> Callable[P, R]:
    """Marks a public BLK-SPRS operation.

    Every public operation that launches kernels shares this decorator. It passes all arguments on unchanged, whether
    given by position or by keyword, records nothing on them, and runs the operation on the CUDA device of its operands,
    see :func:`on_operand_device`.

    """
    return on_operand_device(fn)


def opaque_public_op(fn: Callable[P, R]) -> Callable[P, R]:
    """Marks a public BLK-SPRS operation that ``torch.compile`` does not trace.

    The operation runs eagerly with real tensors, also when it is called from a compiled function, which graph-breaks
    at the call. This applies to operations that are not registered as Triton operators, such as Flash Attention.

    """
    return cast(Callable[P, R], torch.compiler.disable(public_op(fn)))


def on_operand_device(fn: Callable[P, R]) -> Callable[P, R]:
    """Runs ``fn`` with the device of its first CUDA tensor argument as the current CUDA device.

    Triton launches a kernel on the current CUDA device regardless of the device of its operands, so that operands on
    another GPU than the current one would otherwise be accessed from the wrong device. Calls without CUDA tensor
    arguments run unchanged.

    The device is only set in eager execution. While ``torch.compile`` traces ``fn``, the guard is omitted, so that it
    adds nothing to the traced graph: Inductor launches every kernel of a compiled graph on the device of its operands.
    The implementations of the registered Triton operators carry this decorator as well, so that operator calls that
    run eagerly within compiled code, for example after a graph break, also set the device of their operands.

    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # With a single device, every CUDA operand is on the current device
        if torch.compiler.is_compiling() or torch.cuda.device_count() < 2:
            return fn(*args, **kwargs)

        device_index = _operand_device_index(args, kwargs.values())
        if device_index is None or device_index == torch.cuda.current_device():
            return fn(*args, **kwargs)

        with torch.cuda.device(device_index):
            return fn(*args, **kwargs)

    return wrapper


def _operand_device_index(args: tuple, kwargs_values) -> int | None:
    for values in (args, kwargs_values):
        for value in values:
            if isinstance(value, Tensor):
                if type(value) is Tensor:
                    device = value.device
                else:
                    # Attribute accesses of tensor subclasses such as ``BlksprsTensor`` dispatch through
                    # ``__torch_function__`` otherwise, which costs more than the guard itself.
                    with torch._C.DisableTorchFunctionSubclass():
                        device = value.device
                if device.type == "cuda":
                    return device.index
    return None
