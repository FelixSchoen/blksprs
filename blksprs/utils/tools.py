import tomllib
from pathlib import Path

import torch
from torch import Tensor, Size

# Capture scalar outputs for JIT compilation
torch._dynamo.config.capture_scalar_outputs = True


def version():
    with open(Path(__file__).parent.parent.parent.joinpath("pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]["version"]


def do_shape_blocksparse(x: Tensor):
    if x.dim() == 3:
        return x.contiguous(), x.size()

    return x.reshape(-1, x.size(-2), x.size(-1)).contiguous(), x.size()


def undo_shape_blocksparse(x: Tensor, shape: Size):
    if x.shape[:-2] == shape[:-2]:
        return x

    return x.reshape((*shape[:-2], *x.shape[-2:]))


def stride(x: Tensor):
    if x.dim() == 2:
        return x.size(1), 1
    elif x.dim() == 3:
        return x.size(1) * x.size(2), x.size(2), 1
    else:
        raise NotImplementedError
