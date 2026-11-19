import os
import random

import numpy as np
import pytest
import torch

os.environ.setdefault("BLKSPRS_AUTOTUNE", "TEST")


def pytest_addoption(parser):
    parser.addoption("--seed", type=int, default=0, help="Seed used by the test suite")


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    seed = request.config.getoption("--seed")
    seed_32 = seed % (2 ** 32)

    random.seed(seed)
    np.random.seed(seed_32)
    torch.manual_seed(seed)
    torch.set_printoptions(edgeitems=64, linewidth=10000)

    normal_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = lambda self, *args, **kwargs: f"{self.shape}, {self.dtype}:\n{normal_repr(self)}"

    print("Seed:", seed)
    yield
    print("Seed:", seed)

    torch.Tensor.__repr__ = normal_repr


def pytest_collection_modifyitems(config, items):
    """Skips the tests of CUDA-only modules and tests marked ``cuda`` on machines without a CUDA device.

    A module is CUDA-only if its ``DEVICE`` is a CUDA device; the other tests, such as those of the layout builders,
    also run on the CPU.

    """
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires a CUDA device")
    for item in items:
        device = getattr(item.module, "DEVICE", None)
        if (isinstance(device, torch.device) and device.type == "cuda") or item.get_closest_marker("cuda"):
            item.add_marker(skip)
