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
