import torch
from torch import Tensor


class BlksprsTensor(Tensor):
    """A wrapper class representing a block-sparse tensor in compressed form.
    """

    def __repr__(self):
        return f"BlksprsTensor({torch.Tensor(self).__repr__()})"