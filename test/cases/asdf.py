import torch


def test_transpose():
    size = 8
    tensor = torch.arange(0,size*size).reshape(size, size)

    print()
    print(tensor)
    print()
    print(tensor.transpose(0,1))