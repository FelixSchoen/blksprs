import torch


def dbg_set_print_options():
    torch.set_printoptions(edgeitems=99999, linewidth=99999)


def dbg_reset_print_options():
    torch.set_printoptions(edgeitems=3, linewidth=80)


def dbg_tensor_full(tensor: torch.Tensor):
    dbg_set_print_options()
    torch_representation = str(tensor)
    dbg_reset_print_options()
    return torch_representation


def dbg_tensor_inv(tensor: torch.Tensor):
    return dbg_tensor_nan(tensor) or dbg_tensor_inf(tensor)


def dbg_tensor_nan(tensor: torch.Tensor):
    return torch.isnan(tensor).any().item()


def dbg_tensor_inf(tensor: torch.Tensor):
    return torch.isinf(tensor).any().item()