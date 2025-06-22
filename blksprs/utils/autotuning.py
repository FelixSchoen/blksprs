import os

blksprs_autotune_mode = os.getenv("BLKSPRS_AUTOTUNE", "DEFAULT")

if blksprs_autotune_mode == "DEFAULT":
    autotune_parameters = [
        (16, 3, 8),
        (16, 4, 4),
        (16, 5, 2),

        (32, 3, 8),
        (32, 4, 4),
        (32, 5, 2),

        (64, 3, 8),
        (64, 4, 4),
        (64, 5, 2),

        (128, 3, 8),
        (128, 4, 4),
        (128, 5, 2),
    ]
elif blksprs_autotune_mode == "TEST":
    autotune_parameters = [
        (16, 3, 8),

        (32, 3, 8),

        (64, 3, 8),
    ]
else:
    raise NotImplementedError(f"Unknown autotune mode: {blksprs_autotune_mode}")

import torch
import triton


def prune_autotune_configs(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size = kernel_args["sparsity_block_size"]

    pruned_configs = []

    for config in autotune_configs:
        if config.kwargs["TRITON_BLOCK_SIZE"] <= sparsity_block_size:
            pruned_configs.append(config)

    assert len(pruned_configs) > 0, f"No valid autotune configs found for sparsity block size {sparsity_block_size}"

    return pruned_configs


def prune_autotune_configs_conversion(autotune_configs, kernel_args, **kwargs):
    sparsity_block_size_from = kernel_args["sparsity_block_size_from"]
    sparsity_block_size_to = kernel_args["sparsity_block_size_to"]
    sparsity_block_size = min(sparsity_block_size_from, sparsity_block_size_to)

    pruned_configs = []

    for config in autotune_configs:
        if config.kwargs["TRITON_BLOCK_SIZE"] <= sparsity_block_size:
            pruned_configs.append(config)

    assert len(pruned_configs) > 0, f"No valid autotune configs found for sparsity block size {sparsity_block_size}"

    return pruned_configs


@torch.compile
def get_autotune_configs():
    global autotune_parameters

    autotune_configs = []

    for block_size, num_stages, num_warps in autotune_parameters:
        autotune_configs.append(
            triton.Config({"TRITON_BLOCK_SIZE": block_size}, num_stages=num_stages, num_warps=num_warps))

    return autotune_configs