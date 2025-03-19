import time

import torch

from blksprs.layouting.sparsity_layout import build_sparsity_layout_matmul_fast
from blksprs.ops.conversion import to_dense, to_sparse
from blksprs.ops.matmul import matmul
from cases.test_blocksparse import _get_blocksparse_layout, _blocksparse_roundtrip, DEVICE, ATOL, RTOL

torch._dynamo.config.capture_scalar_outputs = True

num_call = 1000
log_frequency = 10
configuration = (2, 2048, 2048, 128, 32, 32, 0.75)
b, m, n, k, sparsity_block_size, triton_block_size, sparsity_percentage = configuration

x_d = torch.randn(size=(b, m, k), device=DEVICE)
sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size), device=DEVICE)
y_d = torch.randn(size=(b, n, k), device=DEVICE).transpose(-1, -2).contiguous()
sparsity_layout_y_d = torch.ones(size=(b, k // sparsity_block_size, n // sparsity_block_size), device=DEVICE)

sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size, triton_block_size)
sparsity_layout_y_bs = _get_blocksparse_layout(b, k, n, sparsity_block_size, sparsity_percentage)
y_bs = _blocksparse_roundtrip(y_d, sparsity_layout_y_bs, sparsity_block_size, triton_block_size)

sparsity_layout_o_d = torch.ones(size=(b, m // sparsity_block_size, n // sparsity_block_size), device=DEVICE)
sparsity_layout_o_bs = build_sparsity_layout_matmul_fast(sparsity_layout_x_bs,
                                                         sparsity_layout_y_bs)

x, sparsity_layout_x, y, sparsity_layout_y, sparsity_layout_o = (
x_bs, sparsity_layout_x_bs, y_bs, sparsity_layout_y_bs, sparsity_layout_o_d)

start_time = time.perf_counter()
start_compile_time = None

for i in range(num_call):
    if i % log_frequency == 0:
        print(f"Iteration {i}")
    if i == 9:
        start_compile_time = time.perf_counter()

    x_stock = x.clone().requires_grad_(True)
    y_stock = y.clone().requires_grad_(True)
    x_blksprs = x.clone().requires_grad_(True)
    y_blksprs = y.clone().requires_grad_(True)

    stock_matmul_out = torch.matmul(x_stock, y_stock)
    blksprs_matmul_out = torch.compile(matmul)(to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                                               sparsity_layout_x,
                                               to_sparse(y_blksprs, sparsity_layout_y, sparsity_block_size),
                                               sparsity_layout_y,
                                               sparsity_layout_o, sparsity_block_size)
    blksprs_matmul_dense_out = to_dense(blksprs_matmul_out, sparsity_layout_o, sparsity_block_size)

    assert torch.allclose(blksprs_matmul_dense_out, stock_matmul_out, atol=ATOL, rtol=RTOL)

    target = torch.randn_like(stock_matmul_out)
    stock_loss = torch.nn.L1Loss()
    blksprs_loss = torch.nn.L1Loss()
    stock_loss = stock_loss(stock_matmul_out, target)
    blksprs_loss = blksprs_loss(blksprs_matmul_dense_out, target)

    stock_loss.backward()
    blksprs_loss.backward()

    assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(y_blksprs.grad, y_stock.grad, atol=ATOL, rtol=RTOL)

end_time = time.perf_counter()

print(f"Time taken for {num_call} calls: {end_time - start_time:.2f} seconds")
print(f"Time taken for compilation: {start_compile_time - start_time:.2f} seconds")
print(f"Time taken after compilation: {end_time - start_compile_time:.2f} seconds")
