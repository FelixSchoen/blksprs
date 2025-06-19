from blksprs.utils.debugging import dbg_tensor_full
from cases.test_blocksparse import _get_blocksparse_layout, _blocksparse_roundtrip, _get_autocast_min_val
from test_blocksparse import *


def test_softmax_single():
    b, m, n, k, sparsity_block_size, sparsity_percentage = (2, 32, 32, 32, 16, 0.5)

    x_d = torch.randn(size=(b, m, k), device=DEVICE)
    sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
    x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                  fill_value=_get_autocast_min_val())

    for x, sparsity_layout_x in [
        # (x_d, sparsity_layout_x_d),
        (x_bs, sparsity_layout_x_bs)]:
        x_stock = x.clone().requires_grad_(True)
        x_blksprs = x.clone().requires_grad_(True)

        stock_softmax_out = torch.softmax(x_stock, dim=-1)
        stock_dtype = stock_softmax_out.dtype

        blksprs_softmax_out = bs.ops.softmax_fused(
            bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
            sparsity_layout_x, sparsity_block_size)
        blksprs_softmax_dense_out = bs.ops.to_dense(blksprs_softmax_out, sparsity_layout_x,
                                                    sparsity_block_size)

        asdf = dbg_tensor_full(blksprs_softmax_dense_out.to(stock_dtype))
        bsdf = dbg_tensor_full(stock_softmax_out)

        assert torch.allclose(blksprs_softmax_dense_out.to(stock_dtype), stock_softmax_out, atol=ATOL,
                              rtol=RTOL)
        #
        # target = torch.randn_like(stock_softmax_out)
        # stock_loss = torch.nn.L1Loss()
        # blksprs_loss = torch.nn.L1Loss()
        # stock_loss = stock_loss(stock_softmax_out, target)
        # blksprs_loss = blksprs_loss(blksprs_softmax_dense_out, target)
        #
        # stock_loss.backward()
        # blksprs_loss.backward()
        #
        # assert torch.allclose(x_blksprs.grad, x_stock.grad, atol=ATOL, rtol=RTOL)


# @pytest.mark.parametrize("config", TEST_CONFIGURATIONS)
@pytest.mark.parametrize("config", [(2, 32, 32, 32, 16, 0.5)])
def test_blksprs_softmax(config: list):
        b, m, n, k, sparsity_block_size, sparsity_percentage = config

        x_d = torch.randn(size=(b, m, k), device=DEVICE)
        sparsity_layout_x_d = torch.ones(size=(b, m // sparsity_block_size, k // sparsity_block_size),
                                         device=DEVICE)

        sparsity_layout_x_bs = _get_blocksparse_layout(b, m, k, sparsity_block_size, sparsity_percentage)
        x_bs = _blocksparse_roundtrip(x_d, sparsity_layout_x_bs, sparsity_block_size,
                                      fill_value=_get_autocast_min_val())

        for x, sparsity_layout_x in [(x_d, sparsity_layout_x_d), (x_bs, sparsity_layout_x_bs)]:
            x_stock = x.clone().requires_grad_(True)
            x_blksprs = x.clone().requires_grad_(True)

            stock_softmax_out = _blocksparse_roundtrip(torch.softmax(x_stock, dim=-1), sparsity_layout_x, sparsity_block_size)
            stock_dtype = stock_softmax_out.dtype

            blksprs_softmax_out = bs.ops.softmax_fused(
                bs.ops.to_sparse(x_blksprs, sparsity_layout_x, sparsity_block_size),
                sparsity_layout_x, sparsity_block_size)
            blksprs_softmax_dense_out = bs.ops.to_dense(blksprs_softmax_out, sparsity_layout_x,
                                                        sparsity_block_size)

            assert torch.allclose(blksprs_softmax_dense_out.to(stock_dtype), stock_softmax_out, atol=ATOL,
                                  rtol=RTOL)