import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x,
               x_stride_0, x_stride_1,
               out,
               TRITON_BLOCK_SIZE: tl.constexpr):
    blk_idx = ((tl.arange(0, TRITON_BLOCK_SIZE) * x_stride_0)[:, None] +
               (tl.arange(0, TRITON_BLOCK_SIZE) * x_stride_1)[None, :])
    blk = tl.load(x + blk_idx)

    buf = tl.reshape(tl.max(blk, axis=-1), (TRITON_BLOCK_SIZE, 1))

    blk_idx = ((tl.arange(0, TRITON_BLOCK_SIZE) * x_stride_0)[:, None] +
               (tl.arange(0, 1) * x_stride_1)[None, :])
    tl.atomic_max(out + blk_idx, buf)

def reproduce_max_error():
    subject = torch.tensor([[-0, torch.finfo(torch.float32).min],
                            [-1, -4]], dtype=torch.float, device='cuda').contiguous()
    subject = torch.where(subject == 0.0, torch.tensor(-0.0), subject)

    # Triton kernel
    out = torch.full_like(subject, fill_value=torch.finfo(subject.dtype).min)
    max_kernel[(1,)](subject, subject.size(1), 1, out, TRITON_BLOCK_SIZE=2)

    print(out)


reproduce_max_error()