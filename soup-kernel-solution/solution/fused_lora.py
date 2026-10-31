"""A LoRA path of the frozen base: base + bias + scaling * B(A(dropout(x))) in one kernel.

In peft this is a clone, three launches of dropout, two small matmuls, a mul, an add and the casts
autocast adds around `lora_A` and `lora_B`, and about as many in the backward, for each of the 56
wrapped projections, in each pass of a step. It is also what keeps a dropout noise tensor and the
dropped input alive until the backward.

Here the matmul with the frozen weight stays with cuBLAS, and everything else goes through one launch
each way. The dropout has the same distribution as `nn.Dropout` (each element kept with probability
1 - p and divided by it), but its mask is a hash of the position of the element and a seed drawn per
call, so the backward regenerates it instead of keeping it.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048  # covers hidden_size = 1536


@triton.jit(do_not_specialize=("seed",))
def lora_fwd_kernel(x_ptr, base_ptr, bias_ptr, a_ptr, b_ptr, out_ptr, low_ptr, n_in, n_out, rank,
                    scaling, keep_prob, seed, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_mask = cols < n_in
    out_mask = cols < n_out

    bits = (token * n_in + cols + seed).to(tl.uint32)
    bits = ((bits >> 16) ^ bits) * 0x45d9f3b
    bits = ((bits >> 16) ^ bits) * 0x45d9f3b
    bits = (bits >> 16) ^ bits
    keep = (bits >> 8).to(tl.float32) * 5.9604644775390625e-08 < keep_prob

    x = tl.load(x_ptr + token * n_in + cols, mask=in_mask, other=0.0).to(tl.float32)
    dropped = tl.where(keep, x / keep_prob, 0.0)
    total = tl.zeros((BLOCK,), tl.float32)
    for r in range(rank):
        row = tl.load(a_ptr + r * n_in + cols, mask=in_mask, other=0.0).to(tl.float32)
        low = tl.sum(row * dropped, axis=0) * scaling
        tl.store(low_ptr + token * rank + r, low.to(low_ptr.dtype.element_ty))
        column = tl.load(b_ptr + r * n_out + cols, mask=out_mask, other=0.0).to(tl.float32)
        total += column * low

    base = tl.load(base_ptr + token * n_out + cols, mask=out_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=out_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * n_out + cols, (base + bias + total).to(out_ptr.dtype.element_ty),
             mask=out_mask)


@triton.jit(do_not_specialize=("seed",))
def lora_bwd_kernel(grad_ptr, x_ptr, a_ptr, b_ptr, base_ptr, out_ptr, low_grad_ptr, dropped_ptr,
                    n_in, n_out, rank, scaling, keep_prob, seed, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_mask = cols < n_in
    out_mask = cols < n_out

    bits = (token * n_in + cols + seed).to(tl.uint32)
    bits = ((bits >> 16) ^ bits) * 0x45d9f3b
    bits = ((bits >> 16) ^ bits) * 0x45d9f3b
    bits = (bits >> 16) ^ bits
    keep = (bits >> 8).to(tl.float32) * 5.9604644775390625e-08 < keep_prob

    grad = tl.load(grad_ptr + token * n_out + cols, mask=out_mask, other=0.0).to(tl.float32)
    total = tl.zeros((BLOCK,), tl.float32)
    for r in range(rank):
        column = tl.load(b_ptr + r * n_out + cols, mask=out_mask, other=0.0).to(tl.float32)
        low = tl.sum(column * grad, axis=0) * scaling
        tl.store(low_grad_ptr + token * rank + r, low.to(low_grad_ptr.dtype.element_ty))
        row = tl.load(a_ptr + r * n_in + cols, mask=in_mask, other=0.0).to(tl.float32)
        total += row * low

    x = tl.load(x_ptr + token * n_in + cols, mask=in_mask, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + token * n_in + cols, mask=in_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * n_in + cols,
             (base + tl.where(keep, total / keep_prob, 0.0)).to(out_ptr.dtype.element_ty),
             mask=in_mask)
    tl.store(dropped_ptr + token * n_in + cols,
             tl.where(keep, x / keep_prob, 0.0).to(dropped_ptr.dtype.element_ty), mask=in_mask)


class LoraLinear(torch.autograd.Function):
    """(rows, n_in) -> (rows, n_out). `a` is (rank, n_in) and `b` is (rank, n_out), the transpose of
    lora_B, so that the kernel reads both as contiguous rows."""

    @staticmethod
    def forward(ctx, x, a, b, weight, bias, scaling, keep_prob, seed):
        rows, n_in = x.shape
        rank, n_out = b.shape
        base = x @ weight.t()
        out = torch.empty_like(base)
        low = torch.empty(rows, rank, dtype=x.dtype, device=x.device)
        lora_fwd_kernel[(rows,)](x, base, bias, a, b, out, low, n_in, n_out, rank, scaling, keep_prob,
                                 seed, BLOCK=BLOCK)
        ctx.save_for_backward(x, a, b, low)
        ctx.weight, ctx.scaling, ctx.keep_prob, ctx.seed = weight, scaling, keep_prob, seed
        return out

    @staticmethod
    def backward(ctx, grad):
        x, a, b, low = ctx.saved_tensors
        grad = grad.contiguous()
        rows, n_in = x.shape
        rank, n_out = b.shape
        base = grad @ ctx.weight
        dx = torch.empty_like(base)
        low_grad = torch.empty(rows, rank, dtype=x.dtype, device=x.device)
        dropped = torch.empty_like(base)
        lora_bwd_kernel[(rows,)](grad, x, a, b, base, dx, low_grad, dropped, n_in, n_out, rank,
                                 ctx.scaling, ctx.keep_prob, ctx.seed, BLOCK=BLOCK)
        return dx, low_grad.t() @ dropped, low.t() @ grad, None, None, None, None, None


def lora_linear(x, a, b, weight, bias, scaling, keep_prob, seed):
    return LoraLinear.apply(x, a, b, weight, bias, scaling, keep_prob, seed)
