"""Residual add + RMSNorm in one kernel, forward and backward.

Every RMSNorm in Qwen2 reads a residual sum: `h = h + x`, then `n = norm(h)`. In plain PyTorch that
is an add, a cast to fp32, pow, mean, add, rsqrt, mul, a cast back and a mul by the norm weight, and
as many launches again in the backward. Here one launch writes both outputs, and the backward of
both goes through one launch as well: it only needs `h` and the frozen norm weight.

The arithmetic runs in fp32 inside the kernel, like Qwen2RMSNorm, and is stored in the input dtype.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048  # the smallest power of two that covers hidden_size = 1536


@triton.jit
def residual_norm_fwd_kernel(h_ptr, add_ptr, weight_ptr, h_out_ptr, norm_out_ptr, n_cols, add_scale,
                             eps, BLOCK: tl.constexpr):
    row = tl.program_id(0) * n_cols
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    h = tl.load(h_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    add = tl.load(add_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = h + add * add_scale
    inv_rms = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / n_cols + eps)
    tl.store(h_out_ptr + row + cols, h.to(h_out_ptr.dtype.element_ty), mask=mask)
    tl.store(norm_out_ptr + row + cols, (weight * h * inv_rms).to(norm_out_ptr.dtype.element_ty),
             mask=mask)


@triton.jit
def residual_norm_bwd_kernel(grad_h_ptr, grad_norm_ptr, h_ptr, weight_ptr, out_ptr, n_cols,
                             grad_h_scale, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0) * n_cols
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    grad_h = tl.load(grad_h_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    grad_norm = tl.load(grad_norm_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    h = tl.load(h_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / n_cols + eps)
    weighted = grad_norm * weight
    # d(w h r)/dh with r = (mean(h^2) + eps)^(-1/2): the diagonal term minus the rank-one term.
    rank_one = tl.sum(weighted * h, axis=0) * inv_rms * inv_rms * inv_rms / n_cols
    out = grad_h * grad_h_scale + weighted * inv_rms - h * rank_one
    tl.store(out_ptr + row + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


class ResidualNorm(torch.autograd.Function):
    """(h, add) -> (h + add, norm(h + add)) for tensors of shape (rows, hidden_size)."""

    @staticmethod
    def forward(ctx, h, add, weight, eps):
        h_out, norm = torch.empty_like(h), torch.empty_like(h)
        rows, cols = h.shape
        residual_norm_fwd_kernel[(rows,)](h, add, weight, h_out, norm, cols, 1.0, eps, BLOCK=BLOCK)
        ctx.save_for_backward(h_out)
        ctx.weight, ctx.eps = weight, eps
        return h_out, norm

    @staticmethod
    def backward(ctx, grad_h, grad_norm):
        h = ctx.saved_tensors[0]
        out = torch.empty_like(h)
        rows, cols = h.shape
        # The last block of the model has no use for `h`, so autograd hands in None for it.
        scale = 1.0 if grad_h is not None else 0.0
        grad_norm = grad_norm.contiguous()
        grad_h = grad_norm if grad_h is None else grad_h.contiguous()
        residual_norm_bwd_kernel[(rows,)](grad_h, grad_norm, h, ctx.weight, out, cols, scale,
                                          ctx.eps, BLOCK=BLOCK)
        return out, out, None, None


def residual_norm(h, add, weight, eps):
    return ResidualNorm.apply(h, add, weight, eps)


def first_norm(h, weight, eps):
    """The norm of the embeddings: no residual and nothing to differentiate."""
    norm = torch.empty_like(h)
    rows, cols = h.shape
    residual_norm_fwd_kernel[(rows,)](h, h, weight, h, norm, cols, 0.0, eps, BLOCK=BLOCK)
    return norm
