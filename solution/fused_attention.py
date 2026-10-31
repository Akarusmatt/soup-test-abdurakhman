"""Causal self-attention of one pass: RoPE, softmax and the head layouts in fused kernels.

The matmuls stay with cuBLAS. Everything around them is what makes attention expensive on top of them:
ten launches of RoPE, repeating k and v for the shared key/value heads, building and applying the
causal mask, the softmax, and the transposes between token order and head order, each with its
backward, and with the logits and both repeated tensors kept alive for it.

Here one launch applies RoPE and moves q, k and v from token order to head order. The 12 query heads
share 2 key/value heads, so q is laid out as (groups * rows, repeats * length, head_dim) and the matmul
with k needs no repeated k at all. A second launch turns the scores into masked probabilities. In the
backward the softmax needs only the probabilities, so the scores are never saved.

A pass holds `rows` right-padded rows of `length` tokens. The causal mask is all the mask they need:
a real token only sees earlier positions, which are real, and the outputs of padding are never read.

Scores are the one place in the stack that stays in fp32. k is multiplied by the attention scale
before the matmul, like in PyTorch's own attention, because the unscaled scores of this model pass
the range of fp16; even scaled they reach 13,000, where fp16 steps by 8, and they feed an exponential.
Measured against a full fp32 reference on real weights, this took the error of the LoRA gradient from
4.6% to 2.2%, next to 1.6% for base Soup. The matmul is under 1% of the flops of a layer.

    q, k, v: (rows * length, heads * head_dim) -> out: (rows * length, heads * head_dim)
"""

import torch
import triton
import triton.language as tl

Q_BLOCK = 2048     # q: 12 heads of 128
KV_BLOCK = 512     # k and v: 2 heads of 128 here, up to 4 fit
ROW_BLOCK = 512    # the longest row: data.max_length
HEAD_DIM = 128     # a power of two, so the offsets below are bit operations
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit(do_not_specialize=("length", "rows"))
def attention_split_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr, v_out_ptr,
                           length, rows, repeats, q_cols, kv_cols, scale, Q_BLOCK: tl.constexpr,
                           KV_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    """RoPE of q and k, and head order for q, k and v, for one token per program."""
    token = tl.program_id(0)
    position = token % length
    row = token // length
    half = HEAD_DIM // 2
    table = position * half

    cols = tl.arange(0, Q_BLOCK)
    mask = cols < q_cols
    dim = cols & (HEAD_DIM - 1)
    head = cols // HEAD_DIM
    low = dim < half
    cos = tl.load(cos_ptr + table + (cols & (half - 1))).to(tl.float32)
    sin = tl.load(sin_ptr + table + (cols & (half - 1))).to(tl.float32)
    x = tl.load(q_ptr + token * q_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(q_ptr + token * q_cols + (cols ^ half), mask=mask, other=0.0).to(tl.float32)
    out = x * cos + tl.where(low, -y, y) * sin
    line = (((head // repeats) * rows + row) * repeats + head % repeats) * length + position
    tl.store(q_out_ptr + line * HEAD_DIM + dim, out.to(q_out_ptr.dtype.element_ty), mask=mask)

    cols = tl.arange(0, KV_BLOCK)
    mask = cols < kv_cols
    dim = cols & (HEAD_DIM - 1)
    head = cols // HEAD_DIM
    low = dim < half
    cos = tl.load(cos_ptr + table + (cols & (half - 1))).to(tl.float32)
    sin = tl.load(sin_ptr + table + (cols & (half - 1))).to(tl.float32)
    x = tl.load(k_ptr + token * kv_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(k_ptr + token * kv_cols + (cols ^ half), mask=mask, other=0.0).to(tl.float32)
    out = (x * cos + tl.where(low, -y, y) * sin) * scale
    line = ((head * rows + row) * length + position) * HEAD_DIM + dim
    tl.store(k_out_ptr + line, out.to(k_out_ptr.dtype.element_ty), mask=mask)

    value = tl.load(v_ptr + token * kv_cols + cols, mask=mask, other=0.0)
    tl.store(v_out_ptr + line, value, mask=mask)


@triton.jit(do_not_specialize=("length", "rows"))
def attention_merge_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr, v_out_ptr,
                           length, rows, repeats, q_cols, kv_cols, scale, Q_BLOCK: tl.constexpr,
                           KV_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
    """The backward of attention_split_kernel: token order again, RoPE turned the other way."""
    token = tl.program_id(0)
    position = token % length
    row = token // length
    half = HEAD_DIM // 2
    table = position * half

    cols = tl.arange(0, Q_BLOCK)
    mask = cols < q_cols
    dim = cols & (HEAD_DIM - 1)
    head = cols // HEAD_DIM
    low = dim < half
    cos = tl.load(cos_ptr + table + (cols & (half - 1))).to(tl.float32)
    sin = tl.load(sin_ptr + table + (cols & (half - 1))).to(tl.float32)
    line = ((((head // repeats) * rows + row) * repeats + head % repeats) * length + position) * HEAD_DIM
    x = tl.load(q_ptr + line + dim, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(q_ptr + line + (dim ^ half), mask=mask, other=0.0).to(tl.float32)
    out = x * cos + tl.where(low, y, -y) * sin
    tl.store(q_out_ptr + token * q_cols + cols, out.to(q_out_ptr.dtype.element_ty), mask=mask)

    cols = tl.arange(0, KV_BLOCK)
    mask = cols < kv_cols
    dim = cols & (HEAD_DIM - 1)
    head = cols // HEAD_DIM
    low = dim < half
    cos = tl.load(cos_ptr + table + (cols & (half - 1))).to(tl.float32)
    sin = tl.load(sin_ptr + table + (cols & (half - 1))).to(tl.float32)
    line = ((head * rows + row) * length + position) * HEAD_DIM
    x = tl.load(k_ptr + line + dim, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(k_ptr + line + (dim ^ half), mask=mask, other=0.0).to(tl.float32)
    out = (x * cos + tl.where(low, y, -y) * sin) * scale
    tl.store(k_out_ptr + token * kv_cols + cols, out.to(k_out_ptr.dtype.element_ty), mask=mask)

    value = tl.load(v_ptr + line + dim, mask=mask, other=0.0)
    tl.store(v_out_ptr + token * kv_cols + cols, value, mask=mask)


@triton.jit(do_not_specialize=("length",))
def causal_softmax_fwd_kernel(scores_ptr, probs_ptr, length, ROW_BLOCK: tl.constexpr):
    """scores -> probabilities. A line of the score matrix belongs to the token at position
    `index % length` of its row and sees the positions up to and including that one."""
    index = tl.program_id(0)
    cols = tl.arange(0, ROW_BLOCK)
    mask = cols < length
    visible = cols <= index % length
    start = index * length
    scores = tl.load(scores_ptr + start + cols, mask=mask, other=0.0).to(tl.float32)
    scores = tl.where(visible, scores, NEG_INF)
    shifted = scores - tl.max(scores, axis=0)
    probs = tl.where(visible, tl.exp(shifted), 0.0)
    probs = probs / tl.sum(probs, axis=0)
    tl.store(probs_ptr + start + cols, probs.to(probs_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=("length",))
def causal_softmax_bwd_kernel(grad_ptr, probs_ptr, length, ROW_BLOCK: tl.constexpr):
    """d(probabilities)/d(scores) transposed, in place: p * (g - sum(p * g))."""
    index = tl.program_id(0)
    cols = tl.arange(0, ROW_BLOCK)
    mask = cols < length
    grad = tl.load(grad_ptr + index * length + cols, mask=mask, other=0.0).to(tl.float32)
    probs = tl.load(probs_ptr + index * length + cols, mask=mask, other=0.0).to(tl.float32)
    out = probs * (grad - tl.sum(probs * grad, axis=0))
    tl.store(grad_ptr + index * length + cols, out.to(grad_ptr.dtype.element_ty), mask=mask)


class CausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cos, sin, scale, length):
        tokens, q_cols = q.shape
        kv_cols = k.shape[1]
        rows, groups, repeats = tokens // length, kv_cols // HEAD_DIM, q_cols // kv_cols
        lines = (groups * rows, repeats * length, HEAD_DIM)
        q_heads = torch.empty(lines, dtype=q.dtype, device=q.device)
        k_heads = torch.empty(groups * rows, length, HEAD_DIM, dtype=q.dtype, device=q.device)
        v_heads = torch.empty_like(k_heads)
        attention_split_kernel[(tokens,)](q, k, v, cos, sin, q_heads, k_heads, v_heads, length, rows,
                                          repeats, q_cols, kv_cols, scale, Q_BLOCK=Q_BLOCK,
                                          KV_BLOCK=KV_BLOCK, HEAD_DIM=HEAD_DIM)
        scores = q_heads.float() @ k_heads.float().transpose(1, 2)
        probs = torch.empty_like(scores, dtype=q.dtype)
        causal_softmax_fwd_kernel[(lines[0] * lines[1],)](scores, probs, length, ROW_BLOCK=ROW_BLOCK)
        del scores
        out = probs @ v_heads
        ctx.save_for_backward(q_heads, k_heads, v_heads, probs)
        ctx.cos, ctx.sin, ctx.scale, ctx.length = cos, sin, scale, length
        ctx.rows, ctx.groups, ctx.repeats = rows, groups, repeats
        return out.view(groups, rows, repeats, length, HEAD_DIM).permute(1, 3, 0, 2, 4).reshape(
            tokens, -1)

    @staticmethod
    def backward(ctx, grad):
        q_heads, k_heads, v_heads, probs = ctx.saved_tensors
        length, rows, groups, repeats = ctx.length, ctx.rows, ctx.groups, ctx.repeats
        grad = grad.view(rows, length, groups, repeats, HEAD_DIM).permute(2, 0, 3, 1, 4).reshape(
            groups * rows, repeats * length, HEAD_DIM)
        dv = probs.transpose(1, 2) @ grad
        scores = grad @ v_heads.transpose(1, 2)
        causal_softmax_bwd_kernel[(scores.shape[0] * scores.shape[1],)](scores, probs, length,
                                                                        ROW_BLOCK=ROW_BLOCK)
        dq = scores @ k_heads
        dk = scores.transpose(1, 2) @ q_heads
        tokens = rows * length
        kv_cols = groups * HEAD_DIM
        q_cols = repeats * kv_cols
        out_q = torch.empty(tokens, q_cols, dtype=dq.dtype, device=dq.device)
        out_k = torch.empty(tokens, kv_cols, dtype=dq.dtype, device=dq.device)
        out_v = torch.empty_like(out_k)
        attention_merge_kernel[(tokens,)](dq, dk, dv, ctx.cos, ctx.sin, out_q, out_k, out_v, length,
                                          rows, repeats, q_cols, kv_cols, ctx.scale, Q_BLOCK=Q_BLOCK,
                                          KV_BLOCK=KV_BLOCK, HEAD_DIM=HEAD_DIM)
        return out_q, out_k, out_v, None, None, None, None


def causal_attention(q, k, v, cos, sin, scale, length):
    return CausalAttention.apply(q, k, v, cos, sin, scale, length)
