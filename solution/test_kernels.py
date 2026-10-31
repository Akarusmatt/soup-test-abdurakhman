"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


def close(actual, expected, tol):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    close(out, ref, tol)
    close(gate.grad, ref_dgate, tol)
    close(up.grad, ref_dup, tol)


def reference_rms(hidden, weight, eps):
    """Qwen2RMSNorm.forward in fp32."""
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    return weight * (hidden.float() * torch.rsqrt(variance + eps))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows, cols", [(512, 1536), (7, 1536), (5, 300)])
def test_residual_norm_matches_pytorch(dtype, rows, cols):
    from solution.fused_norm import residual_norm

    torch.manual_seed(1)
    eps = 1e-6
    h = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
    add = torch.randn(rows, cols, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(cols, device="cuda", dtype=torch.float32)
    grad_h = torch.randn(rows, cols, device="cuda", dtype=dtype)
    grad_norm = torch.randn(rows, cols, device="cuda", dtype=dtype)

    ref_h = h.float() + add.float()
    ref_norm = reference_rms(ref_h, weight, eps)
    torch.autograd.backward((ref_h, ref_norm), (grad_h.float(), grad_norm.float()))
    ref_dh, ref_dadd = h.grad.float(), add.grad.float()
    h.grad = add.grad = None

    out_h, out_norm = residual_norm(h, add, weight, eps)
    torch.autograd.backward((out_h, out_norm), (grad_h, grad_norm))
    tol = 4 * TOLERANCE[dtype]
    close(out_h, ref_h, tol)
    close(out_norm, ref_norm, tol)
    close(h.grad, ref_dh, tol)
    close(add.grad, ref_dadd, tol)


def test_residual_norm_last_block_has_no_grad_on_h():
    from solution.fused_norm import residual_norm

    torch.manual_seed(2)
    eps = 1e-6
    h = torch.randn(9, 1536, device="cuda", requires_grad=True)
    add = torch.randn(9, 1536, device="cuda", requires_grad=True)
    weight = torch.rand(1536, device="cuda")
    grad_norm = torch.randn(9, 1536, device="cuda")

    reference_rms(h + add, weight, eps).backward(grad_norm)
    ref_dh = h.grad.clone()
    h.grad = None
    residual_norm(h, add, weight, eps)[1].backward(grad_norm)
    close(h.grad, ref_dh, 1e-4)


def test_first_norm_matches_pytorch():
    from solution.fused_norm import first_norm

    torch.manual_seed(3)
    h = torch.randn(31, 1536, device="cuda", dtype=torch.float16)
    weight = torch.rand(1536, device="cuda")
    close(first_norm(h, weight, 1e-6), reference_rms(h, weight, 1e-6), 2e-3)


def reference_rope(x, cos, sin):
    """transformers' apply_rotary_pos_emb for (rows, heads, head_dim)."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    cos = torch.cat((cos, cos), dim=-1)[:, None, :]
    sin = torch.cat((sin, sin), dim=-1)[:, None, :]
    return x * cos + rotated * sin


def reference_attention(q, k, v, cos, sin, scale, kv_heads, batch):
    """RoPE and scaled_dot_product_attention in fp32, heads shared the way Qwen2 shares them."""
    length, repeats = q.shape[0] // batch, 12 // kv_heads
    q = reference_rope(q.float().view(-1, 12, 128), cos.float().repeat(batch, 1),
                       sin.float().repeat(batch, 1)).view(batch, length, 12, 128).transpose(1, 2)
    k = reference_rope(k.float().view(-1, kv_heads, 128), cos.float().repeat(batch, 1),
                       sin.float().repeat(batch, 1)).view(batch, length, kv_heads, 128).transpose(1, 2)
    v = v.float().view(batch, length, kv_heads, 128).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k.repeat_interleave(repeats, 1),
                                         v.repeat_interleave(repeats, 1), is_causal=True, scale=scale)
    return out.transpose(1, 2).reshape(batch * length, -1)


@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("length, batch", [(512, 1), (300, 1), (9, 1), (2, 1), (128, 3), (40, 2), (7, 4)])
def test_attention_matches_pytorch(length, batch, kv_heads):
    from solution.fused_attention import causal_attention

    torch.manual_seed(4)
    half = {"dtype": torch.float16, "device": "cuda"}
    tokens = length * batch
    q = torch.randn(tokens, 12 * 128, **half, requires_grad=True)
    k = torch.randn(tokens, kv_heads * 128, **half, requires_grad=True)
    v = torch.randn(tokens, kv_heads * 128, **half, requires_grad=True)
    grad = torch.randn(tokens, 12 * 128, **half)
    angles = torch.rand(length, 64, **half) * 6.28
    cos, sin = angles.cos(), angles.sin()
    scale = 128 ** -0.5

    ref = reference_attention(q, k, v, cos, sin, scale, kv_heads, batch)
    ref.backward(grad.float())
    ref_grads = [t.grad.float() for t in (q, k, v)]
    q.grad = k.grad = v.grad = None

    out = causal_attention(q, k, v, cos, sin, scale, length)
    out.backward(grad)
    close(out, ref, 2e-2)
    for actual, expected in zip((q.grad, k.grad, v.grad), ref_grads):
        close(actual, expected, 3e-2)


def test_attention_scores_stay_in_fp16_range():
    """Magnitudes of the real model: q @ k.T passes 65504, the scaled scores do not."""
    from solution.fused_attention import causal_attention

    torch.manual_seed(8)
    half = {"dtype": torch.float16, "device": "cuda"}
    rows = 64
    q = torch.randn(rows, 12 * 128, **half) * 20
    k = q[:, :2 * 128] * 8
    v = torch.randn(rows, 2 * 128, **half)
    assert (q[:, :128].float() @ k[:, :128].float().t()).abs().max() > 65504
    angles = torch.rand(rows, 64, **half) * 6.28
    out = causal_attention(q, k, v, angles.cos(), angles.sin(), 128 ** -0.5, rows)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n_out", [1536, 512])
def test_lora_matches_pytorch(n_out):
    from solution.fused_lora import lora_linear

    torch.manual_seed(6)
    rows, n_in, rank = 40, 1536, 16
    half = {"dtype": torch.float16, "device": "cuda"}
    x = torch.randn(rows, n_in, **half, requires_grad=True)
    a = (torch.randn(rank, n_in, **half) * 0.02).requires_grad_()
    b = (torch.randn(rank, n_out, **half) * 0.02).requires_grad_()
    weight = torch.randn(n_out, n_in, **half) * 0.05
    bias = torch.randn(n_out, **half)
    grad = torch.randn(rows, n_out, **half)
    scaling = 2.0

    wide = [t.float() for t in (x, a, b, weight, bias)]
    ref = F.linear(wide[0], wide[3], wide[4]) + (wide[0] @ wide[1].t() @ wide[2]) * scaling
    ref.backward(grad.float())
    ref_grads = [t.grad.float() for t in (x, a, b)]
    x.grad = a.grad = b.grad = None

    out = lora_linear(x, a, b, weight, bias, scaling, 1.0, 12345)
    out.backward(grad)
    close(out, ref, 5e-3)
    for actual, expected in zip((x.grad, a.grad, b.grad), ref_grads):
        close(actual, expected, 2e-2)


def test_lora_dropout_keeps_the_rate_and_repeats_in_the_backward():
    from solution.fused_lora import lora_linear

    torch.manual_seed(7)
    rows, n_in, rank, keep = 512, 1536, 1, 0.95
    half = {"dtype": torch.float16, "device": "cuda"}
    x = torch.ones(rows, n_in, **half, requires_grad=True)
    weight = torch.zeros(n_in, n_in, **half)
    bias = torch.zeros(n_in, **half)
    # One component that sums the input and one output: each row of `out` counts what was kept.
    a = torch.ones(rank, n_in, **half, requires_grad=True)
    b = torch.ones(rank, n_in, **half, requires_grad=True)
    out = lora_linear(x, a, b, weight, bias, 1.0 / n_in, keep, 999)
    assert abs(out.float().mean().item() - 1.0) < 2e-3  # 1 / keep undoes the dropout on average
    out.sum().backward()
    kept = (x.grad.float() != 0).float().mean().item()
    assert abs(kept - keep) < 2e-3
    # Rows must not repeat the same mask, and neither must seeds.
    masks = (x.grad != 0)
    assert (masks[0] ^ masks[1]).any()
    x.grad = None
    lora_linear(x, a, b, weight, bias, 1.0 / n_in, keep, 1000).sum().backward()
    assert (masks[0] ^ (x.grad[0] != 0)).any()


def reference_head(hidden, targets, weight):
    """The head, the loss and the metrics the way transformers, TRL and Soup compute them."""
    hidden = hidden.clone().requires_grad_()
    logits = hidden @ weight.t()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    token_ce = F.nll_loss(log_probs, targets, ignore_index=-100, reduction="none")
    correct = (logits.argmax(-1) == targets) & (targets != -100)
    return hidden, token_ce, entropy, correct


@pytest.mark.parametrize("rows, first, last, vocab", [(300, 40, 299, 151936), (7, 0, 7, 5000),
                                                      (140, 130, 131, 4097), (9, 9, 9, 4096)])
def test_head_matches_pytorch(rows, first, last, vocab):
    from solution.fused_head import STATS, head_and_loss

    torch.manual_seed(5)
    hidden = torch.randn(rows, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(vocab, 64, device="cuda", dtype=torch.float16) * 0.3
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[:first] = targets[last:] = -100
    if last - first > 4:
        targets[first + 2] = -100  # a hole inside the run of answer tokens
    grad_scale = 100.0

    ref_hidden, token_ce, entropy, correct = reference_head(hidden, targets, weight)
    (token_ce.sum() * grad_scale).backward()

    grad = torch.zeros_like(hidden)
    stats = head_and_loss(hidden, targets, (first, last), weight, grad_scale, grad)
    assert stats.shape == (rows, STATS)
    close(stats[:, 1], entropy, 2e-3)
    close(stats[:, 2], token_ce, 2e-3)
    assert stats[:, 3].sum().item() == correct.sum().item()
    close(grad, ref_hidden.grad.float(), 5e-3 * max(ref_hidden.grad.abs().max().item(), 1.0))


def test_head_argmax_is_exact_on_the_last_block():
    from solution.fused_head import head_and_loss

    vocab = 4096 * 3 + 7
    hidden = torch.zeros(1, 2, device="cuda", dtype=torch.float16)
    hidden[0, 0] = 1.0
    weight = torch.zeros(vocab, 2, device="cuda", dtype=torch.float16)
    weight[vocab - 1, 0] = 5.0
    targets = torch.tensor([vocab - 1], device="cuda")
    stats = head_and_loss(hidden, targets, (0, 1), weight, 1.0, torch.zeros_like(hidden))
    assert stats[0, 3].item() == 1.0


def test_labelled_range():
    from solution.fused_head import labelled_range

    targets = torch.tensor([[-100, 1, 2, -100], [-100, -100, 5, 6], [-100, -100, -100, -100]])
    assert labelled_range(targets) == [(1, 3), (2, 4), (4, 4)]


@pytest.mark.parametrize("lengths", [[205, 79, 221, 379], [512, 512, 512, 512], [20, 21, 22, 23],
                                     [512, 3, 3, 3], [100, 130, 150, 400], [300], [90, 90]])
def test_plan_passes_covers_every_row_within_the_limit(lengths):
    from solution.step import plan_passes

    passes = plan_passes(lengths, 512)
    assert sorted(row for group in passes for row in group) == list(range(len(lengths)))
    for group in passes:
        assert len(group) * max(lengths[row] for row in group) <= 512


def test_plan_passes_shares_a_pass_between_short_rows():
    from solution.step import plan_passes

    assert len(plan_passes([20, 21, 22, 23], 512)) == 1
    assert len(plan_passes([512, 512, 512, 512], 512)) == 4
    assert len(plan_passes([512, 3, 3, 3], 512)) == 2


def test_step_without_labels_leaves_zero_gradients():
    """All rows of a batch cut off before the answer: no backward runs, and the optimizer still needs
    a gradient on every parameter."""
    from solution.stack import FusedStack, LoraPath

    lora_a = torch.nn.Parameter(torch.ones(16, 1536, device="cuda"))
    lora_b = torch.nn.Parameter(torch.ones(256, 16, device="cuda"))
    stack = FusedStack.__new__(FusedStack)
    stack.lora = [LoraPath(None, None, lora_a, lora_b, 2.0, 0.95)]
    stack.begin_step()
    stack.end_step()
    assert lora_a.grad.shape == lora_a.shape and lora_b.grad.shape == lora_b.shape
    assert lora_a.grad.abs().sum() == 0 and lora_b.grad.abs().sum() == 0
    assert lora_a.grad.dtype == torch.float32
