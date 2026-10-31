"""The output head and everything computed from the logits, chunk by chunk, in one kernel.

Base Soup builds [batch, seq, 151936] logits, upcasts them for the loss, keeps the log-probabilities
for the backward, upcasts them again for the trainer's outputs, and runs softmax over them three more
times for the loss, TRL's entropy and Soup's per-row losses. That is about 4 GB on a T4 and every
softmax over the vocabulary in the step.

The logits of this module live for one chunk of positions. One launch of `head_stats_kernel` gives
the cross-entropy, the entropy and the argmax of every position of the chunk in one pass over the
vocabulary, using the normalizer of each row from the same pass (the "online softmax" idea of
Milakov & Gimelshein, https://arxiv.org/abs/1805.02867, also used by FlashAttention), and it leaves
the unnormalized probabilities in the logits buffer. The gradient on the hidden states then follows
with one matmul by the frozen head: g = (p - onehot) @ W. The chunk is dropped, and the backward of
the model starts from the gradient, which is the only thing that outlives the chunk.

The numbers are the ones TRL and Soup compute from the same fp16 logits: this is the same formula in
fp32 inside the kernel, not an approximation.
"""

import torch
import triton
import triton.language as tl

BLOCK = 4096
CHUNK = 128        # positions per chunk: 128 * 151936 * 2 B = 39 MB
PROB_SCALE = 256.0  # keeps small probabilities out of the fp16 underflow range
IGNORE_INDEX = -100
STATS = 4          # per position: normalizer, entropy, cross-entropy, argmax == target
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit(do_not_specialize_on_alignment=("target_ptr",))
def head_stats_kernel(logits_ptr, target_ptr, stats_ptr, n_vocab, prob_scale, BLOCK: tl.constexpr):
    pos = tl.program_id(0)
    row = logits_ptr + pos * n_vocab
    offsets = tl.arange(0, BLOCK)
    target = tl.load(target_ptr + pos)
    labelled = target >= 0
    at_target = tl.load(row + tl.where(labelled, target, 0)).to(tl.float32)

    top = tl.full((), NEG_INF, tl.float32)
    argmax = tl.full((), 0, tl.int64)
    for start in range(0, n_vocab, BLOCK):
        mask = start + offsets < n_vocab
        logits = tl.load(row + start + offsets, mask=mask, other=NEG_INF).to(tl.float32)
        block_top = tl.max(logits, axis=0)
        block_arg = tl.argmax(logits, axis=0).to(tl.int64) + start
        argmax = tl.where(block_top > top, block_arg, argmax)
        top = tl.maximum(top, block_top)

    # With e = exp(logit - top): normalizer = sum(e), entropy = log(sum(e)) - sum(e * logit) / sum(e)
    normalizer = tl.full((), 0.0, tl.float32)
    moment = tl.full((), 0.0, tl.float32)
    for start in range(0, n_vocab, BLOCK):
        mask = start + offsets < n_vocab
        logits = tl.load(row + start + offsets, mask=mask, other=NEG_INF).to(tl.float32)
        shifted = logits - top
        probs = tl.where(mask, tl.exp(shifted), 0.0)
        normalizer += tl.sum(probs, axis=0)
        moment += tl.sum(probs * tl.where(mask, shifted, 0.0), axis=0)
        tl.store(row + start + offsets, (probs * prob_scale).to(logits_ptr.dtype.element_ty), mask=mask)

    log_norm = tl.log(normalizer)
    stats = stats_ptr + pos * 4
    tl.store(stats, normalizer)
    tl.store(stats + 1, log_norm - moment / normalizer)
    tl.store(stats + 2, tl.where(labelled, log_norm - (at_target - top), 0.0))
    tl.store(stats + 3, tl.where(labelled & (argmax == target), 1.0, 0.0))


def labelled_range(targets):
    """(first, last + 1) of the labelled positions of each row, read once per step: the labels of a
    chat row are one run of answer tokens, and the positions around it only count for the entropy."""
    labelled = targets != IGNORE_INDEX
    length, any_label = targets.shape[1], labelled.any(dim=1)
    first = labelled.float().argmax(dim=1).where(any_label, length)
    last = (length - labelled.flip(1).float().argmax(dim=1)).where(any_label, length)
    return list(zip(first.tolist(), last.tolist()))


def head_and_loss(hidden, targets, labelled, head_weight, grad_scale, grad):
    """Everything the step needs from the logits of one row.

    hidden: (tokens, hidden_size), detached; targets: (tokens,) with IGNORE_INDEX where no loss
    applies; labelled: (first, last + 1) from `labelled_range`; head_weight: (vocab, hidden_size);
    grad_scale: d(training objective) / d(sum of cross-entropy); grad: zeros of the shape of `hidden`,
    filled with the gradient on `hidden`.
    Returns a (tokens, STATS) float32 tensor.
    """
    tokens, n_vocab = hidden.shape[0], head_weight.shape[0]
    stats = torch.empty(tokens, STATS, device=hidden.device, dtype=torch.float32)
    first, last = min(labelled[0], tokens), min(labelled[1], tokens)
    for begin, end, with_grad in ((0, first, False), (first, last, True), (last, tokens, False)):
        for start in range(begin, end, CHUNK):
            stop = min(start + CHUNK, end)
            target = targets[start:stop]
            logits = hidden[start:stop] @ head_weight.t()
            piece_stats = stats[start:stop]
            head_stats_kernel[(stop - start,)](logits, target, piece_stats, n_vocab, PROB_SCALE,
                                               BLOCK=BLOCK)
            if with_grad:
                inverse = (grad_scale / PROB_SCALE) / piece_stats[:, 0:1]
                piece_grad = (logits @ head_weight) * inverse.to(hidden.dtype)
                piece_grad -= head_weight[target.clamp(min=0)] * grad_scale
                piece_grad[target < 0] = 0.0
                grad[start:stop] = piece_grad
    return stats
