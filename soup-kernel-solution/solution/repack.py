"""Weights for a step that makes several passes: dequantize once, then matmul directly.

On a T4 a bitsandbytes layer dequantizes its NF4 weight for each forward and again for each backward.
Rows here go through the model one at a time, so the same scheme would dequantize eight times a
step. Instead every Linear4bit is dequantized once with bitsandbytes' own function, into the same
fp16 values it hands to the matmul on every call, and its packed copy is released.

Frozen weights in fp16 also cover the tied embedding / lm_head: base Soup keeps it in fp32 and copies
it to fp16 for the head on every forward under autocast. The trained adapter never contains any of
these tensors, and `eval_adapter.py` reloads the 4-bit model on its own.
"""

import torch


def dequantized(linear):
    """The fp16 weight of a bitsandbytes Linear4bit, shape (out, in)."""
    import bitsandbytes as bnb

    weight = linear.weight
    return bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(torch.float16)


def release_packed(linear):
    """Drop the packed weight and its quantization state; nothing reads them after this."""
    weight = linear.weight
    weight.data = torch.empty(0, dtype=weight.dtype, device=weight.device)
    weight.quant_state = None
    linear.quant_state = None


def repack_linear(linear):
    """(weight, bias) of a Linear4bit in fp16; the packed storage is gone afterwards."""
    weight = dequantized(linear)
    release_packed(linear)
    bias = None
    if linear.bias is not None:
        linear.bias.data = linear.bias.data.to(torch.float16)
        bias = linear.bias.data
    return weight, bias


def repack_embedding(embedding):
    """The frozen token embedding (tied to lm_head) in fp16, the dtype the head uses anyway."""
    embedding.weight.data = embedding.weight.data.to(torch.float16)
    return embedding.weight.data
