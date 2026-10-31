"""The Qwen2 decoder as one fp16 pipeline over the rows of one pass.

Base Soup runs the decoder in fp32 with fp16 islands: the embeddings are fp32 after
`prepare_model_for_kbit_training`, and a bitsandbytes Linear4bit casts its input to fp16 and its
output back to the dtype of the input. So every activation that autograd keeps is fp32, and every
matmul comes with two casts in the forward and two in the backward. Here the embeddings are read in
fp16 and stay there, which is what `eval_adapter.py` runs as well: half the memory, no casts.

`forward` is `Qwen2Model.forward` written out for the frozen 4-bit model under LoRA: same weights,
same order of operations, no attention mask because right-padded rows only need the causal one.
Tensors are flat, (rows * length, width), so everything but attention treats a token as a token.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .fused_attention import causal_attention
from .fused_lora import lora_linear
from .fused_norm import first_norm, residual_norm
from .repack import repack_embedding, repack_linear
from .swiglu import swiglu

SEED_STRIDE = 1000003  # more than the elements of a pass of 512 tokens: no two calls hash alike


@dataclass
class LoraPath:
    """A q_proj or v_proj with its adapter."""
    weight: torch.Tensor
    bias: torch.Tensor
    lora_a: torch.nn.Parameter
    lora_b: torch.nn.Parameter
    scaling: float
    keep_prob: float
    a: torch.Tensor = None  # fp16 leaves of this step, see begin_step()
    b: torch.Tensor = None

    def __call__(self, x, seed):
        return lora_linear(x, self.a, self.b, self.weight, self.bias, self.scaling, self.keep_prob,
                           seed)


@dataclass
class Block:
    q: LoraPath
    v: LoraPath
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    o_weight: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    post_norm: torch.Tensor
    next_norm: torch.Tensor


def lora_path(module):
    adapter = module.active_adapters[0]
    weight, bias = repack_linear(module.base_layer)
    dropout = module.lora_dropout[adapter]
    return LoraPath(weight, bias, module.lora_A[adapter].weight, module.lora_B[adapter].weight,
                    float(module.scaling[adapter]), 1.0 - float(getattr(dropout, "p", 0.0)))


class FusedStack:
    """Takes the weights out of a loaded PEFT Qwen2ForCausalLM; the module tree stays in place for
    `save_pretrained`, which only reads the adapter."""

    def __init__(self, causal_lm, max_positions):
        decoder = causal_lm.model
        self.max_positions = max_positions
        config = causal_lm.config
        self.embed = repack_embedding(decoder.embed_tokens)
        self.scale = (config.hidden_size // config.num_attention_heads) ** -0.5
        self.eps = config.rms_norm_eps
        self.first_norm_weight = decoder.layers[0].input_layernorm.weight.data
        self.blocks = []
        layers = decoder.layers
        for index, layer in enumerate(layers):
            attn, mlp = layer.self_attn, layer.mlp
            k_weight, k_bias = repack_linear(attn.k_proj)
            following = layers[index + 1].input_layernorm if index + 1 < len(layers) else decoder.norm
            self.blocks.append(Block(lora_path(attn.q_proj), lora_path(attn.v_proj), k_weight, k_bias,
                                     repack_linear(attn.o_proj)[0], repack_linear(mlp.gate_proj)[0],
                                     repack_linear(mlp.up_proj)[0], repack_linear(mlp.down_proj)[0],
                                     layer.post_attention_layernorm.weight.data,
                                     following.weight.data))
        # Qwen2RotaryEmbedding.forward once for every position: one row of frequencies per position.
        inv_freq = decoder.rotary_emb.inv_freq.float()
        angles = torch.outer(torch.arange(max_positions, device=inv_freq.device).float(), inv_freq)
        self.cos = angles.cos().to(torch.float16)
        self.sin = angles.sin().to(torch.float16)
        self.lora = [path for block in self.blocks for path in (block.q, block.v)]

    def begin_step(self):
        """fp16 leaves for the adapter, shared by the passes of a step. Under autocast the same copy
        is made for every call of every layer."""
        for path in self.lora:
            path.a = path.lora_a.detach().to(torch.float16).requires_grad_()
            path.b = path.lora_b.detach().t().to(
                torch.float16, memory_format=torch.contiguous_format).requires_grad_()

    def end_step(self):
        """The optimizer step needs a gradient on every parameter. A batch whose rows were all cut
        off before the answer has none, so its gradient is zero."""
        for path in self.lora:
            grad_a = path.a.grad if path.a.grad is not None else torch.zeros_like(path.a)
            grad_b = path.b.grad if path.b.grad is not None else torch.zeros_like(path.b)
            path.lora_a.grad = grad_a.to(torch.float32)
            path.lora_b.grad = grad_b.t().to(torch.float32)
            path.a = path.b = None

    def attention(self, block, x, cos, sin, seed, length):
        out = causal_attention(block.q(x, seed), F.linear(x, block.k_weight, block.k_bias),
                               block.v(x, seed + SEED_STRIDE), cos, sin, self.scale, length)
        return F.linear(out, block.o_weight)

    def forward(self, ids):
        """Last hidden state, shape (rows * length, hidden_size), for right-padded token ids of shape
        (rows, length)."""
        length = ids.shape[1]
        cos, sin = self.cos[:length], self.sin[:length]
        h = self.embed[ids.view(-1)]
        norm = first_norm(h, self.first_norm_weight, self.eps)
        seed = int(torch.randint(2 ** 30, ()))
        for block in self.blocks:
            seed += 2 * SEED_STRIDE
            h, norm = residual_norm(h, self.attention(block, norm, cos, sin, seed, length),
                                    block.post_norm, self.eps)
            gated = swiglu(F.linear(norm, block.gate), F.linear(norm, block.up))
            h, norm = residual_norm(h, F.linear(gated, block.down), block.next_norm, self.eps)
        return norm
