"""Compile the kernels next to Soup's startup, not inside the first training step.

Compilation counts in the measured time, and importing Soup and loading the model takes many seconds
during which the kernels would otherwise wait for nothing. Triton's own async mode compiles them on a
small thread pool inside this process (the slow part, `ptxas`, runs outside the GIL); a launch that
arrives before its kernel is ready simply waits for it. If anything here fails, the first launch
compiles the kernel the normal way.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import triton

from . import fused_attention, fused_head, fused_lora, fused_norm, swiglu

WORKERS = 2
EPS = 1e-6  # any float: only the types of the arguments select the compiled kernel


def compile_all():
    half, wide, long = torch.float16, torch.float32, torch.int64
    hidden, kv = 1536, 256
    with ThreadPoolExecutor(WORKERS) as pool, triton.AsyncCompileMode(pool, ignore_errors=True):
        fused_norm.residual_norm_fwd_kernel.warmup(half, half, wide, half, half, hidden, 1.0, EPS,
                                                   grid=(1,), BLOCK=fused_norm.BLOCK)
        fused_norm.residual_norm_bwd_kernel.warmup(half, half, half, wide, half, hidden, 1.0, EPS,
                                                   grid=(1,), BLOCK=fused_norm.BLOCK)
        layout = {"Q_BLOCK": fused_attention.Q_BLOCK, "KV_BLOCK": fused_attention.KV_BLOCK,
                  "HEAD_DIM": fused_attention.HEAD_DIM}
        for kernel in (fused_attention.attention_split_kernel,
                       fused_attention.attention_merge_kernel):
            kernel.warmup(half, half, half, half, half, half, half, half, 2, 2, 6, hidden, kv,
                          1.0, grid=(1,), **layout)
        fused_attention.causal_softmax_fwd_kernel.warmup(wide, half, 2, grid=(1,),
                                                         ROW_BLOCK=fused_attention.ROW_BLOCK)
        fused_attention.causal_softmax_bwd_kernel.warmup(half, half, 2, grid=(1,),
                                                         ROW_BLOCK=fused_attention.ROW_BLOCK)
        fused_head.head_stats_kernel.warmup(half, long, wide, 151936, fused_head.PROB_SCALE,
                                            grid=(1,), BLOCK=fused_head.BLOCK)
        fused_lora.lora_fwd_kernel.warmup(half, half, half, half, half, half, half, hidden, hidden,
                                          16, 2.0, 0.95, 0, grid=(1,), BLOCK=fused_lora.BLOCK)
        fused_lora.lora_bwd_kernel.warmup(half, half, half, half, half, half, half, half, hidden,
                                          hidden, 16, 2.0, 0.95, 0, grid=(1,),
                                          BLOCK=fused_lora.BLOCK)
        # Triton compiles an integer per divisibility by 16, and tokens * 8640 always divides.
        swiglu.example_swiglu_fwd_kernel.warmup(half, half, half, 16, grid=(1,), BLOCK=swiglu.BLOCK)
        swiglu.example_swiglu_bwd_kernel.warmup(half, half, half, half, half, 16, grid=(1,),
                                                BLOCK=swiglu.BLOCK)


def start():
    if torch.cuda.is_available():
        threading.Thread(target=compile_all, name="kernel-compile", daemon=True).start()
