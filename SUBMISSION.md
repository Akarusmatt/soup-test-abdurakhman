# Submission: <team name>

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in the
  README): https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
Shares come from `own_kernel_ms` in `runs/result.json` of the notebook above (check 4).

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `head_stats_kernel` | `solution/fused_head.py` | softmax, cross-entropy, entropy and accuracy over the vocabulary | % |
| `residual_norm_fwd_kernel` | `solution/fused_norm.py` | residual add + RMSNorm | % |
| `residual_norm_bwd_kernel` | `solution/fused_norm.py` | backward of both | % |
| `attention_split_kernel` | `solution/fused_attention.py` | RoPE of q and k, head order for q, k and v | % |
| `attention_merge_kernel` | `solution/fused_attention.py` | backward of the same | % |
| `causal_softmax_fwd_kernel` | `solution/fused_attention.py` | causal mask and softmax | % |
| `causal_softmax_bwd_kernel` | `solution/fused_attention.py` | backward of the softmax | % |
| `lora_fwd_kernel` | `solution/fused_lora.py` | dropout, `lora_A`, `lora_B`, scale, bias, add | % |
| `lora_bwd_kernel` | `solution/fused_lora.py` | backward of the same path | % |

The kit's `example_swiglu_*` kernels keep running next to them and are not counted.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Online normalizer for softmax: Milakov & Gimelshein, "Online normalizer calculation for softmax",
  https://arxiv.org/abs/1805.02867. The same idea is used by FlashAttention.
- Taking the head's gradient in chunks and handing it to the backward: `soup_cli/utils/stream_loss_chunk.py`
  in Soup 0.77.0, a Soup switch we do not turn on. Ours is a separate implementation around a fused kernel.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
See "Limits" in `README.md`.
