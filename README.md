# Soup Kernel Challenge: fused fp16 training

Our solution to the [Soup Kernel Challenge](https://github.com/MakazhanAlpamys/soup-kernel-challenge)
at Decentrathon 6.0. It makes the fixed `soup train` run (Qwen2.5-1.5B, 4-bit NF4, LoRA r=16, 64 steps)
faster and much lighter on a free Kaggle T4, with the same task, the same Soup outputs and an adapter of
the same quality.

**Public Kaggle notebook:** <https://www.kaggle.com/code/YOU/NOTEBOOK?scriptVersionId=VERSION>

## Results

### On a Kaggle T4 (the numbers that count)

Fill this table from `runs/result.json` of the public notebook.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | KAGGLE | KAGGLE | KAGGLE |
| Peak GPU memory, MiB | KAGGLE | KAGGLE | KAGGLE |
| Eval loss, public rows | KAGGLE | KAGGLE | KAGGLE |
| Own kernels' share (check 4) | | KAGGLE | |

### Measured locally, on the T4's code path

We ran the kit's unchanged harness (base, solution, solution, base) on an RTX 5060 Ti 16 GB under Windows,
with Soup forced onto the T4's fp16 path, and with `triton-windows` 3.6.0 standing in for Triton 3.6.0.
This is not a T4 and not a scored result. Memory and quality carry over; time does not.

| | Public rows, seed 1 | Rows swapped, seed 7 |
|---|---|---|
| Base peak allocated by PyTorch, MiB | 13,433 | 13,401 |
| Our peak allocated by PyTorch, MiB | **3,955** | **3,955** |
| Our net nvidia-smi peak, MiB | 4,330 | 4,274–4,799 |
| Eval loss, base → ours | 2.04967 → 2.04930 (−0.018%) | 2.09548 → 2.09556 (+0.004%) |
| Loss curve vs base: first step, mean | 0.01%, 0.06% | 0.01%, 0.05% |
| Own kernels' share of training time, lowest run | 10.2% | 11.7% |
| Passes per step, of 4 rows | 2.58 | 2.89 |
| Median training step, ours | 0.235 s | 0.275–0.414 s |
| Whole `soup train`, ours | 28.8 s | 37.2–43.5 s |

Checks 1–8 and 10 of `harness/check_rules.py` pass on both, including all 79 Soup regression tests.
Check 9 cannot pass off a T4: it requires one.

We do not quote a local speedup. Base needs more memory than this card has on the public rows and spills
into system RAM, which inflates its time to about 101 s, and on the other rows a shared desktop GPU gave
the same code steps of 0.28 s and 0.41 s. From the kit's T4 figures we **expect** roughly 1.5–1.9x in time
and 3.2–3.4x in memory on a T4. That is an estimate, not a measurement.

### How we tested correctness

- `solution/test_kernels.py`: 45 tests of every kernel against plain PyTorch, forward and backward.
- On real weights, against a full fp32 reference: the LoRA gradient of base Soup is 1.6% off, ours 2.2%
  (cosine 0.9997), with the loss within 0.02%.
- Each kernel compiles exactly once across row lengths of every divisibility by 8 and 16.

## What we did

Reading the pinned sources showed where the 14.3 GB and most of the elementwise time really go:

- The embeddings are fp32 after `prepare_model_for_kbit_training`, and `Linear4bit` casts its output back
  to the dtype of its input. So the whole decoder runs in **fp32 with fp16 islands** around each matmul.
  Every activation autograd keeps is fp32, and every matmul brings casts in both directions.
- Batches are padded to the longest row. The peak batch is 4 × 512 tokens, so padded tokens are in every
  matmul, and the attention activations grow with the square of the padded length.
- Full `[batch, seq, 151936]` logits are read by the loss, accelerate's fp32 conversion, TRL's entropy
  and accuracy, and Soup's per-row loss: about 4 GB and several softmaxes over the vocabulary.

Five changes follow from that. All are in `solution/` and are installed by `apply()`.

| # | Change | File | Time | Memory |
|---|---|---|---|---|
| 1 | Rows go through in passes of at most 512 positions, not as one padded batch | `step.py` | almost no padded tokens in any matmul | one pass of activations alive, not the whole batch |
| 2 | fp16 activations end to end, with attention scores in fp32 inside | `stack.py`, `fused_attention.py` | no casts | half the size of every activation |
| 3 | Logits live for one chunk; one fused kernel gives loss, entropy and accuracy, and one matmul gives the gradient | `fused_head.py` | one pass over the vocabulary instead of several | about 4 GB of logit buffers gone |
| 4 | Weights are dequantized once | `repack.py` | no dequantization in the loop | **costs** 1.9 GiB of static memory, the one real trade-off |
| 5 | Residual + RMSNorm, attention and LoRA are fused kernels | `fused_norm.py`, `fused_attention.py`, `fused_lora.py` | launches do not multiply by the number of rows | no stored dropout noise, no saved attention scores or repeated k and v |

### Why this is the same task

- **Passes instead of one padded batch.** The objective of a batch is a sum over its rows, so the
  gradients of the passes add up to the gradient of the batch (`step.py`). The batch, the row order, the
  optimizer step and `gradient_accumulation_steps: 1` are unchanged, and rows never mix across batches.
  This is padding removal in our own code, which the rules allow. Layers see no difference: Qwen2 has no
  batch statistics. Inside a pass, short rows of similar length share right padding: a real token only
  sees earlier positions, so no attention mask is needed, as in base Soup's batch.
- **fp16.** It is the path `harness/eval_adapter.py` runs: there the model is loaded in fp16. Attention
  scores stay in fp32, because they reach 13,000, where fp16 steps by 8, and they feed an exponential.
- **Resident weights.** `repack.py` calls bitsandbytes' own `dequantize_4bit`, so the values are exactly
  the ones bitsandbytes passes to its matmul on every call.
- **Metrics.** Loss, `entropy`, `mean_token_accuracy`, `num_tokens` and the rewind rows use TRL's and
  Soup's formulas on the same fp16 logits, in fp32 inside the kernel. None of them is sampled.
- **Dropout.** `lora_dropout` keeps its rate and its `1 / (1 - p)` scaling. Its mask is a hash of the
  element's position and a per-call seed, so the backward regenerates it instead of storing it.
- **Adapter.** A standard PEFT LoRA adapter; no trainable parameter was added.

`supported()` in `step.py` limits all of this to Qwen2 under this LoRA, in fp16, without evaluation.
Anything else keeps the stock path, which is also what Soup's regression tests see.

### Our kernels

| Kernel | Replaces |
|---|---|
| `head_stats_kernel` | softmax, cross-entropy, entropy, argmax over 151,936 tokens, one launch per chunk |
| `residual_norm_fwd_kernel` / `_bwd_kernel` | residual add + RMSNorm: 9 ops into one launch that writes both outputs |
| `attention_split_kernel` / `attention_merge_kernel` | RoPE of q and k and the moves between token order and head order, one launch each way |
| `causal_softmax_fwd_kernel` / `_bwd_kernel` | the causal mask, the softmax and its backward, in place. No repeated k or v: q is laid out so that 12 query heads share 4 key/value heads directly |
| `lora_fwd_kernel` / `_bwd_kernel` | dropout, `lora_A`, `lora_B`, scale, bias and add |

Kernels compile on a small thread pool during Soup's startup (`precompile.py`), using Triton's own async
mode, because compilation counts in the measured time. Integer arguments that vary between launches are
marked `do_not_specialize`, because Triton otherwise compiles one variant per divisibility by 8 and 16,
and each kernel compiles exactly once. The kit's SwiGLU keeps running; it is not counted.

Ideas taken, code written by us: the online softmax normalizer of
[Milakov & Gimelshein](https://arxiv.org/abs/1805.02867), and the chunked head gradient of Soup's
`utils/stream_loss_chunk.py` (a Soup switch we do not turn on).

## How to run it

On Kaggle: import `templates/kaggle_notebook.ipynb`, set `REPO_URL`, choose **GPU T4 x2** with Internet on,
and press **Run All** (about 10 minutes). Nothing else needs editing.

Locally on Linux with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
```

```bash
python task/prepare_data.py
```

```bash
python -m pytest solution -q
```

```bash
python harness/run_task.py
```

```bash
python harness/check_rules.py
```

```bash
python harness/score.py runs/result.json
```

Only numbers from a T4 count: on newer GPUs Soup picks bf16, and `supported()` then keeps the stock path.

## Limits

- **It has not run on a T4 yet.** It was built without one, so everything above is local until the
  notebook has run. The riskiest difference is the CPU: Kaggle's is slower, and a local step spent about
  160 ms on the main thread before short rows shared passes. If the T4 run turns out CPU-bound,
  `PASS_COST` in `step.py` is the first thing to raise.
- **Only this run is fused.** `supported()` keeps the stock path for anything else, such as bf16, other
  models, other LoRA targets, `max_length` above 512 or evaluation during training.
- **Resident weights are a trade-off.** They cost 1.9 GiB, about 45% of our peak, to save dequantization.
  Keeping weights packed would give about 2.4 GB but dequantize about five times a step. We chose time.
- **The memory floor.** The peak is 2,980 MiB of static weights plus 975 MiB for the longest row. Rows
  cannot go below that without recomputing activations.
- **The time floor.** About 13 s here (about 20 s in the kit's T4 figures) is imports, model load and
  saving, which we did not touch.
- **fp16 is a little noisier.** The gradient error is 2.2% against 1.6% for base. It did not show in eval
  loss, but it is real.
- **Dropout uses our own hash**, so its mask differs from `torch`'s for the same seed, like any other
  seed. The rate and scaling are the same.
- **Next steps:** 8-bit `gate`/`up` activations (about 250 MiB), lower-triangular attention probabilities
  (about 90 MiB), and one kernel for LoRA and the attention split, to cut launches further.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. `harness/`, `tests/`, `task/` and `templates/` are the kit's,
unchanged. Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2).
Model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
