"""The training step: the same batch, the same gradient, in passes without padding to the batch.

Base Soup pads a batch to its longest row and sends all of it through the model at once. Here the
rows go through in passes that hold no more than one row of the maximum length, and only one pass's
activations are alive at any moment. So padded tokens drop out of every matmul, and the peak follows
the longest row, not the whole batch. A pass is a set of rows of similar length, padded to its longest
row, or a single row; it costs a fixed amount however short it is, so short rows share one.

The objective of a batch is a sum over its rows, so the gradients of the passes add up to the gradient
of the batch: the batch, its order and the optimizer step stay what they were, and
`gradient_accumulation_steps` stays 1. Rows are never mixed across batches.

TRL's and Soup's outputs for the step are filled in here from the head kernel: the loss, `entropy`,
`mean_token_accuracy`, `num_tokens` and the rewind record of the batch.
"""

import torch

from .fused_attention import ROW_BLOCK
from .fused_head import IGNORE_INDEX, head_and_loss, labelled_range
from .stack import FusedStack

PASS_COST = 200  # what one more pass costs, in tokens: launches, and reading every weight once more


def plan_passes(lengths, max_tokens):
    """Rows of a batch in passes, as lists of row indices: the cheapest split of the rows sorted by
    length in which no pass holds more than `max_tokens` positions, its rows times its longest row."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    best = {0: (0, [])}  # rows planned -> (cost, passes)
    for stop in range(1, len(order) + 1):
        options = []
        for start in range(stop):
            area = (stop - start) * lengths[order[stop - 1]]
            if start in best and (area <= max_tokens or stop - start == 1):
                cost, passes = best[start]
                options.append((cost + PASS_COST + area, passes + [order[start:stop]]))
        best[stop] = min(options, key=lambda option: option[0])
    return best[len(order)][1]


def supported(trainer):
    """Only the run this package was written for: Qwen2 under LoRA, in fp16, without evaluation.
    Anything else keeps the stock path."""
    model = getattr(trainer, "model", None)
    args = getattr(trainer, "args", None)
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "get_base_model"):
        return False
    if not getattr(args, "fp16", False) or getattr(args, "gradient_accumulation_steps", 0) != 1:
        return False
    if getattr(trainer, "eval_dataset", None) is not None:
        return False
    if not torch.cuda.is_available() or getattr(args, "loss_type", None) != "nll":
        return False
    if not 0 < getattr(args, "max_length", 0) <= ROW_BLOCK:
        return False
    causal_lm = model.get_base_model()
    if type(causal_lm).__name__ != "Qwen2ForCausalLM" or causal_lm.config.hidden_act != "silu":
        return False
    if set(causal_lm.config.layer_types) != {"full_attention"}:
        return False
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    lora = [name for name in trainable if ".q_proj." in name or ".v_proj." in name]
    return bool(trainable) and len(trainable) == len(lora) \
        and all("lora_A" in name or "lora_B" in name for name in trainable)


def record_outputs(trainer, batch_stats, row_tokens, lengths):
    """What TRL's compute_loss and Soup's rewind recorder record, from the numbers of the kernel."""
    entropy, ce, correct = batch_stats.sum(dim=0).tolist()
    labelled, tokens = sum(row_tokens), sum(lengths)
    metrics = trainer._metrics["train"]
    metrics["entropy"].append(entropy / tokens)
    trainer._total_train_tokens += tokens
    metrics["num_tokens"] = [trainer._total_train_tokens]
    metrics["mean_token_accuracy"].append(correct / labelled if labelled else 0.0)
    state = getattr(trainer, "_soup_rewind_state", None)
    if state is not None and not state.failed:
        row_loss = [row[1] / max(count, 1) for row, count in zip(batch_stats.tolist(), row_tokens)]
        try:
            state.on_micro_batch(step=int(trainer.state.global_step) + 1, row_loss=row_loss,
                                 row_tokens=row_tokens)
        except Exception as exc:  # noqa: BLE001 - the recorder must never stop training
            state.fail(exc)


def fused_training_step(trainer, stack, model, inputs, num_items_in_batch=None):
    if not model.training:
        model.train()
    inputs = trainer._prepare_inputs(inputs)
    input_ids, labels = inputs["input_ids"], inputs["labels"]
    lengths = inputs["attention_mask"].sum(dim=1).tolist()
    targets = labels.roll(-1, dims=1)
    targets[:, -1] = IGNORE_INDEX
    row_tokens = (targets != IGNORE_INDEX).sum(dim=1).tolist()
    ranges = labelled_range(targets)
    denominator = int(num_items_in_batch) if num_items_in_batch is not None else sum(row_tokens)
    denominator = max(denominator, 1)  # a batch of prompts cut off by max_length has no labels
    loss_scale = trainer.accelerator.scaler.scale(torch.ones((), device=input_ids.device)).item()
    grad_scale = loss_scale / denominator

    stack.begin_step()
    rows = [None] * len(lengths)
    for group in plan_passes(lengths, stack.max_positions):
        length = max(lengths[row] for row in group)
        hidden = stack.forward(input_ids[group, :length])
        grad = torch.zeros_like(hidden)
        flat = hidden.detach()
        with torch.no_grad():
            for slot, row in enumerate(group):
                start, stop = slot * length, slot * length + lengths[row]
                stats = head_and_loss(flat[start:stop], targets[row, :lengths[row]], ranges[row],
                                      stack.embed, grad_scale, grad[start:stop])
                rows[row] = stats[:, 1:].sum(dim=0)
        if any(row_tokens[row] for row in group):
            hidden.backward(grad)
        del hidden, grad, flat, stats
    stack.end_step()

    batch_stats = torch.stack(rows).double()
    record_outputs(trainer, batch_stats, row_tokens, lengths)
    return (batch_stats[:, 1].sum() / denominator).float()


def install(trainer):
    """Called once the trainer and the model exist, before training starts."""
    if not supported(trainer) or getattr(trainer, "_fused_stack", None) is not None:
        return False
    causal_lm = trainer.model.get_base_model()
    stack = FusedStack(causal_lm, trainer.args.max_length)
    torch.cuda.empty_cache()
    trainer._fused_stack = stack

    def training_step(model, inputs, num_items_in_batch=None):
        return fused_training_step(trainer, stack, model, inputs, num_items_in_batch)

    trainer.training_step = training_step
    return True


def skip_dataset_fingerprints():
    """`datasets` pickles the whole tokenizer, once per `map`, to name a cache file. Soup's dataset is
    built in memory, in a fresh process, so nothing would ever read that cache."""
    import datasets.fingerprint as fingerprint

    fingerprint.update_fingerprint = lambda *args, **kwargs: fingerprint.generate_random_fingerprint()


def patch_soup_trainer():
    """Attach to Soup's wrapper at the point where the trainer exists and training has not begun."""
    from soup_cli.trainer import sft

    original = sft.SFTTrainerWrapper.train

    def train(self, *args, **kwargs):
        install(self.trainer)
        return original(self, *args, **kwargs)

    sft.SFTTrainerWrapper.train = train
