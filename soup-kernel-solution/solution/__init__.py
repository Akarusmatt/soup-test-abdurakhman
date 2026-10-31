"""Fused fp16 training for `soup train`: see README.md for what each module does and why.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.
"""

from . import precompile
from .step import patch_soup_trainer, skip_dataset_fingerprints
from .swiglu import patch_qwen2_mlp


def apply():
    precompile.start()
    patch_qwen2_mlp()  # the example from the kit: it does not count as our own kernel
    patch_soup_trainer()
    skip_dataset_fingerprints()
