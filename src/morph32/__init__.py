"""MORPH-32: orbit-recurrent quantization for MLX on Apple M5."""

__version__ = "0.1.0"


def load(checkpoint, *, verify=True):
    """Load a standalone checkpoint and return model, tokenizer, and manifest."""
    from .runtime import load as load_checkpoint

    return load_checkpoint(checkpoint, verify=verify)
