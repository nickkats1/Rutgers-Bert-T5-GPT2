"""Reproducibility utilities for GPT-2 fine-tuning."""

import numpy as np
import torch

from src.gpt2.config import SEED


def set_seed():
    """Set random seeds for reproducibility across torch and numpy.

    Uses SEED from the GPT-2 config to ensure deterministic behaviour.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
