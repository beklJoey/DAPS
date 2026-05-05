"""Utilities for reproducible random seed management across all libraries."""

import os
import random
from typing import Optional

import numpy as np
import torch


DEFAULT_SEED: int = 42
SEEDS: list[int] = [42, 123, 7]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix all random seeds for full reproducibility.

    Sets seeds for Python random, NumPy, PyTorch (CPU and CUDA), and
    configures cuDNN to operate deterministically.

    Args:
        seed: Integer seed value to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed(index: int) -> int:
    """Return the seed at the given index from the canonical seed list.

    Args:
        index: Position in SEEDS (0-based).

    Returns:
        Seed integer at that position.

    Raises:
        IndexError: If index is out of range.
    """
    return SEEDS[index]


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """Create a seeded torch Generator for DataLoader worker reproducibility.

    Args:
        seed: Seed value; defaults to DEFAULT_SEED when None.

    Returns:
        Seeded torch.Generator instance.
    """
    generator = torch.Generator()
    generator.manual_seed(seed if seed is not None else DEFAULT_SEED)
    return generator
