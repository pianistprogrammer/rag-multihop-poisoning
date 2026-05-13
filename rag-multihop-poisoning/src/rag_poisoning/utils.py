"""
Utility functions for device management, reproducibility, and logging.
"""

import random
import logging
from typing import Literal

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority: CUDA > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: MPS backend does not support deterministic operations yet


def setup_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO") -> None:
    """
    Configure logging for the project.

    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_model_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        dict: Device information including type, name, and memory
    """
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info
