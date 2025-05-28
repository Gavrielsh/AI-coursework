"""Utility helpers."""

import random
import numpy as np
import torch

__all__ = ["set_seed", "get_device", "AverageMeter", "save_checkpoint", "load_checkpoint"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)