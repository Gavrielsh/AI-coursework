"""UCI HAR dataset loader and pre‑processing utilities.
All comments are written in English only, as requested.
The dataset is automatically downloaded (≤ 15 MB) if not present.
"""
from pathlib import Path
import urllib.request
import zipfile
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["HARDataset", "get_dataloaders"]

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
_DATA_DIR = Path("data/uci_har")


def _download_dataset() -> None:
    """Download and extract the UCI HAR dataset if it does not exist."""
    if (_DATA_DIR / "UCI HAR Dataset").exists():
        return
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _DATA_DIR / "har.zip"
    print("Downloading UCI HAR dataset …")
    urllib.request.urlretrieve(_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_DATA_DIR)
    zip_path.unlink()  # remove zip file to save space


def _load_signals(split: str) -> np.ndarray:
    """Load inertial signals for the given split (train/test).

    Each signal file has shape [samples, 128] (time dimension).
    We stack 9 such signals along the channel dimension, resulting in
    shape [samples, 9, 128].
    """
    base_dir = _DATA_DIR / "UCI HAR Dataset" / f"{split}/Inertial Signals"
    signal_suffixes = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]
    data = [
        np.loadtxt(base_dir / f"{s}{split}.txt")[:, None, :] for s in signal_suffixes
    ]  # list of [N, 1, 128]
    stacked = np.concatenate(data, axis=1)  # [N, 9, 128]
    return stacked.astype(np.float32)


def _load_labels(split: str) -> np.ndarray:
    """Load activity labels (1–6)."""
    path = _DATA_DIR / "UCI HAR Dataset" / f"{split}/y_{split}.txt"
    return np.loadtxt(path).astype(np.int64) - 1  # zero‑based


class HARDataset(Dataset):
    """PyTorch Dataset wrapping UCI HAR inertial signals."""

    def __init__(self, split: str = "train"):
        assert split in {"train", "test"}
        _download_dataset()
        self.signals = _load_signals(split)
        self.labels = _load_labels(split)
        # Standardize per channel
        mean = self.signals.mean(axis=(0, 2), keepdims=True)
        std = self.signals.std(axis=(0, 2), keepdims=True) + 1e-6
        self.signals = (self.signals - mean) / std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.signals[idx])  # [9, 128]
        y = torch.tensor(self.labels[idx])
        return x, y


def get_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders."""
    train_ds = HARDataset("train")
    test_ds = HARDataset("test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )