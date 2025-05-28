"""Linear evaluation on frozen CPC encoder."""

import argparse
from pathlib import Path
from typing import cast

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from har_dataset import get_dataloaders
from cpc_model import Encoder
from utils import get_device, AverageMeter, load_checkpoint


def _feature_dim(encoder: Encoder) -> int:
    """Return feature dimension produced by the encoder."""
    dummy = torch.zeros(1, 9, 128)           # [B, C, T]
    with torch.no_grad():
        feat = encoder(dummy)                # [1, L, D]
    return feat.shape[-1]                    # D


def evaluate(
    encoder: Encoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 20,
):
    device = get_device()
    encoder.to(device).eval()

    # Freeze encoder parameters
    for p in encoder.parameters():
        p.requires_grad = False

    # -------- OPTION 1: dynamic feature-dim --------
    feat_dim = _feature_dim(encoder)

    # -------- OPTION 2: explicit cast (uncomment) --------
    # from torch.nn import BatchNorm1d
    # feat_dim = cast(BatchNorm1d, encoder.cnn[-2]).num_features

    clf = nn.Linear(feat_dim, 6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        clf.train()
        loss_meter = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                feat = encoder(x)[:, -1]     # last timestep representation

            logits = clf(feat)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))

        print(f"Epoch {epoch}/{epochs} | Probe Loss: {loss_meter.avg:.4f}")

    # ---------------- Evaluation ----------------
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            feat = encoder(x)[:, -1]
            preds = clf(feat).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100.0 * correct / total
    print(f"Linear Probe Accuracy: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear evaluation on CPC-pretrained encoder"
    )
    parser.add_argument("--ckpt", type=str, default="checkpoints/cpc_encoder.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders(args.batch_size)

    enc = Encoder()
    enc.load_state_dict(load_checkpoint(Path(args.ckpt), map_location=get_device()))
    evaluate(enc, train_loader, test_loader, epochs=args.epochs)