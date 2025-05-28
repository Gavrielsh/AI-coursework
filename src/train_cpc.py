"""CPC pretraining script."""

import argparse
from pathlib import Path
import torch
from torch.optim import Adam

from har_dataset import get_dataloaders
from cpc_model import CPCModel
from losses import info_nce_loss
from utils import set_seed, get_device, AverageMeter, save_checkpoint


def train(args):
    set_seed(args.seed)
    device = get_device()
    train_loader, _ = get_dataloaders(args.batch_size)
    model = CPCModel(hidden_dim=args.hidden_dim, projection_dim=args.projection_dim, k_steps=args.k)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    loss_meter = AverageMeter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_meter.reset()
        for x, _ in train_loader:
            x = x.to(device)
            
            # Forward through the model
            z_projected, c = model(x)                    # [B, L, P]
            
            # Build future prediction targets - all have same length L-k_steps
            valid_length = z_projected.shape[1] - model.k_steps
            targets = []
            for i in range(1, model.k_steps + 1):
                # Each target: [B, L-k_steps, P]
                target = z_projected[:, i:i + valid_length]
                targets.append(target)
            
            preds = model.predict_future(c)  # All preds have shape [B, L-k_steps, P]

            # Compute InfoNCE loss
            loss = info_nce_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))

        print(f"Epoch {epoch}/{args.epochs} | CPC Loss: {loss_meter.avg:.4f}")

    ckpt_path = Path(args.checkpoint_dir) / "cpc_encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.encoder.state_dict(), ckpt_path)
    print(f"Saved encoder weights to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPC Pretraining on HAR")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)