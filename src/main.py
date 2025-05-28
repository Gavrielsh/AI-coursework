"""Experiment controller: pretrain CPC then run linear probe."""

import subprocess
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPC + linear probe pipeline")
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_dir) / "cpc_encoder.pt"

    # Step 1: CPC pretraining
    pretrain_cmd = [
        "python",
        "train_cpc.py",
        "--epochs",
        str(args.pretrain_epochs),
        "--hidden_dim",
        str(args.hidden_dim),
        "--checkpoint_dir",
        args.checkpoint_dir,
    ]
    subprocess.run(pretrain_cmd, check=True)

    # Step 2: Linear probe
    probe_cmd = [
        "python",
        "linear_probe.py",
        "--ckpt",
        str(checkpoint_path),
        "--epochs",
        str(args.probe_epochs),
    ]
    subprocess.run(probe_cmd, check=True)