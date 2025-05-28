"""CPC model components: Encoder, Projector, Transformer context, and prediction heads."""

from typing import List

import torch
import torch.nn as nn

__all__ = ["Encoder", "CPCModel"]


class Encoder(nn.Module):
    """Simple 1D CNN encoder mapping [B, 9, 128] -> [B, T, D]."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, 9, 128] -> Output: [B, T, D]
        feat = self.cnn(x)  # [B, D, L]
        return feat.permute(0, 2, 1)  # [B, L, D]


class CPCModel(nn.Module):
    """Full CPC model with Transformer context and k prediction heads."""

    def __init__(self, hidden_dim: int = 128, projection_dim: int = 64, k_steps: int = 3):
        super().__init__()
        self.k_steps = k_steps
        self.encoder = Encoder(hidden_dim)
        self.projector = nn.Linear(hidden_dim, projection_dim)
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim, nhead=4, dim_feedforward=projection_dim * 2,
            batch_first=True,
        )
        self.context_net = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # One prediction head per future step
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(projection_dim, projection_dim) for _ in range(k_steps)]
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 9, 128]
        z = self.encoder(x)               # [B, L, D]
        z = self.projector(z)             # [B, L, P]
        c = self.context_net(z)           # [B, L, P]
        return z, c

    def predict_future(self, context: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of k predictions, each of shape [B, L-k_steps, P]."""
        preds = []
        # All predictions have the same length: L - k_steps
        # This ensures alignment with targets
        context_truncated = context[:, :-self.k_steps]  # [B, L-k_steps, P]
        for head in self.prediction_heads:
            preds.append(head(context_truncated))
        return preds