"""InfoNCE loss for CPC."""

import torch
import torch.nn.functional as F

__all__ = ["info_nce_loss"]


def info_nce_loss(preds, targets):
    """Compute InfoNCE loss across the batch for each future step.

    Args:
        preds  (List[Tensor]): list of predicted embeddings [B, L-k, P]
        targets(List[Tensor]): list of ground‑truth embeddings [B, L-k, P]
    Returns:
        Tensor: scalar loss value
    """
    device = preds[0].device
    total_loss = torch.tensor(0.0, device=device)
    for p, t in zip(preds, targets):
        B, L, P = p.shape
        p_flat = p.reshape(B * L, P)        # [BL, P]
        t_flat = t.reshape(B * L, P)        # [BL, P]
        
        # Normalize embeddings for better contrastive learning
        p_flat = F.normalize(p_flat, dim=1)
        t_flat = F.normalize(t_flat, dim=1)
        
        logits = p_flat @ t_flat.T          # [BL, BL]
        labels = torch.arange(B * L, device=device)
        total_loss += F.cross_entropy(logits, labels)
    return total_loss / len(preds)