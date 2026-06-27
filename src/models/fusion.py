from __future__ import annotations

import torch
from torch import nn


class MultimodalFusion(nn.Module):
    """Concatenate graph, SMILES, and image representations."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, graph_z: torch.Tensor, smiles_z: torch.Tensor, image_z: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([graph_z, smiles_z, image_z], dim=-1))
