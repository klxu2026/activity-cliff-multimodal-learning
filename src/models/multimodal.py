from __future__ import annotations

import torch
from torch import nn

from .encoders import GraphEncoder, Image2DEncoder, SmilesEncoder
from .fusion import MultimodalFusion


class ActivityCliffModel(nn.Module):
    """Minimal pair model for multimodal activity cliff experiments."""

    def __init__(
        self,
        node_dim: int,
        smiles_vocab_size: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = GraphEncoder(node_dim, hidden_dim, embedding_dim)
        self.smiles_encoder = SmilesEncoder(smiles_vocab_size, hidden_dim, embedding_dim)
        self.image_encoder = Image2DEncoder(embedding_dim)
        self.fusion = MultimodalFusion(embedding_dim, hidden_dim, dropout)
        self.activity_head = nn.Linear(embedding_dim, 1)

    def encode_molecule(self, graph_x: torch.Tensor, graph_adj: torch.Tensor, smiles: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        graph_z = self.graph_encoder(graph_x, graph_adj)
        smiles_z = self.smiles_encoder(smiles)
        image_z = self.image_encoder(image)
        return self.fusion(graph_z, smiles_z, image_z)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_a = self.encode_molecule(batch["graph_x_a"], batch["graph_adj_a"], batch["smiles_a"], batch["image_a"])
        z_b = self.encode_molecule(batch["graph_x_b"], batch["graph_adj_b"], batch["smiles_b"], batch["image_b"])
        pred_a = self.activity_head(z_a).squeeze(-1)
        pred_b = self.activity_head(z_b).squeeze(-1)
        return {
            "z_a": z_a,
            "z_b": z_b,
            "activity_a": pred_a,
            "activity_b": pred_b,
            "activity_delta": pred_a - pred_b,
        }
