from __future__ import annotations

import torch
from torch import nn


class GraphEncoder(nn.Module):
    """Tiny message-passing style graph encoder for scaffold experiments."""

    def __init__(self, node_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_mean = adjacency.bmm(node_features) / degree
        node_repr = self.node_mlp(node_features + neighbor_mean)
        return node_repr.mean(dim=1)


class SmilesEncoder(nn.Module):
    """Small GRU encoder over tokenized toy SMILES strings."""

    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, embedding_dim, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)


class Image2DEncoder(nn.Module):
    """Lightweight CNN for synthetic 2D molecular depictions."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
