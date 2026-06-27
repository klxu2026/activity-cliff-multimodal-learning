from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_samples: int = 128
    batch_size: int = 16
    num_nodes: int = 12
    node_dim: int = 8
    image_size: int = 32
    smiles_vocab_size: int = 32
    smiles_length: int = 24
    cliff_threshold: float = 1.0
    train_fraction: float = 0.8


class MolecularPairDataset(Dataset):
    """Synthetic paired molecules with graph, SMILES, image, and activity labels."""

    def __init__(self, config: SyntheticDataConfig, seed: int = 0) -> None:
        self.config = config
        generator = torch.Generator().manual_seed(seed)
        n = config.num_samples * 2

        self.node_features = torch.randn(n, config.num_nodes, config.node_dim, generator=generator)
        adj = torch.randint(0, 2, (n, config.num_nodes, config.num_nodes), generator=generator).float()
        self.adjacency = torch.triu(adj, diagonal=1)
        self.adjacency = self.adjacency + self.adjacency.transpose(1, 2)
        self.smiles = torch.randint(1, config.smiles_vocab_size, (n, config.smiles_length), generator=generator)
        self.images = torch.randn(n, 1, config.image_size, config.image_size, generator=generator)

        graph_signal = self.node_features.mean(dim=(1, 2))
        smiles_signal = self.smiles.float().mean(dim=1) / config.smiles_vocab_size
        image_signal = self.images.mean(dim=(1, 2, 3))
        noise = 0.05 * torch.randn(n, generator=generator)
        self.activity = graph_signal + smiles_signal + image_signal + noise

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        left = idx
        right = idx + self.config.num_samples
        delta = self.activity[left] - self.activity[right]
        cliff = delta.abs() >= self.config.cliff_threshold
        return {
            "graph_x_a": self.node_features[left],
            "graph_adj_a": self.adjacency[left],
            "smiles_a": self.smiles[left],
            "image_a": self.images[left],
            "activity_a": self.activity[left],
            "graph_x_b": self.node_features[right],
            "graph_adj_b": self.adjacency[right],
            "smiles_b": self.smiles[right],
            "image_b": self.images[right],
            "activity_b": self.activity[right],
            "activity_delta": delta,
            "is_cliff": cliff.float(),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def create_dataloaders(config: dict, seed: int = 0) -> tuple[DataLoader, DataLoader]:
    data_cfg = SyntheticDataConfig(**config)
    dataset = MolecularPairDataset(data_cfg, seed=seed)
    train_size = int(len(dataset) * data_cfg.train_fraction)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=data_cfg.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_set, batch_size=data_cfg.batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, val_loader
