import torch

from src.data.synthetic import SyntheticDataConfig, MolecularPairDataset
from src.losses import CliffAwareLoss
from src.models import ActivityCliffModel


def test_model_forward_and_loss():
    cfg = SyntheticDataConfig(num_samples=4)
    sample = MolecularPairDataset(cfg, seed=2)[0]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    model = ActivityCliffModel(node_dim=cfg.node_dim, smiles_vocab_size=cfg.smiles_vocab_size)
    outputs = model(batch)
    losses = CliffAwareLoss()(outputs, batch)
    assert outputs["z_a"].shape == (1, 32)
    assert outputs["activity_delta"].shape == (1,)
    assert torch.isfinite(losses["loss"])
