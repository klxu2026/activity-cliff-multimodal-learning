from src.data.synthetic import MolecularPairDataset, SyntheticDataConfig, create_dataloaders


def test_pair_dataset_shapes():
    cfg = SyntheticDataConfig(num_samples=8, num_nodes=6, node_dim=4, image_size=16, smiles_length=10)
    sample = MolecularPairDataset(cfg, seed=1)[0]
    assert sample["graph_x_a"].shape == (6, 4)
    assert sample["graph_adj_a"].shape == (6, 6)
    assert sample["smiles_a"].shape == (10,)
    assert sample["image_a"].shape == (1, 16, 16)
    assert sample["activity_delta"].ndim == 0


def test_create_dataloaders_batches():
    train_loader, val_loader = create_dataloaders({"num_samples": 16, "batch_size": 4}, seed=1)
    batch = next(iter(train_loader))
    assert batch["graph_x_a"].shape[0] == 4
    assert len(val_loader.dataset) == 4
