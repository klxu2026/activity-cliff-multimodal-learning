from pathlib import Path

from src.training import train


def test_toy_training_smoke(tmp_path):
    config = {
        "seed": 3,
        "device": "cpu",
        "data": {"num_samples": 16, "batch_size": 4, "image_size": 16, "smiles_length": 8},
        "model": {"hidden_dim": 16, "embedding_dim": 8, "dropout": 0.0},
        "training": {"epochs": 1, "learning_rate": 0.001, "contrastive_weight": 0.25, "margin": 1.0},
        "logging": {"output_dir": str(tmp_path), "csv_name": "metrics.csv", "tensorboard": False},
    }
    metrics = train(config)
    assert "val_loss" in metrics
    assert (Path(tmp_path) / "metrics.csv").exists()
    assert (Path(tmp_path) / "model.pt").exists()
