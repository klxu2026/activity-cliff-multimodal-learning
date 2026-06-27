from __future__ import annotations

import random
from collections import defaultdict

import torch

from src.data import create_dataloaders
from src.losses import CliffAwareLoss
from src.models import ActivityCliffModel
from src.training.logger import ExperimentLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_model(config: dict) -> ActivityCliffModel:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return ActivityCliffModel(
        node_dim=data_cfg["node_dim"],
        smiles_vocab_size=data_cfg["smiles_vocab_size"],
        hidden_dim=model_cfg["hidden_dim"],
        embedding_dim=model_cfg["embedding_dim"],
        dropout=model_cfg.get("dropout", 0.1),
    )


def _average(metrics: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / max(len(values), 1) for key, values in metrics.items()}


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = defaultdict(list)

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(training):
            outputs = model(batch)
            losses = criterion(outputs, batch)
            mae = (outputs["activity_delta"] - batch["activity_delta"]).abs().mean()
            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()

        for key, value in losses.items():
            metrics[key].append(float(value.detach().cpu()))
        metrics["mae"].append(float(mae.detach().cpu()))

    return _average(metrics)


def train(config: dict) -> dict[str, float]:
    set_seed(config.get("seed", 0))
    device = torch.device(config.get("device", "cpu"))
    train_loader, val_loader = create_dataloaders(config["data"], seed=config.get("seed", 0))
    model = build_model(config).to(device)
    criterion = CliffAwareLoss(
        contrastive_weight=config["training"].get("contrastive_weight", 0.5),
        margin=config["training"].get("margin", 1.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-3),
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    logger = ExperimentLogger(
        output_dir=config["logging"]["output_dir"],
        csv_name=config["logging"].get("csv_name", "metrics.csv"),
        use_tensorboard=config["logging"].get("tensorboard", True),
    )

    final_metrics = {}
    for epoch in range(1, config["training"].get("epochs", 1) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        logger.log(epoch, train_metrics, "train")
        logger.log(epoch, val_metrics, "val")
        final_metrics = {f"val_{key}": value for key, value in val_metrics.items()}

    torch.save(model.state_dict(), f"{config['logging']['output_dir']}/model.pt")
    logger.close()
    return final_metrics


def evaluate(config: dict, checkpoint: str | None = None) -> dict[str, float]:
    set_seed(config.get("seed", 0))
    device = torch.device(config.get("device", "cpu"))
    _, val_loader = create_dataloaders(config["data"], seed=config.get("seed", 0))
    model = build_model(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    criterion = CliffAwareLoss(
        contrastive_weight=config["training"].get("contrastive_weight", 0.5),
        margin=config["training"].get("margin", 1.0),
    )
    return run_epoch(model, val_loader, criterion, device)
