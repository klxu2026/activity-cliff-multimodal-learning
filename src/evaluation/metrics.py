from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    return {
        "mae": float(error.abs().mean().detach().cpu()),
        "rmse": float(error.pow(2).mean().sqrt().detach().cpu()),
    }
