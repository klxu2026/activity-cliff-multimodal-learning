from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def cliff_contrastive_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    is_cliff: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pull non-cliff pairs together and push activity cliffs at least `margin` apart."""

    distance = F.pairwise_distance(z_a, z_b)
    non_cliff = 1.0 - is_cliff
    pull = non_cliff * distance.pow(2)
    push = is_cliff * F.relu(margin - distance).pow(2)
    return (pull + push).mean()


class CliffAwareLoss(nn.Module):
    def __init__(self, contrastive_weight: float = 0.5, margin: float = 1.0) -> None:
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.margin = margin

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        regression = F.mse_loss(outputs["activity_delta"], batch["activity_delta"])
        contrastive = cliff_contrastive_loss(outputs["z_a"], outputs["z_b"], batch["is_cliff"], self.margin)
        total = regression + self.contrastive_weight * contrastive
        return {"loss": total, "regression_loss": regression, "contrastive_loss": contrastive}
