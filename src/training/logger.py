from __future__ import annotations

import csv
from pathlib import Path


class ExperimentLogger:
    """CSV logger with optional TensorBoard support."""

    def __init__(self, output_dir: str, csv_name: str = "metrics.csv", use_tensorboard: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / csv_name
        self.writer = None
        self.fieldnames: list[str] | None = None
        self.tb = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except Exception:
                self.tb = None

    def log(self, step: int, metrics: dict[str, float], prefix: str) -> None:
        row = {"step": step, "split": prefix, **metrics}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            self.writer = self.csv_path.open("w", newline="", encoding="utf-8")
            csv.DictWriter(self.writer, fieldnames=self.fieldnames).writeheader()

        assert self.writer is not None and self.fieldnames is not None
        csv_writer = csv.DictWriter(self.writer, fieldnames=self.fieldnames)
        csv_writer.writerow(row)
        self.writer.flush()

        if self.tb is not None:
            for key, value in metrics.items():
                self.tb.add_scalar(f"{prefix}/{key}", value, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.writer is not None:
            self.writer.close()
