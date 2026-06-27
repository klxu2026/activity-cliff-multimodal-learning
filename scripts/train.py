from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the toy multimodal activity cliff model.")
    parser.add_argument("--config", default="configs/toy.yaml", help="Path to a YAML experiment config.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
