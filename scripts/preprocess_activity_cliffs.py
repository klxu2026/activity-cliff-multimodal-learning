from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.preprocessing import PreprocessConfig, preprocess_activity_cliffs


def load_config(path: str | Path) -> tuple[Path, Path, PreprocessConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    columns = raw.get("columns", {})
    pairing = raw.get("pairing", {})
    features = raw.get("features", {})
    splits = raw.get("splits", {})
    config = PreprocessConfig(
        smiles_column=columns.get("smiles", "smiles"),
        target_column=columns.get("target_id", "target_id"),
        activity_column=columns.get("activity", "pIC50"),
        tanimoto_threshold=pairing.get("tanimoto_threshold", 0.85),
        cliff_activity_gap=pairing.get("cliff_activity_gap", 1.0),
        non_cliff_activity_gap=pairing.get("non_cliff_activity_gap", 0.3),
        fingerprint_radius=pairing.get("fingerprint_radius", 2),
        fingerprint_bits=pairing.get("fingerprint_bits", 2048),
        randomized_smiles=features.get("randomized_smiles", 3),
        image_size=tuple(features.get("image_size", [256, 256])),
        train_fraction=splits.get("train_fraction", 0.8),
        val_fraction=splits.get("val_fraction", 0.1),
        test_fraction=splits.get("test_fraction", 0.1),
        seed=splits.get("seed", 13),
    )
    return Path(raw["input_csv"]), Path(raw["output_dir"]), config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess molecular activity data into activity cliff pairs.")
    parser.add_argument("--config", default="configs/preprocess_toy.yaml", help="YAML preprocessing config.")
    args = parser.parse_args()

    input_csv, output_dir, config = load_config(args.config)
    outputs = preprocess_activity_cliffs(input_csv, output_dir, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
