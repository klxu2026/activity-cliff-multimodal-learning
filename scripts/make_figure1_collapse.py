from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.analysis import analyze_representation_collapse
from src.analysis.representation_collapse import CollapseAnalysisConfig


def load_config(path: str | Path) -> tuple[Path | None, Path | None, Path, CollapseAnalysisConfig]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    columns = raw.get("columns", {})
    pairing = raw.get("pairing", {})
    encoder = raw.get("encoder", {})
    plot = raw.get("plot", {})
    config = CollapseAnalysisConfig(
        smiles_column=columns.get("smiles", "smiles"),
        target_column=columns.get("target_id", "target_id"),
        activity_column=columns.get("activity", "pIC50"),
        tanimoto_threshold=pairing.get("tanimoto_threshold", 0.85),
        cliff_activity_gap=pairing.get("cliff_activity_gap", 1.0),
        non_cliff_activity_gap=pairing.get("non_cliff_activity_gap", 0.3),
        random_pair_count=pairing.get("random_pair_count", 1024),
        random_seed=pairing.get("random_seed", 17),
        fingerprint_radius=encoder.get("fingerprint_radius", 2),
        fingerprint_bits=encoder.get("fingerprint_bits", 2048),
        figure_title=plot.get("title", "Activity-sensitive representation collapse in existing molecular encoders"),
        figure_name=plot.get("figure_name", "figure1_representation_collapse"),
    )
    input_csv = Path(raw["input_csv"]) if raw.get("input_csv") else None
    processed_dir = Path(raw["processed_dir"]) if raw.get("processed_dir") else None
    output_dir = Path(raw.get("output_dir", "figures/figure1"))
    return input_csv, processed_dir, output_dir, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1 representation collapse analysis.")
    parser.add_argument("--config", default="configs/figure1.yaml", help="YAML Figure 1 config.")
    args = parser.parse_args()

    input_csv, processed_dir, output_dir, config = load_config(args.config)
    outputs = analyze_representation_collapse(input_csv, processed_dir, output_dir, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
