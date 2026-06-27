from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from src.data.preprocessing import canonicalize_smiles, molecule_fingerprint, pair_label, tanimoto_similarity


try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is dependency-gated.
    plt = None


@dataclass(frozen=True)
class CollapseAnalysisConfig:
    smiles_column: str = "smiles"
    target_column: str = "target_id"
    activity_column: str = "pIC50"
    tanimoto_threshold: float = 0.85
    cliff_activity_gap: float = 1.0
    non_cliff_activity_gap: float = 0.3
    random_pair_count: int = 1024
    random_seed: int = 17
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    figure_title: str = "Activity-sensitive representation collapse in existing molecular encoders"
    figure_name: str = "figure1_representation_collapse"


def require_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for Figure 1 generation. Install it with `pip install matplotlib`.")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_molecules_from_raw(input_csv: str | Path, config: CollapseAnalysisConfig) -> list[dict[str, Any]]:
    molecules = []
    for idx, row in enumerate(read_csv(input_csv)):
        canonical_smiles, mol = canonicalize_smiles(row[config.smiles_column])
        molecules.append(
            {
                "molecule_id": f"mol_{idx:06d}",
                "canonical_smiles": canonical_smiles,
                "target_id": row[config.target_column],
                "activity": float(row[config.activity_column]),
                "mol": mol,
            }
        )
    return molecules


def load_molecules_from_processed(processed_dir: str | Path) -> list[dict[str, Any]]:
    rows = read_csv(Path(processed_dir) / "molecules.csv")
    molecules = []
    for row in rows:
        _, mol = canonicalize_smiles(row["canonical_smiles"])
        molecules.append(
            {
                "molecule_id": row["molecule_id"],
                "canonical_smiles": row["canonical_smiles"],
                "target_id": row["target_id"],
                "activity": float(row["activity"]),
                "mol": mol,
            }
        )
    return molecules


def bit_vector_distance(fp_a: Any, fp_b: Any) -> float:
    return 1.0 - tanimoto_similarity(fp_a, fp_b)


def collapse_index(embedding_distance: float, activity_gap: float) -> float:
    if activity_gap <= 0:
        return math.nan
    return embedding_distance / activity_gap


def pair_category(similarity: float, activity_gap: float, config: CollapseAnalysisConfig) -> str:
    pair_config = type(
        "PairConfig",
        (),
        {
            "tanimoto_threshold": config.tanimoto_threshold,
            "cliff_activity_gap": config.cliff_activity_gap,
            "non_cliff_activity_gap": config.non_cliff_activity_gap,
        },
    )()
    label = pair_label(
        similarity,
        activity_gap,
        config=pair_config,
    )
    if label == "activity_cliff":
        return "cliff"
    if label == "similar_non_cliff":
        return "similar_non_cliff"
    return "similar_intermediate"


def build_pair_metrics(molecules: list[dict[str, Any]], config: CollapseAnalysisConfig) -> list[dict[str, Any]]:
    fps = [
        molecule_fingerprint(molecule["mol"], radius=config.fingerprint_radius, bits=config.fingerprint_bits)
        for molecule in molecules
    ]
    rng = random.Random(config.random_seed)
    rows = []
    random_candidates = []
    pair_index = 0

    for left, right in combinations(range(len(molecules)), 2):
        mol_a = molecules[left]
        mol_b = molecules[right]
        if mol_a["target_id"] != mol_b["target_id"]:
            continue
        similarity = tanimoto_similarity(fps[left], fps[right])
        distance = bit_vector_distance(fps[left], fps[right])
        activity_gap = abs(float(mol_a["activity"]) - float(mol_b["activity"]))
        category = pair_category(similarity, activity_gap, config)
        row = {
            "pair_id": f"pair_{pair_index:06d}",
            "molecule_id_a": mol_a["molecule_id"],
            "molecule_id_b": mol_b["molecule_id"],
            "target_id": mol_a["target_id"],
            "canonical_smiles_a": mol_a["canonical_smiles"],
            "canonical_smiles_b": mol_b["canonical_smiles"],
            "structural_similarity": round(similarity, 6),
            "embedding_distance": round(distance, 6),
            "activity_gap": round(activity_gap, 6),
            "collapse_index": round(collapse_index(distance, activity_gap), 6) if activity_gap > 0 else "",
            "pair_group": category,
        }
        pair_index += 1
        if category in {"cliff", "similar_non_cliff"}:
            rows.append(row)
        else:
            random_candidates.append({**row, "pair_group": "random"})

    rng.shuffle(random_candidates)
    rows.extend(random_candidates[: config.random_pair_count])
    return rows


def _values(rows: list[dict[str, Any]], group: str, key: str) -> list[float]:
    values = []
    for row in rows:
        if row["pair_group"] == group and row[key] != "":
            values.append(float(row[key]))
    return values


def _scatter(ax: Any, rows: list[dict[str, Any]], groups: list[str], colors: dict[str, str]) -> None:
    for group in groups:
        group_rows = [row for row in rows if row["pair_group"] == group]
        ax.scatter(
            [float(row["activity_gap"]) for row in group_rows],
            [float(row["embedding_distance"]) for row in group_rows],
            s=34,
            alpha=0.82,
            color=colors[group],
            label=group.replace("_", " "),
            linewidth=0,
        )
    ax.set_xlabel("Activity gap (log units)")
    ax.set_ylabel("Embedding distance")
    ax.set_title("Embedding distance stays small for cliffs")
    ax.legend(frameon=False)


def _boxplot(ax: Any, rows: list[dict[str, Any]], groups: list[str], colors: dict[str, str]) -> None:
    data = [_values(rows, group, "collapse_index") for group in groups]
    positions = list(range(1, len(groups) + 1))
    box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.78)
        patch.set_linewidth(0)
    for median in box["medians"]:
        median.set_color("#1f2933")
        median.set_linewidth(1.5)

    for pos, group_values, group in zip(positions, data, groups):
        jitter = [pos + (idx - len(group_values) / 2) * 0.012 for idx in range(len(group_values))]
        ax.scatter(jitter, group_values, s=22, alpha=0.62, color=colors[group], edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels([group.replace("_", "\n") for group in groups])
    ax.set_ylabel("Collapse index\nembedding distance / activity gap")
    ax.set_title("Lower index indicates activity-sensitive collapse")


def plot_figure1(rows: list[dict[str, Any]], output_dir: str | Path, config: CollapseAnalysisConfig) -> dict[str, Path]:
    require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = ["cliff", "similar_non_cliff", "random"]
    colors = {"cliff": "#c44536", "similar_non_cliff": "#2f6f73", "random": "#6b7280"}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    _scatter(axes[0], rows, groups, colors)
    _boxplot(axes[1], rows, groups, colors)
    fig.suptitle(config.figure_title, fontsize=10, fontweight="bold")

    pdf_path = output_dir / f"{config.figure_name}.pdf"
    svg_path = output_dir / f"{config.figure_name}.svg"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "svg": svg_path}


def analyze_representation_collapse(
    input_csv: str | Path | None,
    processed_dir: str | Path | None,
    output_dir: str | Path,
    config: CollapseAnalysisConfig,
) -> dict[str, Path]:
    if processed_dir is not None and (Path(processed_dir) / "molecules.csv").exists():
        molecules = load_molecules_from_processed(processed_dir)
    elif input_csv is not None:
        molecules = load_molecules_from_raw(input_csv, config)
    else:
        raise ValueError("Either input_csv or a processed_dir containing molecules.csv is required.")

    rows = build_pair_metrics(molecules, config)
    output_dir = Path(output_dir)
    metrics_path = output_dir / "figure1_pair_metrics.csv"
    fieldnames = [
        "pair_id",
        "molecule_id_a",
        "molecule_id_b",
        "target_id",
        "canonical_smiles_a",
        "canonical_smiles_b",
        "structural_similarity",
        "embedding_distance",
        "activity_gap",
        "collapse_index",
        "pair_group",
    ]
    write_csv(metrics_path, rows, fieldnames)
    figure_paths = plot_figure1(rows, output_dir, config)
    return {"metrics": metrics_path, **figure_paths}
