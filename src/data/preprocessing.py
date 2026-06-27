from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover - exercised in environments without RDKit.
    Chem = None
    DataStructs = None
    AllChem = None
    Draw = None
    MurckoScaffold = None


@dataclass(frozen=True)
class PreprocessConfig:
    smiles_column: str = "smiles"
    target_column: str = "target_id"
    activity_column: str = "pIC50"
    tanimoto_threshold: float = 0.85
    cliff_activity_gap: float = 1.0
    non_cliff_activity_gap: float = 0.3
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    randomized_smiles: int = 3
    image_size: tuple[int, int] = (256, 256)
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 13


def require_rdkit() -> None:
    if Chem is None:
        raise ImportError("RDKit is required for preprocessing. Install it with `pip install rdkit` or via conda.")


def canonicalize_smiles(smiles: str) -> tuple[str, Any]:
    require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True), mol


def randomize_smiles(mol: Any, count: int, seed: int) -> list[str]:
    require_rdkit()
    rng = random.Random(seed)
    randomized = []
    for _ in range(count):
        atom_order = list(range(mol.GetNumAtoms()))
        rng.shuffle(atom_order)
        renumbered = Chem.RenumberAtoms(mol, atom_order)
        randomized.append(Chem.MolToSmiles(renumbered, canonical=False, doRandom=True))
    return randomized


def murcko_scaffold_smiles(mol: Any) -> str:
    require_rdkit()
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold or Chem.MolToSmiles(mol, canonical=True)


def molecule_fingerprint(mol: Any, radius: int, bits: int) -> Any:
    require_rdkit()
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)


def tanimoto_similarity(fp_a: Any, fp_b: Any) -> float:
    require_rdkit()
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def atom_features(atom: Any) -> dict[str, Any]:
    return {
        "atomic_num": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "is_aromatic": atom.GetIsAromatic(),
        "hybridization": str(atom.GetHybridization()),
        "num_h": atom.GetTotalNumHs(),
    }


def graph_features(mol: Any) -> dict[str, Any]:
    nodes = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge = {"source": begin, "target": end, "bond_type": str(bond.GetBondType()), "is_conjugated": bond.GetIsConjugated()}
        edges.append(edge)
        edges.append({**edge, "source": end, "target": begin})
    return {"nodes": nodes, "edges": edges}


def save_molecule_image(mol: Any, path: Path, size: tuple[int, int]) -> None:
    require_rdkit()
    path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(mol, str(path), size=size)


def assign_scaffold_splits(rows: list[dict[str, Any]], config: PreprocessConfig) -> dict[str, str]:
    rng = random.Random(config.seed)
    split_by_key: dict[str, str] = {}
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["target_id"]), []).append(str(row["scaffold"]))

    for target_id, scaffolds in grouped.items():
        unique_scaffolds = sorted(set(scaffolds))
        rng.shuffle(unique_scaffolds)
        n_total = len(unique_scaffolds)
        n_train = int(n_total * config.train_fraction)
        n_val = int(n_total * config.val_fraction)
        if n_total > 0 and n_train == 0:
            n_train = 1
        for idx, scaffold in enumerate(unique_scaffolds):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_by_key[f"{target_id}::{scaffold}"] = split
    return split_by_key


def pair_label(similarity: float, activity_gap: float, config: PreprocessConfig) -> str:
    if similarity < config.tanimoto_threshold:
        return "dissimilar"
    if activity_gap >= config.cliff_activity_gap:
        return "activity_cliff"
    if activity_gap < config.non_cliff_activity_gap:
        return "similar_non_cliff"
    return "similar_intermediate"


def read_input_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_activity_cliffs(input_csv: str | Path, output_dir: str | Path, config: PreprocessConfig) -> dict[str, Path]:
    require_rdkit()
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    graphs_dir = output_dir / "graphs"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_input_csv(input_csv)
    molecules: list[dict[str, Any]] = []
    fps: list[Any] = []
    metadata_columns = [
        col
        for col in (raw_rows[0].keys() if raw_rows else [])
        if col not in {config.smiles_column, config.target_column, config.activity_column}
    ]

    for idx, raw in enumerate(raw_rows):
        canonical_smiles, mol = canonicalize_smiles(raw[config.smiles_column])
        activity = float(raw[config.activity_column])
        target_id = raw[config.target_column]
        mol_id = f"mol_{idx:06d}"
        graph_path = graphs_dir / f"{mol_id}.json"
        image_path = images_dir / f"{mol_id}.png"
        graph_path.write_text(json.dumps(graph_features(mol), indent=2), encoding="utf-8")
        save_molecule_image(mol, image_path, config.image_size)

        randomized = randomize_smiles(mol, config.randomized_smiles, seed=config.seed + idx)
        scaffold = murcko_scaffold_smiles(mol)
        row = {
            "molecule_id": mol_id,
            "input_smiles": raw[config.smiles_column],
            "canonical_smiles": canonical_smiles,
            "randomized_smiles": json.dumps(randomized),
            "target_id": target_id,
            "activity": activity,
            "scaffold": scaffold,
            "graph_path": graph_path.as_posix(),
            "image_path": image_path.as_posix(),
        }
        for col in metadata_columns:
            row[col] = raw.get(col, "")
        molecules.append(row)
        fps.append(molecule_fingerprint(mol, config.fingerprint_radius, config.fingerprint_bits))

    split_lookup = assign_scaffold_splits(molecules, config)
    for row in molecules:
        row["split"] = split_lookup[f"{row['target_id']}::{row['scaffold']}"]

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(range(len(molecules)), 2):
        mol_a = molecules[left]
        mol_b = molecules[right]
        if mol_a["target_id"] != mol_b["target_id"]:
            continue
        if mol_a["split"] != mol_b["split"]:
            continue
        similarity = tanimoto_similarity(fps[left], fps[right])
        activity_gap = abs(float(mol_a["activity"]) - float(mol_b["activity"]))
        label = pair_label(similarity, activity_gap, config)
        if label == "dissimilar":
            continue
        pairs.append(
            {
                "pair_id": f"pair_{len(pairs):06d}",
                "molecule_id_a": mol_a["molecule_id"],
                "molecule_id_b": mol_b["molecule_id"],
                "target_id": mol_a["target_id"],
                "split": mol_a["split"],
                "tanimoto_similarity": round(similarity, 6),
                "activity_a": mol_a["activity"],
                "activity_b": mol_b["activity"],
                "activity_gap": round(activity_gap, 6),
                "pair_label": label,
                "is_cliff": int(label == "activity_cliff"),
                "is_similar_non_cliff": int(label == "similar_non_cliff"),
            }
        )

    molecule_fields = [
        "molecule_id",
        "input_smiles",
        "canonical_smiles",
        "randomized_smiles",
        "target_id",
        "activity",
        "scaffold",
        "split",
        "graph_path",
        "image_path",
        *metadata_columns,
    ]
    pair_fields = [
        "pair_id",
        "molecule_id_a",
        "molecule_id_b",
        "target_id",
        "split",
        "tanimoto_similarity",
        "activity_a",
        "activity_b",
        "activity_gap",
        "pair_label",
        "is_cliff",
        "is_similar_non_cliff",
    ]
    split_fields = ["molecule_id", "target_id", "scaffold", "split"]

    molecules_csv = output_dir / "molecules.csv"
    pairs_csv = output_dir / "pairs.csv"
    splits_csv = output_dir / "splits.csv"
    write_csv(molecules_csv, molecules, molecule_fields)
    write_csv(pairs_csv, pairs, pair_fields)
    write_csv(splits_csv, [{key: row[key] for key in split_fields} for row in molecules], split_fields)
    return {"molecules": molecules_csv, "pairs": pairs_csv, "splits": splits_csv, "graphs": graphs_dir, "images": images_dir}
