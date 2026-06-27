import csv
import importlib.util

import pytest

from src.data.preprocessing import PreprocessConfig, pair_label, preprocess_activity_cliffs


pytestmark = pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="RDKit is not installed")


def test_pair_label_definitions():
    cfg = PreprocessConfig()
    assert pair_label(0.90, 1.0, cfg) == "activity_cliff"
    assert pair_label(0.90, 0.29, cfg) == "similar_non_cliff"
    assert pair_label(0.90, 0.5, cfg) == "similar_intermediate"
    assert pair_label(0.50, 2.0, cfg) == "dissimilar"


def test_preprocess_activity_cliffs_outputs(tmp_path):
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        "\n".join(
            [
                "smiles,target_id,pIC50,assay",
                "CCO,T1,5.0,binding",
                "CCO,T1,6.2,binding",
                "CCN,T1,5.1,binding",
                "CCN,T1,5.2,binding",
            ]
        ),
        encoding="utf-8",
    )
    cfg = PreprocessConfig(randomized_smiles=2, image_size=(128, 128), train_fraction=1.0, val_fraction=0.0, test_fraction=0.0)
    outputs = preprocess_activity_cliffs(input_csv, tmp_path / "processed", cfg)

    assert outputs["molecules"].exists()
    assert outputs["pairs"].exists()
    assert outputs["splits"].exists()
    assert any(outputs["graphs"].glob("*.json"))
    assert any(outputs["images"].glob("*.png"))

    with outputs["pairs"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["pair_label"] for row in rows}
    assert "activity_cliff" in labels
    assert "similar_non_cliff" in labels
