# Activity Cliff Multimodal Learning

Minimal PyTorch scaffold for activity cliff-aware multimodal molecular representation learning.

The codebase is intentionally small and runnable on synthetic toy data. It is set up for future expansion toward graph neural networks, language-model SMILES encoders, molecular image encoders, and Nature Communications-style experiment tracking.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/preprocess_activity_cliffs.py --config configs/preprocess_toy.yaml
python scripts/make_figure1_collapse.py --config configs/figure1.yaml
python scripts/train.py --config configs/toy.yaml
python scripts/evaluate.py --config configs/toy.yaml --checkpoint results/toy_run/model.pt
pytest
```

Training writes CSV metrics to `results/toy_run/metrics.csv` and TensorBoard logs to `results/toy_run/tensorboard/`.

## Current Scaffold

- Synthetic molecular pair dataset with graph tensors, tokenized SMILES, 2D image tensors, activity deltas, and cliff labels.
- Tiny graph, SMILES, and 2D image encoders.
- Concatenation-based multimodal fusion module.
- Activity delta regression head.
- Cliff-aware contrastive loss that pulls non-cliff pairs together and pushes cliffs apart.
- YAML-driven train/evaluate scripts.
- CSV logging and optional TensorBoard logging.
- Unit tests for data loading, model forward pass, loss computation, and toy training.

## Activity Cliff Preprocessing

The preprocessing entry point converts assay tables into activity cliff learning artifacts:

```bash
python scripts/preprocess_activity_cliffs.py --config configs/preprocess_toy.yaml
```

Input rows must include molecular SMILES, target ID, and an activity value such as pIC50. Additional assay metadata columns are preserved in `molecules.csv`.

Generated outputs include:

- `molecules.csv`: canonical SMILES, randomized SMILES, Murcko scaffold, graph path, image path, and scaffold split.
- `pairs.csv`: matched molecular pairs within each target, Tanimoto similarity, activity gap, cliff labels, and split.
- `splits.csv`: molecule-level train/val/test assignment grouped by target and scaffold.
- `graphs/*.json`: atom and bond features for graph models.
- `images/*.png`: 2D molecular depictions.

Definitions:

- Activity cliff pair: Tanimoto similarity >= 0.85 and activity difference >= 1.0 log unit.
- Similar non-cliff pair: Tanimoto similarity >= 0.85 and activity difference < 0.3 log unit.

## Figure 1: Representation Collapse

The Figure 1 analysis tests whether existing molecular encoders place activity cliff pairs too close in representation space. It computes:

- structural similarity,
- embedding distance,
- activity gap,
- cliff collapse index = embedding distance / activity gap.

Run:

```bash
python scripts/make_figure1_collapse.py --config configs/figure1.yaml
```

Outputs are written to `figures/figure1/`:

- `figure1_pair_metrics.csv`
- `figure1_representation_collapse.pdf`
- `figure1_representation_collapse.svg`

The first scaffold uses RDKit Morgan fingerprints as the baseline existing molecular encoder. A low collapse index for cliff pairs indicates activity-sensitive representation collapse: structurally similar molecules with large activity differences remain too close in the encoder space.

## Structure

- `configs/`: YAML experiment configs.
- `data/raw/`: original input datasets.
- `data/processed/`: cleaned and featurized datasets.
- `data/splits/`: train, validation, and test split files.
- `src/data/`: data loading and preprocessing code.
- `src/models/`: model definitions and multimodal fusion.
- `src/losses/`: training objectives and loss functions.
- `src/training/`: training loops, evaluation runner, and logging.
- `src/evaluation/`: metric utilities.
- `src/generation/`: placeholder for candidate generation code.
- `scripts/`: command-line entry points.
- `tests/`: unit and smoke tests.
- `notebooks/`: exploratory notebooks.
- `results/`: experiment outputs.
- `figures/`: generated plots and paper figures.
- `paper/`: manuscript and related writing assets.
