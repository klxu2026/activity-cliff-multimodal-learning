# Activity Cliff Multimodal Learning

Minimal PyTorch scaffold for activity cliff-aware multimodal molecular representation learning.

The codebase is intentionally small and runnable on synthetic toy data. It is set up for future expansion toward graph neural networks, language-model SMILES encoders, molecular image encoders, and Nature Communications-style experiment tracking.

## Quick Start

```bash
pip install -r requirements.txt
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
