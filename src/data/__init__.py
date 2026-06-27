from .preprocessing import PreprocessConfig, preprocess_activity_cliffs

__all__ = ["MolecularPairDataset", "PreprocessConfig", "create_dataloaders", "preprocess_activity_cliffs"]


def __getattr__(name: str):
    if name in {"MolecularPairDataset", "create_dataloaders"}:
        from .synthetic import MolecularPairDataset, create_dataloaders

        return {"MolecularPairDataset": MolecularPairDataset, "create_dataloaders": create_dataloaders}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
