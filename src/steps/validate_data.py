from pathlib import Path

from zenml.steps import step


@step
def validate_data(data_root: str) -> str:
    folder = Path(data_root) / "cifar-10-batches-py"
    required = [
        folder / "data_batch_1",
        folder / "test_batch",
        folder / "batches.meta",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 files: {missing}")
    return data_root
