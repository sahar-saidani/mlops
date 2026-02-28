import json
from pathlib import Path

from zenml.steps import step

TRAIN_SAMPLES = 50_000
TEST_SAMPLES = 10_000


@step(enable_cache=False)
def split_data(data_root: str) -> tuple[str, str, str]:
    """Use the full CIFAR-10 split: all train samples, no validation split, all test samples."""
    cifar_dir = Path(data_root) / "cifar-10-batches-py"
    required = [cifar_dir / f"data_batch_{i}" for i in range(1, 6)] + [cifar_dir / "test_batch"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CIFAR-10 files: {missing}")

    train_idx = list(range(TRAIN_SAMPLES))
    val_idx: list[int] = []
    test_idx = list(range(TEST_SAMPLES))

    out_dir = Path("artifacts/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_idx.json"
    val_path = out_dir / "val_idx.json"
    test_path = out_dir / "test_idx.json"

    train_path.write_text(json.dumps(train_idx), encoding="utf-8")
    val_path.write_text(json.dumps(val_idx), encoding="utf-8")
    test_path.write_text(json.dumps(test_idx), encoding="utf-8")

    return str(train_path), str(val_path), str(test_path)
