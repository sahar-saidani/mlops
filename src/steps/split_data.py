import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from zenml.steps import step


@step
def split_data(data_root: str, seed: int = 42) -> tuple[str, str, str]:
    indices = list(range(50000))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=seed, shuffle=True)
    test_idx = list(range(10000))

    out_dir = Path("artifacts/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_idx.json"
    val_path = out_dir / "val_idx.json"
    test_path = out_dir / "test_idx.json"

    train_path.write_text(json.dumps(train_idx), encoding="utf-8")
    val_path.write_text(json.dumps(val_idx), encoding="utf-8")
    test_path.write_text(json.dumps(test_idx), encoding="utf-8")

    return str(train_path), str(val_path), str(test_path)
