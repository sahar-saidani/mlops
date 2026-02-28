import json
from pathlib import Path

from zenml.steps import step


@step
def preprocess(train_idx_path: str, val_idx_path: str, test_idx_path: str) -> str:
    config = {
        "train_idx_path": train_idx_path,
        "val_idx_path": val_idx_path,
        "test_idx_path": test_idx_path,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
    }

    out_dir = Path("artifacts/preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return str(cfg_path)
