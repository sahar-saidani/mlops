import subprocess
from pathlib import Path

import torchvision.datasets as datasets
from zenml.steps import step


@step
def ingest_data() -> str:
    base_root = Path("data/raw")
    data_root = base_root / "cifar-10-batches-py"

    if not data_root.exists():
        pull = subprocess.run(["python", "scripts/pull_cifar_with_dvc.py"], check=False)
        if pull.returncode != 0:
            datasets.CIFAR10(root=str(base_root), train=True, download=True)
            datasets.CIFAR10(root=str(base_root), train=False, download=True)

    if not data_root.exists():
        raise FileNotFoundError("CIFAR-10 dataset missing in data/raw.")

    return str(base_root.resolve())
