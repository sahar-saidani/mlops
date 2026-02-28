import subprocess
from pathlib import Path

import torchvision.datasets as datasets
from zenml.steps import step


@step
def ingest_data() -> str:
    data_root = Path("data/raw/cifar-10-batches-py")

    if not data_root.exists():
        pull = subprocess.run(["python", "scripts/pull_cifar_with_dvc.py"], check=False)
        if pull.returncode != 0:
            datasets.CIFAR10(root="data/raw", train=True, download=True)
            datasets.CIFAR10(root="data/raw", train=False, download=True)

    if not data_root.exists():
        raise FileNotFoundError("CIFAR-10 dataset missing in data/raw.")

    return "data/raw"
