from pathlib import Path

import torchvision.datasets as datasets


def main() -> None:
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)
    print("CIFAR-10 downloaded in data/raw")


if __name__ == "__main__":
    main()
