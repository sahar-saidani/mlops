import subprocess
from pathlib import Path


def run(cmd: list[str]) -> int:
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    raw_dvc = Path("data/raw.dvc")
    raw_dir = Path("data/raw/cifar-10-batches-py")

    if raw_dvc.exists():
        code = run(["dvc", "pull", "data/raw.dvc"])
        if code == 0 and raw_dir.exists():
            print("CIFAR-10 pulled from DVC remote.")
            return

    print("DVC pull unavailable or data missing. Seeding remote from source dataset.")
    code = run(["python", "scripts/prepare_cifar_for_dvc.py"])
    if code != 0:
        raise RuntimeError("Failed to download CIFAR-10.")

    code = run(["dvc", "add", "data/raw"])
    if code != 0:
        raise RuntimeError("Failed to track data/raw with DVC.")

    code = run(["dvc", "push"])
    if code != 0:
        raise RuntimeError("Failed to push data to DVC remote (MinIO).")

    print("CIFAR-10 prepared and pushed to MinIO-backed DVC remote.")


if __name__ == "__main__":
    main()
