from __future__ import annotations

import logging
import re
import time
import warnings
from pathlib import Path


def _configure_quiet_logging() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for logger_name in (
        "mlflow",
        "mlflow.tracking",
        "mlflow.tracking._tracking_service.client",
        "mlflow.tracking.fluent",
        "mlflow.tracking._model_registry.client",
        "mlflow.utils.git_utils",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    logging.getLogger("git").setLevel(logging.ERROR)


def _extract_test_accuracy(report_path: Path) -> str | None:
    if not report_path.exists():
        return None
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+\d+\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def _extract_val_accuracy(status_path: Path) -> str | None:
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("accuracy="):
            return line.split("=", 1)[1].strip()
    return None


def _print_training_summary(start_ts: float) -> None:
    candidates = [
        Path("artifacts/tiny_cnn.pt"),
        Path("artifacts/splits/train_idx.json"),
        Path("artifacts/splits/val_idx.json"),
        Path("artifacts/splits/test_idx.json"),
        Path("artifacts/preprocess/config.json"),
        Path("artifacts/metrics/confusion_matrix.png"),
        Path("artifacts/metrics/classification_report.txt"),
        Path("artifacts/monitoring_status.txt"),
        Path("monitoring/reference.csv"),
        Path("monitoring/inference_log.csv"),
    ]
    changed = [p for p in candidates if p.exists() and p.stat().st_mtime >= start_ts - 1.0]

    print("\n=== Training Outputs ===")
    if changed:
        for p in changed:
            print(f"- {p.as_posix()}")
    else:
        print("- No new/updated tracked files detected.")

    val_acc = _extract_val_accuracy(Path("artifacts/monitoring_status.txt"))
    test_acc = _extract_test_accuracy(Path("artifacts/metrics/classification_report.txt"))
    print("\n=== Training Metrics ===")
    if val_acc is not None:
        print(f"- val_accuracy: {val_acc}")
    else:
        print("- val_accuracy: not found")
    if test_acc is not None:
        print(f"- test_accuracy: {test_acc}")
    else:
        print("- test_accuracy: not found")


def main() -> None:
    _configure_quiet_logging()
    try:
        from src.pipelines.training_pipeline import training_pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "zenml":
            raise ModuleNotFoundError(
                "zenml is not installed in the current Python environment."
            ) from exc
        raise
    start_ts = time.time()
    training_pipeline()
    _print_training_summary(start_ts)


if __name__ == "__main__":
    main()
