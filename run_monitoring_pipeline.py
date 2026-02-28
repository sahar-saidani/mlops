from __future__ import annotations

import json
import logging
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


def _print_monitoring_summary(start_ts: float) -> None:
    candidates = [
        Path("monitoring/collected_inference_log.csv"),
        Path("monitoring/report.html"),
        Path("monitoring/summary.json"),
    ]
    changed = [p for p in candidates if p.exists() and p.stat().st_mtime >= start_ts - 1.0]

    print("\n=== Monitoring Outputs ===")
    if changed:
        for p in changed:
            print(f"- {p.as_posix()}")
    else:
        print("- No new/updated tracked files detected.")

    summary_path = Path("monitoring/summary.json")
    print("\n=== Monitoring Metrics ===")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"- drift_detected: {summary.get('drift_detected')}")
        print(f"- drift_score: {summary.get('drift_score')}")
        print(f"- rows: {summary.get('rows')}")
        print(f"- action: {summary.get('action')}")
    else:
        print("- summary.json not found")


def main() -> None:
    _configure_quiet_logging()
    try:
        from src.pipelines.monitoring_pipeline import monitoring_pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "zenml":
            raise ModuleNotFoundError(
                "zenml is not installed in the current Python environment."
            ) from exc
        raise
    start_ts = time.time()
    monitoring_pipeline()
    _print_monitoring_summary(start_ts)


if __name__ == "__main__":
    main()
