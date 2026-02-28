import os
import json
import logging
from pathlib import Path

import mlflow
from zenml.steps import step


def _store_monitoring(report: dict, decision: dict) -> str:
    for logger_name in (
        "mlflow",
        "mlflow.tracking._tracking_service.client",
        "mlflow.tracking.fluent",
        "mlflow.tracking._model_registry.client",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    out_dir = Path("monitoring")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.html"
    report_path.write_text(str(report.get("html", "")), encoding="utf-8")

    summary = {
        "drift_detected": bool(report.get("drift_detected", False)),
        "drift_score": report.get("drift_score"),
        "rows": int(report.get("rows", 0)),
        "action": decision.get("action", "no_action"),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("cifar10-monitoring")
        with mlflow.start_run(run_name="monitoring_pipeline"):
            mlflow.log_param("dataset_drift", summary["drift_detected"])
            mlflow.log_param("action", summary["action"])
            drift_score = summary.get("drift_score")
            if drift_score is not None:
                mlflow.log_metric("drift_score", float(drift_score))
            mlflow.log_metric("rows", float(summary["rows"]))
            mlflow.log_artifact(str(report_path), artifact_path="monitoring")
            mlflow.log_artifact(str(summary_path), artifact_path="monitoring")

    return str(report_path)


@step(enable_cache=False)
def store_monitoring_artifacts(report: dict, decision: dict) -> str:
    return _store_monitoring(report, decision)
