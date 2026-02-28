import os
from pathlib import Path

import mlflow
from zenml.steps import step


@step
def store_monitoring_artifacts(html: str, drift: bool) -> str:
    out_dir = Path("monitoring")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("cifar10-monitoring")
        with mlflow.start_run(run_name="monitoring_pipeline"):
            mlflow.log_param("dataset_drift", drift)
            mlflow.log_artifact(str(report_path), artifact_path="monitoring")

    return str(report_path)
