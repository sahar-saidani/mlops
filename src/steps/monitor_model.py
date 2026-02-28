from pathlib import Path

import mlflow
from zenml.steps import step


@step
def monitor_model(model_path: str, accuracy: float, mlflow_tracking_uri: str) -> str:
    """Log a simple monitoring signal and save status output."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("zenml-fast-cnn-monitoring")

    status = "healthy" if accuracy >= 0.60 else "needs_retraining"

    with mlflow.start_run(run_name="monitor"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("observed_accuracy", accuracy)
        mlflow.log_metric("threshold", 0.60)
        mlflow.log_param("status", status)

    out_path = Path("artifacts") / "monitoring_status.txt"
    out_path.write_text(f"status={status}\naccuracy={accuracy:.4f}\n", encoding="utf-8")
    return str(out_path)
