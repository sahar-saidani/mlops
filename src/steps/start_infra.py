import os
from pathlib import Path

from zenml.steps import step


@step
def start_infra() -> str:
    """Prepare local tracking directories and return MLflow URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        Path("artifacts").mkdir(parents=True, exist_ok=True)
        return tracking_uri

    mlruns_dir = Path("mlruns").resolve()
    artifacts_dir = Path("artifacts")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.as_uri()
