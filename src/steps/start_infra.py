from pathlib import Path

from zenml.steps import step


@step
def start_infra() -> str:
    """Prepare local tracking directories and return MLflow URI."""
    mlruns_dir = Path("mlruns").resolve()
    artifacts_dir = Path("artifacts")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.as_uri()
