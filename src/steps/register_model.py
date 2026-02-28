import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
import mlflow.pytorch
import torch
from zenml.steps import step

from src.models.cnn import CNN


@step
def register_model(model_path: str, metrics: dict, training_params: dict, model_name: str = "cifar10-cnn") -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("cifar10-training")

    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with mlflow.start_run(run_name="training_pipeline") as run:
        mlflow.log_params({k: v for k, v in training_params.items() if k != "best_val_accuracy"})
        mlflow.log_metrics(metrics)

        if Path("artifacts/metrics").exists():
            mlflow.log_artifacts("artifacts/metrics", artifact_path="metrics")

        try:
            mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=model_name)
        except MlflowException:
            mlflow.pytorch.log_model(model, artifact_path="model")
        return run.info.run_id
