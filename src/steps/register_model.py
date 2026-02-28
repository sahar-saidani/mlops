import json
import os
import logging
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
import mlflow.pytorch
import torch
from zenml.steps import step

from src.steps.train_cnn_model import TinyCNN


@step(enable_cache=False)
def register_model(
    model_path: str,
    metrics: dict,
    training_params_json: str,
    model_name: str = "cifar10-cnn",
) -> str:
    for logger_name in (
        "mlflow",
        "mlflow.pytorch",
        "mlflow.tracking._tracking_service.client",
        "mlflow.tracking.fluent",
        "mlflow.tracking._model_registry.client",
        "mlflow.store.model_registry.abstract_store",
    ):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("cifar10-training")
    training_params = json.loads(training_params_json)

    model = TinyCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with mlflow.start_run(run_name="training_pipeline") as run:
        mlflow.log_params({k: v for k, v in training_params.items() if k != "best_val_accuracy"})
        mlflow.log_metrics(metrics)

        if Path("artifacts/metrics").exists():
            mlflow.log_artifacts("artifacts/metrics", artifact_path="metrics")

        try:
            mlflow.pytorch.log_model(model, name="model", registered_model_name=model_name)
        except MlflowException:
            mlflow.pytorch.log_model(model, name="model")
        return run.info.run_id
