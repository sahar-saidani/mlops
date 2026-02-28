import json

from zenml.steps import step


@step
def build_training_params(
    mlflow_tracking_uri: str,
    best_val_accuracy: float,
    source_dataset: str = "sklearn_digits",
    evaluation_dataset: str = "cifar10",
) -> str:
    return json.dumps(
        {
            "source_dataset": source_dataset,
            "evaluation_dataset": evaluation_dataset,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "best_val_accuracy": float(best_val_accuracy),
        }
    )
