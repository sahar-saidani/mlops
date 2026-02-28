from pathlib import Path

import mlflow
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from zenml.steps import step


@step
def pull_data(mlflow_tracking_uri: str) -> str:
    """Load sklearn digits dataset and persist train/test arrays."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    digits = load_digits()
    x = (digits.images / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    dataset_path = Path("artifacts") / "digits_dataset.npz"
    np.savez(dataset_path, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
    return str(dataset_path)
