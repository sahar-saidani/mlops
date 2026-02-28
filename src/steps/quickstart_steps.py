from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from zenml.steps import step


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@step
def start_infra() -> str:
    """Prepare local tracking directories and return MLflow URI."""
    mlruns_dir = Path("mlruns").resolve()
    artifacts_dir = Path("artifacts")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return mlruns_dir.as_uri()


@step
def pull_data(mlflow_tracking_uri: str) -> str:
    """Load sklearn digits dataset and persist train/test arrays."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    dataset_path = Path("artifacts") / "digits_dataset.npz"
    np.savez(dataset_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return str(dataset_path)


@step
def train_cnn_model(dataset_path: str, mlflow_tracking_uri: str) -> Tuple[str, float]:
    """Train a tiny CNN quickly and log metrics/model to MLflow."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("zenml-fast-cnn")

    data = np.load(dataset_path)
    X_train = torch.tensor(data["X_train"]).unsqueeze(1)
    y_train = torch.tensor(data["y_train"])
    X_test = torch.tensor(data["X_test"]).unsqueeze(1)
    y_test = torch.tensor(data["y_test"])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3
    with mlflow.start_run(run_name="tiny-cnn"):
        mlflow.log_params({"epochs": epochs, "batch_size": 128, "lr": 1e-3})

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        accuracy = correct / max(total, 1)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.pytorch.log_model(model, artifact_path="model")

    model_path = Path("artifacts") / "tiny_cnn.pt"
    torch.save(model.state_dict(), model_path)
    return str(model_path), float(accuracy)


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
