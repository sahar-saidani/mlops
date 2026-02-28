from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
def train_cnn_model(dataset_path: str, mlflow_tracking_uri: str) -> Tuple[str, float]:
    """Train a tiny CNN quickly and log metrics/model to MLflow."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("zenml-fast-cnn")

    data = np.load(dataset_path)
    x_train = torch.tensor(data["X_train"]).unsqueeze(1)
    y_train = torch.tensor(data["y_train"])
    x_test = torch.tensor(data["X_test"]).unsqueeze(1)
    y_test = torch.tensor(data["y_test"])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 40
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
