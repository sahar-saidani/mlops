import json
import logging
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as T
from zenml.steps import step


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@step
def train_cnn_model(data_root: str, preprocess_cfg_path: str, mlflow_tracking_uri: str) -> Tuple[str, float]:
    """Train TinyCNN on CIFAR-10 and log metrics/model to MLflow."""
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
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("zenml-fast-cnn")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = json.loads(Path(preprocess_cfg_path).read_text(encoding="utf-8"))
    train_idx = json.loads(Path(cfg["train_idx_path"]).read_text(encoding="utf-8"))
    val_idx = json.loads(Path(cfg["val_idx_path"]).read_text(encoding="utf-8"))
    mean = tuple(cfg["mean"])
    std = tuple(cfg["std"])

    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    full_train_ds = datasets.CIFAR10(root=data_root, train=True, transform=train_tf, download=False)
    train_ds = Subset(full_train_ds, train_idx)
    val_loader = None
    if len(val_idx) > 0:
        full_val_ds = datasets.CIFAR10(root=data_root, train=True, transform=eval_tf, download=False)
        val_ds = Subset(full_val_ds, val_idx)

    batch_size = 64
    num_workers = 2
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if len(val_idx) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=256,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 25
    best_val_accuracy = 0.0
    with mlflow.start_run(run_name="tiny-cnn"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": 1e-3,
                "dataset": "cifar10",
                "model_variant": "tinycnn_minimal",
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
            }
        )

        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)

            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = model(xb).argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                val_accuracy = correct / max(total, 1)
            else:
                val_accuracy = train_correct / max(train_total, 1)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_path = Path("artifacts") / "tiny_cnn.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)

        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.pytorch.log_model(model, name="model")

    model_path = Path("artifacts") / "tiny_cnn.pt"
    if not model_path.exists():
        torch.save(model.state_dict(), model_path)
    return str(model_path), float(best_val_accuracy)
