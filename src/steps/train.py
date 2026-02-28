import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as T
from zenml.steps import step

from src.models.cnn import CNN


@step
def train(preprocess_cfg_path: str) -> tuple[str, dict]:
    cfg = json.loads(Path(preprocess_cfg_path).read_text(encoding="utf-8"))
    train_idx = json.loads(Path(cfg["train_idx_path"]).read_text(encoding="utf-8"))
    val_idx = json.loads(Path(cfg["val_idx_path"]).read_text(encoding="utf-8"))

    epochs = int(os.getenv("EPOCHS", "3"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr = float(os.getenv("LR", "0.001"))
    num_workers = int(os.getenv("NUM_WORKERS", "0"))
    max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", "12000"))
    max_val_samples = int(os.getenv("MAX_VAL_SAMPLES", "2000"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(tuple(cfg["mean"]), tuple(cfg["std"])),
        ]
    )
    eval_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(tuple(cfg["mean"]), tuple(cfg["std"])),
        ]
    )

    train_ds = datasets.CIFAR10(root="data/raw", train=True, transform=train_tf, download=False)
    val_ds = datasets.CIFAR10(root="data/raw", train=True, transform=eval_tf, download=False)

    train_subset = train_idx[:max_train_samples]
    val_subset = val_idx[:max_val_samples]

    train_loader = DataLoader(
        Subset(train_ds, train_subset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_ds, val_subset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        val_acc = correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cnn_state_dict.pt"
    torch.save(best_state, model_path)

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_train_samples": len(train_subset),
        "max_val_samples": len(val_subset),
        "best_val_accuracy": float(best_val_acc),
    }
    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "training_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    return str(model_path), params
