import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as T
from zenml.steps import step

from src.steps.train_cnn_model import TinyCNN


@step
def evaluate(model_path: str, preprocess_cfg_path: str) -> dict:
    cfg = json.loads(Path(preprocess_cfg_path).read_text(encoding="utf-8"))

    batch_size = 128
    num_workers = 0
    max_test_samples = 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(tuple(cfg["mean"]), tuple(cfg["std"])),
        ]
    )
    full_test_ds = datasets.CIFAR10(root="data/raw", train=False, transform=eval_tf, download=False)
    test_ds = torch.utils.data.Subset(full_test_ds, list(range(min(max_test_samples, len(full_test_ds)))))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = TinyCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    metrics_dir = Path("artifacts/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(metrics_dir / "confusion_matrix.png")
    plt.close()

    (metrics_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred), encoding="utf-8"
    )

    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = [f"prob_{i}" for i in range(10)]
    df = pd.DataFrame(y_prob, columns=prob_cols)
    df["predicted_label"] = y_pred
    df["true_label"] = y_true

    df.to_csv(monitoring_dir / "reference.csv", index=False)
    df.to_csv(monitoring_dir / "inference_log.csv", index=False)

    return {"accuracy": float(acc), "f1_macro": float(f1)}
