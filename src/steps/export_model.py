from pathlib import Path

import torch
from zenml.steps import step

from src.steps.train_cnn_model import TinyCNN


@step
def export_model(model_path: str) -> str:
    model = TinyCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    out_dir = Path("artifacts/exported")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_path = out_dir / "model.torchscript.pt"
    scripted = torch.jit.script(model)
    scripted.save(str(exported_path))

    return str(exported_path)
