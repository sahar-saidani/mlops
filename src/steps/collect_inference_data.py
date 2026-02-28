from pathlib import Path

import pandas as pd
from zenml.steps import step


@step
def collect_inference_data() -> str:
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    inference = monitoring_dir / "inference_log.csv"
    reference = monitoring_dir / "reference.csv"

    if not inference.exists() and reference.exists():
        pd.read_csv(reference).to_csv(inference, index=False)

    if not inference.exists():
        raise FileNotFoundError("monitoring/inference_log.csv not found. Run training_pipeline first.")

    collected = monitoring_dir / "collected_inference_log.csv"
    pd.read_csv(inference).to_csv(collected, index=False)
    return str(collected)
