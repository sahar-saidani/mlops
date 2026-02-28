import csv
import shutil
from pathlib import Path
from zenml.steps import step


@step
def collect_inference_data() -> str:
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    inference = monitoring_dir / "inference_log.csv"
    reference = monitoring_dir / "reference.csv"

    if not inference.exists() and reference.exists():
        shutil.copyfile(reference, inference)

    if not inference.exists():
        raise FileNotFoundError("monitoring/inference_log.csv not found. Run training_pipeline first.")

    with inference.open("r", encoding="utf-8", newline="") as f:
        inference_reader = csv.DictReader(f)
        inference_columns = inference_reader.fieldnames or []
        inference_rows = sum(1 for _ in inference_reader)

    if not inference_columns or inference_rows == 0:
        raise ValueError("monitoring/inference_log.csv is empty.")

    if reference.exists():
        with reference.open("r", encoding="utf-8", newline="") as f:
            reference_reader = csv.DictReader(f)
            reference_columns = reference_reader.fieldnames or []
            reference_rows = sum(1 for _ in reference_reader)

        if not reference_columns or reference_rows == 0:
            raise ValueError("monitoring/reference.csv is empty.")
        missing_columns = sorted(set(reference_columns) - set(inference_columns))
        if missing_columns:
            raise ValueError(
                f"monitoring/inference_log.csv is missing required columns: {missing_columns}"
            )

    collected = monitoring_dir / "collected_inference_log.csv"
    shutil.copyfile(inference, collected)
    return str(collected)
