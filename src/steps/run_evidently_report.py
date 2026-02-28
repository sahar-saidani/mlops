from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from zenml.steps import step


@step
def run_evidently_report(inference_path: str) -> tuple[bool, str]:
    reference_path = Path("monitoring/reference.csv")
    if not reference_path.exists():
        raise FileNotFoundError("monitoring/reference.csv not found. Run training_pipeline first.")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(inference_path)

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference_df, current_data=current_df)

    html = report.as_html()
    drift = bool(report.as_dict()["metrics"][0]["result"]["dataset_drift"])
    return drift, html
