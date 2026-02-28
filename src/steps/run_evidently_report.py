import csv
from datetime import datetime, timezone

from zenml.steps import step


@step
def run_evidently_report(inference_path: str) -> dict:
    with open(inference_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        row_count = sum(1 for _ in reader)

    if not columns or row_count == 0:
        raise ValueError(f"{inference_path} is empty.")

    html = (
        "<html><body>"
        "<h1>Monitoring Report</h1>"
        f"<p>Rows checked: {row_count}</p>"
        "<p>Drift detection is disabled.</p>"
        "</body></html>"
    )

    return {
        "drift_detected": False,
        "drift_score": None,
        "rows": int(row_count),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "html": html,
    }
