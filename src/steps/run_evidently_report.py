import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import mean, pstdev

from zenml.steps import step


@step(enable_cache=False)
def run_evidently_report(inference_path: str) -> dict:
    def _read_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return (reader.fieldnames or []), list(reader)

    columns, inference_rows = _read_rows(inference_path)
    row_count = len(inference_rows)

    if not columns or row_count == 0:
        raise ValueError(f"{inference_path} is empty.")

    reference_path = Path("monitoring") / "reference.csv"
    if not reference_path.exists():
        html = (
            "<html><body>"
            "<h1>Monitoring Report</h1>"
            f"<p>Rows checked: {row_count}</p>"
            "<p>Reference dataset missing; drift check skipped.</p>"
            "</body></html>"
        )
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "rows": int(row_count),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "html": html,
        }

    ref_columns, reference_rows = _read_rows(str(reference_path))
    if not ref_columns or len(reference_rows) == 0:
        raise ValueError("monitoring/reference.csv is empty.")

    numeric_cols = [c for c in columns if c.startswith("prob_") and c in ref_columns]
    if not numeric_cols:
        raise ValueError("No numeric probability columns found for drift detection.")

    z_scores: list[float] = []
    drifted_cols: list[str] = []
    for col in numeric_cols:
        ref_vals = [float(r[col]) for r in reference_rows]
        inf_vals = [float(r[col]) for r in inference_rows]
        ref_mean = mean(ref_vals)
        inf_mean = mean(inf_vals)
        ref_std = pstdev(ref_vals)
        z = abs(inf_mean - ref_mean) / max(ref_std, 1e-6)
        z_scores.append(z)
        if z > 0.5:
            drifted_cols.append(col)

    drift_ratio = len(drifted_cols) / max(len(numeric_cols), 1)
    avg_z = mean(z_scores) if z_scores else 0.0

    def _label_distribution(rows: list[dict[str, str]], key: str) -> dict[int, float]:
        counts: dict[int, int] = {}
        for r in rows:
            value = int(r.get(key, "0"))
            counts[value] = counts.get(value, 0) + 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total else {}

    ref_dist = _label_distribution(reference_rows, "predicted_label")
    inf_dist = _label_distribution(inference_rows, "predicted_label")
    labels = sorted(set(ref_dist) | set(inf_dist))
    tvd = 0.5 * sum(abs(ref_dist.get(k, 0.0) - inf_dist.get(k, 0.0)) for k in labels)

    drift_score = min(avg_z / 3.0, 1.0) * 0.7 + min(tvd / 0.3, 1.0) * 0.3
    drift_detected = drift_ratio > 0.3 or drift_score > 0.35

    html = (
        "<html><body>"
        "<h1>Monitoring Report</h1>"
        f"<p>Rows checked: {row_count}</p>"
        f"<p>Columns checked: {len(numeric_cols)}</p>"
        f"<p>Avg z-score shift: {avg_z:.4f}</p>"
        f"<p>Label TVD: {tvd:.4f}</p>"
        f"<p>Drifted columns: {len(drifted_cols)}</p>"
        f"<p>Drift score: {drift_score:.4f}</p>"
        f"<p>Drift detected: {drift_detected}</p>"
        "</body></html>"
    )

    return {
        "drift_detected": bool(drift_detected),
        "drift_score": float(drift_score),
        "rows": int(row_count),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "html": html,
    }
