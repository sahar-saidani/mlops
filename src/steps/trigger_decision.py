import os
import subprocess

from zenml.steps import step


@step
def trigger_decision(report: dict) -> dict:
    drift = bool(report.get("drift_detected", False))
    retrain = os.getenv("RETRAIN_ON_DRIFT", "false").lower() == "true"

    action = "retrain_recommended" if drift else "no_action"
    if drift and retrain:
        subprocess.run(["python", "-m", "src.pipelines.training_pipeline"], check=True)

    return {"action": action, "drift_detected": drift}
