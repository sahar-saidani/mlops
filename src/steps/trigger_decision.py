import os
import subprocess

from zenml.steps import step


@step
def trigger_decision(drift: bool) -> bool:
    retrain = os.getenv("RETRAIN_ON_DRIFT", "false").lower() == "true"
    if drift and retrain:
        subprocess.run(["python", "-m", "src.pipelines.training_pipeline"], check=True)
    return drift
