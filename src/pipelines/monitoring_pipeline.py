from zenml.pipelines import pipeline

from src.steps.collect_inference_data import collect_inference_data
from src.steps.run_evidently_report import run_evidently_report
from src.steps.store_monitoring_artifacts import store_monitoring_artifacts
from src.steps.trigger_decision import trigger_decision


@pipeline
def monitoring_pipeline() -> None:
    inference_path = collect_inference_data()
    report = run_evidently_report(inference_path)
    decision = trigger_decision(report)
    store_monitoring_artifacts(report, decision)


if __name__ == "__main__":
    
    monitoring_pipeline()