from zenml.pipelines import pipeline

from src.steps.quickstart_steps import (
    monitor_model,
    pull_data,
    start_infra,
    train_cnn_model,
)


@pipeline
def quickstart_pipeline() -> None:
    tracking_uri = start_infra()
    dataset_path = pull_data(tracking_uri)
    model_path, accuracy = train_cnn_model(dataset_path, tracking_uri)
    monitor_model(model_path, accuracy, tracking_uri)


if __name__ == "__main__":
    quickstart_pipeline()
