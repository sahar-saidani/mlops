from zenml.pipelines import pipeline

from src.steps.monitor_model import monitor_model
from src.steps.pull_data import pull_data
from src.steps.start_infra import start_infra
from src.steps.train_cnn_model import train_cnn_model


@pipeline
def training_pipeline() -> None:
    tracking_uri = start_infra()
    dataset_path = pull_data(tracking_uri)
    model_path, accuracy = train_cnn_model(dataset_path, tracking_uri)
    monitor_model(model_path, accuracy, tracking_uri)


if __name__ == "__main__":
    training_pipeline()
