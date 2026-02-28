from zenml.pipelines import pipeline

from src.steps.build_training_params import build_training_params
from src.steps.evaluate import evaluate
from src.steps.ingest_data import ingest_data
from src.steps.monitor_model import monitor_model
from src.steps.preprocess import preprocess
from src.steps.register_model import register_model
from src.steps.split_data import split_data
from src.steps.start_infra import start_infra
from src.steps.train_cnn_model import train_cnn_model
from src.steps.validate_data import validate_data


@pipeline
def training_pipeline() -> None:
    tracking_uri = start_infra()
    data_root = ingest_data()
    validated_root = validate_data(data_root)
    train_idx_path, val_idx_path, test_idx_path = split_data(validated_root)
    preprocess_cfg_path = preprocess(train_idx_path, val_idx_path, test_idx_path)
    model_path, accuracy = train_cnn_model(validated_root, preprocess_cfg_path, tracking_uri)
    metrics = evaluate(model_path, validated_root, preprocess_cfg_path)
    training_params_json = build_training_params(
        mlflow_tracking_uri=tracking_uri,
        best_val_accuracy=accuracy,
        source_dataset="cifar10",
        evaluation_dataset="cifar10",
    )
    register_model(
        model_path=model_path,
        metrics=metrics,
        training_params_json=training_params_json,
    )
    monitor_model(model_path, metrics, tracking_uri)


if __name__ == "__main__":
    training_pipeline()
