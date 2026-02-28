from zenml.pipelines import pipeline

from src.steps.evaluate import evaluate
from src.steps.export_model import export_model
from src.steps.ingest_data import ingest_data
from src.steps.preprocess import preprocess
from src.steps.register_model import register_model
from src.steps.split_data import split_data
from src.steps.train import train
from src.steps.validate_data import validate_data


@pipeline
def training_pipeline() -> None:
    data_root = ingest_data()
    validated_root = validate_data(data_root)
    train_idx_path, val_idx_path, test_idx_path = split_data(validated_root)
    preprocess_cfg_path = preprocess(train_idx_path, val_idx_path, test_idx_path)
    model_path, params = train(preprocess_cfg_path)
    metrics = evaluate(model_path, preprocess_cfg_path)
    register_model(model_path, metrics, params)
    export_model(model_path)


if __name__ == "__main__":
    training_pipeline()