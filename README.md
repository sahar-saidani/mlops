# CIFAR-10 TinyCNN MLOps

End-to-end MLOps project for image classification on CIFAR-10 with:
- ZenML (pipeline orchestration)
- MLflow (experiment tracking and model registry)
- Docker Compose (local stack orchestration)
- MinIO (S3-compatible object storage)

The project provides two operational pipelines:
- `training_pipeline`: train, evaluate, register, and monitor a TinyCNN model
- `monitoring_pipeline`: collect inference data, compute drift signals, and store monitoring artifacts

## 1) Project Goals

- Train a lightweight CNN on CIFAR-10
- Track experiments and model versions in MLflow
- Produce reproducible artifacts for evaluation and monitoring
- Run the full workflow locally with Docker Compose

## 2) Repository Structure

- `src/pipelines/`: ZenML pipeline definitions
- `src/steps/`: pipeline steps (data, training, evaluation, monitoring, registration)
- `run_training_pipeline.py`: training entrypoint with terminal summary
- `run_monitoring_pipeline.py`: monitoring entrypoint with terminal summary
- `docker-compose.yml`: local stack and runtime services
- `docker/`: Dockerfiles for platform services

## 3) Current Data Strategy

The current split uses the full CIFAR-10 dataset:
- Train: 50,000 samples
- Validation: 0 samples
- Test: 10,000 samples

Implementation:
- `src/steps/split_data.py` writes:
  - `artifacts/splits/train_idx.json` (0..49999)
  - `artifacts/splits/val_idx.json` (empty)
  - `artifacts/splits/test_idx.json` (0..9999)

## 4) Training Pipeline

Pipeline file: `src/pipelines/training_pipeline.py`

Step sequence:
1. `start_infra`
2. `ingest_data`
3. `validate_data`
4. `split_data`
5. `preprocess`
6. `train_cnn_model`
7. `evaluate`
8. `build_training_params`
9. `register_model`
10. `monitor_model`

Key outputs:
- `artifacts/tiny_cnn.pt`
- `artifacts/metrics/confusion_matrix.png`
- `artifacts/metrics/classification_report.txt`
- `artifacts/monitoring_status.txt` (status + accuracy)
- `monitoring/reference.csv`
- `monitoring/inference_log.csv`

## 5) Monitoring Pipeline

Pipeline file: `src/pipelines/monitoring_pipeline.py`

Step sequence:
1. `collect_inference_data`
2. `run_evidently_report`
3. `trigger_decision`
4. `store_monitoring_artifacts`

Key outputs:
- `monitoring/collected_inference_log.csv`
- `monitoring/report.html`
- `monitoring/summary.json`

`run_evidently_report` currently computes a lightweight drift signal (probability shift + label distribution shift) and returns:
- `drift_detected`
- `drift_score`
- `rows`

## 6) Run with Docker Compose

From project root (`C:\Users\ASUS\Desktop\mlops`):

Start infrastructure:
```powershell
docker compose up -d minio init-minio mlflow zenml
```

Run training:
```powershell
docker compose up training
```

Run monitoring:
```powershell
docker compose up monitoring
```

Service URLs:
- MLflow: `http://127.0.0.1:5000`
- MinIO Console: `http://127.0.0.1:9001`
- ZenML Dashboard: `http://127.0.0.1:8237`

## 7) Terminal Summaries

After each run:
- `run_training_pipeline.py` prints:
  - generated/updated files
  - `val_accuracy` and `test_accuracy`
- `run_monitoring_pipeline.py` prints:
  - generated/updated monitoring files
  - drift summary (`drift_detected`, `drift_score`, `rows`, `action`)

## 8) Notes

- Model registry name: `cifar10-cnn`
- Training/monitoring steps that must reflect latest state are configured with cache disabled where needed
- Docker compose is configured to reduce noisy runtime warnings