# CIFAR-10 CNN MLOps (Docker + Compose)

## Scope implemented
This repository runs a containerized open-source MLOps workflow with:
- Git
- DVC (remote on MinIO / S3 API)
- MLflow
- ZenML
- Docker + Docker Compose

Monitoring is implemented, but drift detection and retrain-on-drift are intentionally disabled.

## Requirement check (current state)
### Tools
- Git: OK
- DVC + MinIO remote: OK (`.dvc/config` points to `s3://cifar-remote` on MinIO)
- MLflow: OK
- ZenML: OK
- Docker + Compose: OK
- Evidently drift detection: NOT active (disabled in current code)

### `training_pipeline` required steps
- `ingest_data`: NO (current pipeline uses `pull_data`)
- `validate_data`: NO
- `split_data`: NO
- `preprocess`: NO
- `train with CNN`: YES (`train_cnn_model`)
- `evaluate metrics + artifacts`: NO dedicated evaluate step in current training pipeline
- `register_model via MLflow`: NO dedicated register step in current training pipeline
- `export_model serving-ready`: NO dedicated export step in current training pipeline

Current `training_pipeline` executes:
- `start_infra`
- `pull_data`
- `train_cnn_model`
- `monitor_model`

### `monitoring_pipeline` required steps
- `collect_inference_data`: YES
- `run_evidently_report`: YES (name kept), but simple report only (no Evidently drift calculation)
- `trigger_decision`: YES
- `store_monitoring_artifacts`: YES
- Drift -> retrain trigger logic: DISABLED (`drift_detected` forced to `false`)

## Exact commands
Run from `C:\Users\ASUS\Desktop\mlops`.

### 1) Start infrastructure
```powershell
docker compose up -d minio init-minio mlflow zenml
```

### 2) Pull data with DVC
```powershell
docker exec zenml_dashboard python scripts/pull_cifar_with_dvc.py
```

### 3) Run training pipeline
```powershell
docker compose up training
```

### 4) Run monitoring pipeline
```powershell
docker compose up monitoring
```

### 5) Retrain on drift
Currently disabled by design in this repository.

## Service URLs
- MLflow: `http://127.0.0.1:5000`
- MinIO Console: `http://127.0.0.1:9001`
- ZenML Dashboard: `http://127.0.0.1:8237`

## Data policy
- Dataset files are excluded from Git (`data/` ignored in `.gitignore`)
- DVC is used for data pull/versioning workflow
