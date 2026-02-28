# MLOps Project Documentation

This repository includes:

1. A minimal **quickstart ZenML pipeline** with 4 steps:
   - `start_infra`
   - `pull_data`
   - `train_cnn_model`
   - `monitor_model`
2. A complete CIFAR-10 stack (Docker + MinIO + MLflow + ZenML + DVC + Evidently).

The quickstart path is the fastest way to validate the whole pipeline logic.

## 1) Exact Commands To Run (Submission Flow)

Run from: `C:\Users\ASUS\Desktop\mlops`

### Start infra
```powershell
docker compose up -d minio init-minio mlflow zenml
```

### Pull data
```powershell
docker exec zenml_dashboard python scripts/pull_cifar_with_dvc.py
```

### Train
```powershell
docker exec zenml_dashboard python run_training_pipeline.py
```

### Monitor
```powershell
docker exec zenml_dashboard python run_monitoring_pipeline.py
```

## 2) Fast Working Pipeline (under 1 minute)

This is the tested quick path:

```powershell
docker compose run --rm training python run_quickstart_pipeline.py
```

Recent test result on this machine:
- Pipeline total duration: ~21s
- `train_cnn_model` duration: ~16s

## 3) Local Python Alternative

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
zenml init
python run_quickstart_pipeline.py
```

## 4) UI Links

- MLflow: `http://127.0.0.1:5000`
- MinIO console: `http://127.0.0.1:9001`
- ZenML Dashboard (if started): `http://127.0.0.1:8237`

## 5) Short Video Demo Script (2 to 5 min)

Record this exact sequence:

1. Open terminal in `C:\Users\ASUS\Desktop\mlops`
2. Run:
   - `docker compose up -d minio init-minio mlflow`
3. Show containers up:
   - `docker ps`
4. Run training (quickstart):
   - `docker compose run --rm training python run_quickstart_pipeline.py`
5. Show ZenML step logs in terminal:
   - `start_infra`, `pull_data`, `train_cnn_model`, `monitor_model`
6. Open MLflow UI in browser (`http://127.0.0.1:5000`)
7. Show:
   - experiment `zenml-fast-cnn`
   - logged metric `test_accuracy`
   - model artifact
8. (Monitoring evidence) show monitor output in run logs and MLflow run:
   - `observed_accuracy`
   - `status=healthy` / `needs_retraining`
9. End video with final successful pipeline status.
