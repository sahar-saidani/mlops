#!/usr/bin/env bash
set -euo pipefail

docker compose up -d minio init-minio mlflow zenml
docker exec zenml_dashboard python scripts/pull_cifar_with_dvc.py
docker exec zenml_dashboard python run_training_pipeline.py
