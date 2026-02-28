#!/usr/bin/env bash
set -euo pipefail

docker compose up -d minio init-minio mlflow zenml
docker exec zenml_dashboard python run_monitoring_pipeline.py
