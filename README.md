# ZenML + MLflow Fast CNN (Out-of-the-Box)

This quickstart includes exactly 4 steps:

1. `start_infra`
2. `pull_data`
3. `train_cnn_model`
4. `monitor_model`

It uses a tiny CNN on sklearn digits (8x8 grayscale), so training is fast
and usually completes in less than one minute on CPU.

## File structure

```text
mlops/
|- requirements.txt
|- run_quickstart_pipeline.py
|- README.md
|- src/
|  |- __init__.py
|  |- pipelines/
|  |  |- __init__.py
|  |  |- quickstart_pipeline.py
|  |- steps/
|     |- __init__.py
|     |- quickstart_steps.py
```

## Exact commands (PowerShell)

Run from `C:\Users\ASUS\Desktop\mlops`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
zenml init
python run_quickstart_pipeline.py
```

Open MLflow UI:

```powershell
mlflow ui --backend-store-uri .\mlruns --port 5000
```

Then open `http://127.0.0.1:5000`.
