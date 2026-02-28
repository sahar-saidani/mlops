cleafrom __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_with_venv_if_available() -> None:
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        cmd = [str(venv_python), __file__, *sys.argv[1:]]
        raise SystemExit(subprocess.call(cmd))


def main() -> None:
    _run_with_venv_if_available()
    try:
        from src.pipelines.training_pipeline import training_pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "zenml":
            raise ModuleNotFoundError(
                "zenml is not installed in the current Python environment. "
                "Install requirements or run from .venv."
            ) from exc
        raise
    training_pipeline()


if __name__ == "__main__":
    main()
