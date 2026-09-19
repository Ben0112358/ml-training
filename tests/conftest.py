import importlib
import sys

import pytest


def _purge_package(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


@pytest.fixture
def ml_training_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_HOMELAB_ROOT", str(tmp_path))
    monkeypatch.setenv("OUTPUT_SUFFIX", "smoke_suffix")
    monkeypatch.setenv("CLEAN_DATA_DIR", "data/clean")
    monkeypatch.setenv("MODEL_DIR", "models")
    monkeypatch.setenv("LOGS_DIR", "logs/training")
    _purge_package("ml_training")
    importlib.import_module("ml_training.config")
    yield
