import importlib
import sys


def _purge_ml_training() -> None:
    for name in list(sys.modules):
        if name == "ml_training" or name.startswith("ml_training."):
            del sys.modules[name]


def _load_config_with_yaml(monkeypatch, tmp_path, yaml_text: str):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("ML_HOMELAB_ROOT", str(tmp_path))
    monkeypatch.setenv("OUTPUT_SUFFIX", "smoke_suffix")
    monkeypatch.setenv("CLEAN_DATA_DIR", "data/clean")
    monkeypatch.setenv("MODEL_DIR", "models")
    monkeypatch.setenv("LOGS_DIR", "logs/training")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    _purge_ml_training()
    return importlib.import_module("ml_training.config")


def test_empty_config_yaml_falls_back_to_env(monkeypatch, tmp_path):
    config = _load_config_with_yaml(monkeypatch, tmp_path, "")
    assert config.CONFIG == {}
    assert config.MODEL_DIR.name == "models"
    assert config.CLEAN_DATA_DIR.name == "clean"
    assert config.LOGS_DIR.name == "training"


def test_null_config_yaml_falls_back_to_env(monkeypatch, tmp_path):
    config = _load_config_with_yaml(monkeypatch, tmp_path, "null\n")
    assert config.CONFIG == {}
    assert config.MODEL_DIR.name == "models"


def test_null_paths_mapping_falls_back_to_env(monkeypatch, tmp_path):
    config = _load_config_with_yaml(monkeypatch, tmp_path, "paths:\n")
    assert config.CONFIG.get("paths") is None
    assert config.MODEL_DIR.name == "models"
    assert config.CLEAN_DATA_DIR.name == "clean"
