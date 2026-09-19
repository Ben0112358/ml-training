def test_config_paths_resolve(ml_training_env):
    import ml_training.config as config

    assert config.MODEL_DIR.name == "models"
    assert config.CLEAN_DATA_DIR.name == "clean"
