# ml-training

`ml-training` is the model training stage of the ML pipeline. Its purpose is to take cleaned datasets from `ml-data`, train project-specific ML models, and output ready-to-use model artifacts for serving.  

This submodule forms a self-contained step in the larger ML pipeline:

```
ml-infra → ml-data → ml-training → ml-serving → ml-ui
```

`ml-training` can be run **locally** for development or as part of the **full pipeline** orchestrated via `execute.sh` from https://github.com/Ben0112358/ml-pipeline.

To get an overview of how all sub-repos in the full pipeline are tied together, refer to https://github.com/Ben0112358/ml-meta. Links to all sub-repos can be found therein as well.

---

## 📁 Project Structure

```
ml-training/
├── docker-compose.dummy_project.yaml   # Docker Compose file for containerized run
├── Dockerfile.dummy_project            # Dockerfile for containerized project
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
├── src/
│   └── ml_training/
│       ├── config.py                   # Global configuration (paths, suffixes)
│       ├── dummy_project/
│       │   ├── training.py             # Core training logic
│       │   ├── utils/                  # Project-specific helpers (e.g. fake data)
│       │   ├── __main__.py             # Local dev CLI entrypoint
│       │   └── __init__.py
│       └── utils/                      # Shared utils (logging, etc.)
└── tests/                              # Unit tests
```

---

## ✅ Prerequisites

- **OS**: Linux or macOS  
- **Docker**: Installed and running  
- **Python**: 3.12+  
- **Poetry**: For dependency management  

Set the base directory where shared ML assets and configs are stored:

```bash
export ML_HOMELAB_ROOT=/absolute/path/to/ml-homelab
```

## 🐳 Containerized run (more control)

`ml-training` can be run for example in the following way. You may add args as you see fit.

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
docker-compose -f docker-compose.<project_name>.yaml -p "<project_name>_<mode>" build --no-cache
docker-compose -f docker-compose.<project_name>.yaml -p "<project_name>_<mode>" up
```

For more control, the following can be exported:
```bash
export CLEAN_DATA_DIR=/path/to/clean
export MODELS_DIR=/path/to/models
export LOGS_DIR=/path/to/logs
export OUTPUT_SUFFIX=some_suffix
```

---

## 🐍 Python run (less control; simplified)

Run `ml-training` locally with sensible defaults:

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
python -m ml_training.<project_name>
```

Or with more control over directories and outputs:

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
export CLEAN_DATA_DIR=/path/to/clean
export MODELS_DIR=/path/to/models
export LOGS_DIR=/path/to/logs
export OUTPUT_SUFFIX=some_suffix

python -m ml_training.<project_name>
```

**Notes**:
- This mode is a lightweight wrapper around docker-compose.<project_name>.yaml for convenience during development.

---

## ➕ Adding a New Project
1. Create a folder under `ml_training/` with your project name:

```
src/ml_training/<new_project>/
```

2. Implement the modules (mirroring `dummy_project`):

- `training.py` → core training logic  
- `utils/` → project-specific helpers (data generators, metrics, etc.)  
- `__main__.py` → optional CLI entrypoint for local dev  
- `__init__.py` → marks the package  

3. Add corresponding `docker-compose.<new_project>.yaml` and `Dockerfile.<new_project>`.

4. Set project-specific configuration in `ml_training/config.py` or via environment variables (`CLEAN_DATA_DIR`, `MODELS_DIR`, `OUTPUT_SUFFIX`).  

`src/ml_training/dummy_project` is a very simple project which can be studied to learn how it all ties together.

---

## 🧪 Testing

Run unit tests with Poetry:

```bash
poetry run pytest tests/
```
