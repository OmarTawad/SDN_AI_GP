# Data Subpackage

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Intended to expose dataset builders, preprocessing utilities, and data structures resued across
training and inference. Existing functionality will be migrated from `data_pipeline.py`.