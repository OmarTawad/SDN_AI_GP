# Models Subpackage

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Will host the unified Mixture-of-Experts implementation and expert loading utilities.