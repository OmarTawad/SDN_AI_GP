# Tests

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Pytest-based unit tests validating gateway components. Add new tests as modules are migrated
into the refactored architecture.