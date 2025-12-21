# Core Subpackage

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Holds shared configuration, label metadata, and other cross-cutting helpers used by the
gateway project.