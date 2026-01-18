# Inference Subpackage

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

This subpackage hosts components used during unified Mixture-of-Experts inference:

- `configuration.py` – CLI argument parsing and inference configuration dataclass.
- `model_loader.py` – helpers for loading trained unified MoE checkpoints.
- `dataloader.py` – utilities for streaming PCAP windows via the shared dataset implementation.
- `aggregation.py` – per-window aggregation, class statistics, and verdict logic.
- `suspicion.py` – extraction of suspicious IP and MAC addresses from PCAP payloads.
- `reporting.py` – generation of CSV/JSON reports and formatted console banners.
- `infer_runner.py` – high-level orchestration tying together model loading, aggregation, and reporting.

All modules are documented with Google-style docstrings so that Sphinx or pdoc can generate
API documentation.