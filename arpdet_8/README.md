# ARP Spoof Detector Pipeline

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Modernised training/inference stack for the simulated SDN IoT ARP-spoofing detection project. This `arpdet_8` variant keeps the exact flow of `arpdet` but **forces CUDA-only int8 fake-quantisation** (tested on GTX 1050 Ti / CUDA 11.8) so you can run on Pascal-era GPUs without FP16.

## Project layout

```
data/               # preprocessing + dataset utilities
features/           # sequence/static feature extraction + scaling
models/             # neural network components and helpers
samples/            # small example pcaps
labels/             # labels.csv aligned with samples
train.py            # main training entrypoint (from cached shards)
infer.py            # calibrated inference over pcaps
explain.py          # permutation importances + attention plots
Makefile            # convenience targets (preprocess/train/infer/eval)
config.yaml         # single source of truth for paths + hyper-params
requirements.txt    # legacy dependencies (use repo-root pyproject.toml)
```

Generated artefacts live under `artifacts_int8/`, cached parquet shards under `cache/`, and inference reports under `reports_int8/` (all ignored by git).

## Environment setup

1. Python 3.9+ recommended. CUDA 11.8 GPU required (CUDA-only int8 pipeline, no CPU fallback).
2. Create a virtualenv and install dependencies (use the CUDA 11.8 PyTorch wheel; works on GTX 1050 Ti):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ..[all]
   pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1
   ```
   The training/inference scripts will raise if no CUDA device is available.

All scripts respect the 2 vCPU constraint: BLAS/Torch threads are capped to two, data loaders run single-threaded by default, and every long-running step renders a tqdm progress bar so you can track preprocess/train/infer progress without flooding the terminal.

## End-to-end workflow

The `Makefile` reflects the standard flow; the commands below assume you activated the virtualenv.

1. **Preprocess pcaps** → parquet shards under `cache/`:
   ```bash
   make preprocess
   ```
   Uses `config.yaml:preprocess.pcaps_glob` and `labels_csv`. Adjust those paths (and windowing parameters) before running on your own data.

2. **Train the detector** (reads cached shards, writes to `artifacts_int8/`):
   ```bash
   make train
   ```
   Training performs thin/ full validation splits, optional hard-negative mining, temperature scaling and persists:
   - `artifacts_int8/model_best.pt`
   - `artifacts_int8/scaler.pkl` and PCA components
   - `artifacts_int8/feature_model_meta.json`
   - `artifacts_int8/calibration.json`

3. **Run inference** (produces per-window CSV + per-file JSON, default `reports_int8/`):
   ```bash
   make infer
   ```
   Override CLI flags if you need to point at other pcaps or tweak decision thresholds (see `infer.py --help`).
   Each JSON report contains the binary decision plus the most suspicious MAC address (including the IPs it claimed), derived from ARP reply conflicts observed while scoring the capture.
   The detector emits a single sigmoid logit (attack vs. normal), and the decision is promoted to *attack* only when a MAC responds for multiple distinct IPs; otherwise captures stay `normal` even if the model logits are high.
   To score an arbitrary capture outside `samples/`, simply point `--pcaps` at your file or glob:
   ```bash
   python3 infer.py --config config.yaml --pcaps "/data/captures/new_attack.pcap" --out custom_reports
   ```

4. **Evaluate predictions** (`eval.py`) and **generate explanations** (`explain.py`):
   ```bash
   make eval
   make explain
   ```
   The explanation step writes permutation importances (`perm_importance.json`) and attention heatmaps in `reports_int8/explain/`.

## Configuration highlights (`config.yaml`)

- `paths.cache_dir`, `paths.artifacts_dir`, `paths.reports_dir`: writable output directories.
- `windowing`: controls the 1-second sliding window size, stride, and micro-bin resolution (kept lightweight for 2 vCPU nodes).
- `training`: tuned for small machines (`batch_size=128`, enforced int8 quantisation on CUDA, zero worker loaders). Bump `batch_size` / `dataloader_workers` only if you have more CPU.
- `decision`: hysteresis + ARP conflict gate consumed by `infer.py` (and overridable via CLI flags).

Adjust the config once and rely on the Makefile; scripts read everything from the same file.

## Cleaning up

Use
```bash
make clean
```
to drop `cache/`, `artifacts_int8/`, and `reports_int8/`. The `.gitignore` already keeps these directories out of version control.

## Tests

Unit tests live under `tests/`. Run them with:
```bash
pytest
```
`pytest` is included in the unified install (`pip install -e ..[all]`).
