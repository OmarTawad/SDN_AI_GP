# ARP Spoof Detector Pipeline

Modernised training/inference stack for the simulated SDN IoT ARP-spoofing detection project. The repository focuses on a single workflow: preprocess raw pcaps into parquet shards, train the CNN detector, and run calibrated inference with conflict-aware post-processing.

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
requirements.txt    # python dependencies
```

Generated artefacts live under `artifacts/`, cached parquet shards under `cache/`, and inference reports under `reports/` (all ignored by git).

## Environment setup

1. Python 3.9+ recommended.
2. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
   Install a CUDA-enabled PyTorch build if you plan to train on GPU (see https://pytorch.org/get-started/locally/).

All scripts respect the 2 vCPU constraint: BLAS/Torch threads are capped to two, data loaders run single-threaded by default, and every long-running step renders a tqdm progress bar so you can track preprocess/train/infer progress without flooding the terminal.

## End-to-end workflow

The `Makefile` reflects the standard flow; the commands below assume you activated the virtualenv.

1. **Preprocess pcaps** → parquet shards under `cache/`:
   ```bash
   make preprocess
   ```
   Uses `config.yaml:preprocess.pcaps_glob` and `labels_csv`. Adjust those paths (and windowing parameters) before running on your own data.

2. **Train the detector** (reads cached shards, writes to `artifacts/`):
   ```bash
   make train
   ```
   Training performs thin/ full validation splits, optional hard-negative mining, temperature scaling and persists:
   - `artifacts/model_best.pt`
   - `artifacts/scaler.pkl` and PCA components
   - `artifacts/feature_model_meta.json`
   - `artifacts/calibration.json`

3. **Run inference** (produces per-window CSV + per-file JSON, default `reports/`):
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
   The explanation step writes permutation importances (`perm_importance.json`) and attention heatmaps in `reports/explain/`.

## Configuration highlights (`config.yaml`)

- `paths.cache_dir`, `paths.artifacts_dir`, `paths.reports_dir`: writable output directories.
- `windowing`: controls the 1-second sliding window size, stride, and micro-bin resolution (kept lightweight for 2 vCPU nodes).
- `training`: tuned for small machines (`batch_size=128`, no AMP, zero worker loaders). Bump `batch_size` / `dataloader_workers` only if you have more CPU.
- `decision`: hysteresis + ARP conflict gate consumed by `infer.py` (and overridable via CLI flags).

Adjust the config once and rely on the Makefile; scripts read everything from the same file.

## Cleaning up

Use
```bash
make clean
```
to drop `cache/`, `artifacts/`, and `reports/`. The `.gitignore` already keeps these directories out of version control.

## Tests

Unit tests live under `tests/`. Run them with:
```bash
pytest
```
(install `pytest` via `pip install -r requirements.txt`).
