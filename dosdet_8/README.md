# DOS Detector Pipeline

## Unified install

Install the shared dependencies for every detector from the repo root `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

Modernised training/inference stack for the simulated SDN IoT DoS detection project. The repository is now focused on a single workflow: preprocess raw pcaps into parquet shards, train the CNN detector, and run calibrated inference + evaluation.

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

Generated artefacts live under `artifacts/`, cached parquet shards under `cache/`, and inference reports under `reports/` (all ignored by git).

## Variant Notes (Dynamic int8)

- Dynamic int8 quantization is supported for CPU inference.
- Enable it via `quantization.enabled` in `config.yaml` or with `--quantized` CLI flags.
- Quantized checkpoints are stored under `quantization.checkpoint_path` (set it to `artifacts/model_best_int8_dynamic.pt` for local storage).
- Run the commands below from the `dosdet_8/` directory.
- These commands assume `config.yaml` uses local paths under this directory (`artifacts/`, `reports/`, `cache/`).

## Environment setup

1. Python 3.9+ recommended. CUDA 11.8 GPU is optional (needed only for CUDA fake-quant training or GPU float inference).
2. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ..[all]
   # Optional: install CUDA 11.8 build of PyTorch 2.0.1 (for GPU acceleration)
   pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1
   ```
   Install a CUDA-enabled PyTorch build only if you plan to train with the CUDA fake-quant path or run float inference on GPU.

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
   Besides the existing file-level verdict, each JSON report now includes the most suspicious source IP+MAC pairing plus the top 5 offenders, with companion `*_ips.csv`/`*_macs.csv` ranked exports.
   The detector emits a single sigmoid logit (attack vs. normal); the suspicious actor ranking is derived from those window probabilities so you still get the leading MAC/IP context per file.
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

- `paths.cache_dir`, `paths.artifacts_dir`, `paths.reports_dir`: all writable output directories.
- `windowing` + `data.top_k_udp_ports`: control feature extraction.
- `training`: tuned for small machines (`batch_size=128`, no AMP, zero worker loaders). Bump `batch_size` / `dataloader_workers` only if you have more CPU.
- `decision`: hysteresis + plausibility-gate parameters consumed by `infer.py` (and overridable via CLI flags).

Adjust the config once and rely on the Makefile; scripts read everything from the same file.

## Quick Commands (Quantization)

```bash
# 1) Train the float CNN (optional CUDA fake-quant is controlled by training.int8_quantized in config.yaml)
python3 train.py --config config.yaml

# 2) Create a dynamic int8 checkpoint (Linear layers only)
python3 scripts/quantize_supervised.py \
  --config config.yaml \
  --output artifacts/model_best_int8_dynamic.pt

# 3) Run quantized inference (dynamic)
python3 infer.py --config config.yaml \
  --pcaps "samples/*.pcap" \
  --out reports/int8 \
  --quantized \
  --quantized-checkpoint artifacts/model_best_int8_dynamic.pt

# 4) Optional: verify size/latency/accuracy vs float
python3 scripts/verify_quantization.py \
  --config config.yaml \
  --int8-checkpoint artifacts/model_best_int8_dynamic.pt \
  --backend qnnpack
```

### Post-training Static Quantization (PTQ)

Static int8 quantizes Conv1d + Linear layers but requires calibration data.

```bash
# 1) Ensure cached shards exist (calibration uses cache/manifest.json)
python3 -m data.preprocess --config config.yaml

# 2) Create a static int8 checkpoint (PTQ)
python3 scripts/quantize_static.py \
  --config config.yaml \
  --output artifacts/model_best_int8_static.pt \
  --backend qnnpack \
  --calib-batches 32

# 3) Update config.yaml quantization.mode: static
#    quantization.checkpoint_path: artifacts/model_best_int8_static.pt

# 4) Run quantized inference (static)
python3 infer.py --config config.yaml \
  --pcaps "samples/*.pcap" \
  --out reports/int8_static \
  --quantized \
  --quantized-checkpoint artifacts/model_best_int8_static.pt
```

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
`pytest` is included in the unified install (`pip install -e ..[all]`).
