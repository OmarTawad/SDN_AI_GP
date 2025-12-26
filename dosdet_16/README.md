# DOS Detector Pipeline

Modernised training/inference stack for the simulated SDN IoT DoS detection project. The repository is now focused on a single workflow: preprocess raw pcaps into parquet shards, train the CNN detector, and run calibrated inference + evaluation. This `dosdet_16` variant defaults to AMP/FP16 on CUDA and writes outputs to separate `artifacts_fp16/` + `reports_fp16/` directories so FP32 runs remain untouched.

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

Generated artefacts live under `artifacts_fp16/`, cached parquet shards under `cache/`, and inference reports under `reports_fp16/` (all ignored by git).

## Environment setup

1. Python 3.9+ recommended. This variant runs CPU-only with FP16 enabled by default.
2. Create a virtualenv and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ..[all]
   # Install CUDA 11.8 build of PyTorch 2.0.1 (keeps wheels small/fast on GPU)
   pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1
   ```
   Install a CUDA-enabled PyTorch build if you plan to train on GPU (see https://pytorch.org/get-started/locally/).

All scripts respect the 2 vCPU constraint: BLAS/Torch threads are capped to two, data loaders run single-threaded by default, and every long-running step renders a tqdm progress bar so you can track preprocess/train/infer progress without flooding the terminal.

## End-to-end workflow

The `Makefile` reflects the standard flow; the commands below assume you activated the virtualenv.

1. **Preprocess pcaps** → parquet shards under `cache/`:
   ```bash
   make preprocess
   python3 -m data.preprocess --config config.yaml   --pcaps samples/mixed1.pcap samples/normal*.pcap samples/ssdp_flood*.pcap   --labels labels/labels.csv
   ```
   Uses `config.yaml:preprocess.pcaps_glob` and `labels_csv`. Adjust those paths (and windowing parameters) before running on your own data.

2. **Train the detector** (reads cached shards, writes to `artifacts_fp16/`):
   ```bash
   make train
   ```
   Training performs thin/ full validation splits, optional hard-negative mining, temperature scaling and persists:
   - `artifacts_fp16/model_best.pt`
   - `artifacts_fp16/scaler.pkl` and PCA components
   - `artifacts_fp16/feature_model_meta.json`
   - `artifacts_fp16/calibration.json`

3. **Run inference** (produces per-window CSV + per-file JSON, default `reports_fp16/`):
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
   The explanation step writes permutation importances (`perm_importance.json`) and attention heatmaps in `reports_fp16/explain/`.

## Export FP32 -> FP16 (post-training)

Why: CPU fp16 training is slower and can hurt AUC on this hardware. The safer flow is to train in FP32, then export a FP16 checkpoint for inference/deployment while preserving the trained weights.

Convert the best FP32 checkpoint to FP16:
```bash
python3 - <<'PY'
import os, torch
src = "artifacts/model_best.pt"
dst_dir = "artifacts_fp16"
os.makedirs(dst_dir, exist_ok=True)
dst = os.path.join(dst_dir, "model_best.pt")

ckpt = torch.load(src, map_location="cpu")
ckpt["model"] = {k: (v.half() if torch.is_floating_point(v) else v)
                 for k, v in ckpt["model"].items()}
torch.save(ckpt, dst)
print("Saved fp16 checkpoint to", dst)
PY
```

Verify the checkpoint weights are fp16:
```bash
python3 - <<'PY'
import torch
ckpt = torch.load("artifacts_fp16/model_best.pt", map_location="cpu")
dtypes = {v.dtype for v in ckpt["model"].values() if torch.is_floating_point(v)}
print(dtypes)
PY
```

Copy the remaining artifacts into `artifacts_fp16/` (scaler/PCA/meta/calibration), then run inference as usual.

## Configuration highlights (`config.yaml`)

- `paths.cache_dir`, `paths.artifacts_dir`, `paths.reports_dir`: all writable output directories.
- `windowing` + `data.top_k_udp_ports`: control feature extraction.
- `training`: tuned for small machines (`batch_size=128`, `fp16_cpu: true`, zero worker loaders). Bump `batch_size` / `dataloader_workers` only if you have more CPU.
- `decision`: hysteresis + plausibility-gate parameters consumed by `infer.py` (and overridable via CLI flags).

Adjust the config once and rely on the Makefile; scripts read everything from the same file.

## Cleaning up

Use
```bash
make clean
```
to drop `cache/`, `artifacts_fp16/`, and `reports_fp16/`. The `.gitignore` already keeps these directories out of version control.

## Tests

Unit tests live under `tests/`. Run them with:
```bash
pytest
```
`pytest` is included in the unified install (`pip install -e ..[all]`).
