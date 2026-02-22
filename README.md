# SDN AI GP Monorepo

This repository contains multiple detector pipelines plus a unified gateway Mixture-of-Experts (MoE) model.

## Unified install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[all]
```

## Gateway MoE commands

### 1) MoE health check

```bash
python3 -m py_compile \
  gateway/moe_model.py \
  gateway/models/unified_moe.py \
  gateway/train_moe.py \
  gateway/infer_moe.py \
  gateway/eval_moe.py

python3 -c "from gateway.models.unified_moe import build_unified_moe; print('ok')"
python3 -m pytest -q gateway/tests
```

### 2) Optional cache build (recommended before full eval)

```bash
python3 gateway/preprocess_pcaps.py \
  --tasks dos,arp \
  --batch-size 64 \
  --max-windows-per-file 0 \
  --max-total-windows 0 \
  --num-threads 2
```

### 3) Global eval (full)

```bash
python3 gateway/eval_moe.py \
  --checkpoint gateway/unified_moe.pt \
  --split test \
  --seed 17 \
  --batch-size 64 \
  --use-cache auto \
  --max-windows-per-file 0 \
  --max-total-windows 0 \
  --output-dir gateway/eval
```

### 4) Fast smoke eval

```bash
python3 gateway/eval_moe.py \
  --checkpoint gateway/unified_moe.pt \
  --split test \
  --seed 17 \
  --batch-size 32 \
  --use-cache auto \
  --max-windows-per-file 20 \
  --max-total-windows 500 \
  --output-dir gateway/eval_smoke
```

## Global eval output contract

`gateway/eval_moe.py` writes:

- `metrics.json`
- `confusion_matrix.npy`
- `classification_report.txt`

`metrics.json` schema:

```json
{
  "accuracy": 0.7548724439638222,
  "precision": 0.8658096246031252,
  "recall": 0.8308503408980592,
  "f1": 0.8479698187984681,
  "roc_auc": 0.6436470326608363,
  "threshold": 0.904118537902832,
  "temperature": 1.9090687036514282,
  "samples": 162752,
  "positives": 133911
}
```

## Notes

- Binary report target is `attack vs normal` where positive = `dos` or `arp`.
- If `--temperature` or `--threshold` are omitted, they are fitted on validation windows.
- Split assignment is deterministic file-level (seeded) using files from `samples/labels.csv`.
