# DAE – Streaming Autoencoder Anomaly Detection

This project provides a production-ready pipeline for detecting anomalies in network traffic using a denoising autoencoder trained on normal PCAP data. The design focuses on high-throughput streaming extraction, compact CPU-friendly models, and reproducible artifact management.

## Features
- Streaming PCAP parsing with `scapy.RawPcapReader`; never loads entire files into memory.
- Sliding window feature generation with configurable duration/stride and lightweight aggregations.
- Robust preprocessing (quantile clipping, robust scaling) and deterministic training.
- Compact PyTorch denoising autoencoder with early stopping and threshold selection.
- Inference pipeline with post-processing (consecutive windows, cooldown) and top-k reporting.
- Structured logging via `structlog` and command-line utilities for extraction, training, and inference.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# Optional extras
pip install -e .[tests]
pip install -e .[plots]
```

## Quickstart
Assuming PCAPs containing only normal traffic are located under `data/raw/normal/`:

```bash
# 1) Extract streaming windows into Parquet
python scripts/extract_pcaps.py \
  --config config.yaml \
  --pcaps "data/raw/normal/*.pcap" \
  --out data/windows/normal.parquet

# 2) Train the autoencoder and compute thresholds
python scripts/train_autoencoder.py \
  --config config.yaml \
  --windows data/windows/normal.parquet

# 3) Score a new capture and export JSON + CSV reports
python scripts/infer_file.py \
  --config config.yaml \
  --pcap data/raw/test_800mb.pcap \
  --out data/reports/test_800mb.json
```

The JSON report resembles:
```json
{
  "file": "test_800mb.pcap",
  "decision": "attack detected",
  "anomalous_windows": 42,
  "total_windows": 1600,
  "threshold": 0.0371,
  "method": "quantile",
  "params": {"min_consecutive": 2, "cooldown": 1, "min_attack_windows": 3},
  "topk_windows": [
    {
      "idx": 712,
      "start": "2025-09-26T12:51:00Z",
      "end": "2025-09-26T12:51:01Z",
      "error": 0.214,
      "pkt_count": 420,
      "pps": 420.0,
      "src_ip_entropy": 3.1,
      "dst_ip_entropy": 2.8
    }
  ],
  "max_error_window": {
    "idx": 712,
    "start": "2025-09-26T12:51:00Z",
    "end": "2025-09-26T12:51:01Z",
    "error": 0.214
  }
}
```
The paired CSV contains full per-window scores, thresholds, and anomaly flags.

## Data Flow
1. **Extraction** – `extract_pcaps.py` streams packets, maintains overlapping windows (default 1s window, 0.5s stride), and writes feature rows to Parquet in bounded batches.
2. **Training** – `train_autoencoder.py` loads Parquet data, applies clipping/scaling, trains the PyTorch denoising autoencoder with early stopping, and saves artifacts (`model.pt`, `scaler.pkl`, `feature_list.json`, `clip_bounds.json`, `threshold.json`).
3. **Inference** – `infer_file.py` reuses the streaming extractor, scales with saved artifacts, scores batches on CPU, enforces threshold/post-processing rules, and exports JSON/CSV reports.

## Configuration
All behaviour is controlled through `config.yaml`. Key sections:
- `paths`: default directories for raw PCAPs, extracted windows, model artifacts, and reports.
- `extract`: streaming parameters (window/stride, parallel workers placeholder, batch row group size).
- `features`: toggles for feature inclusion and ratio calculations.
- `preprocess`: quantile clipping, scaler selection, shuffling, and reproducibility seed.
- `model`/`train`: architecture, optimizer, training schedule, and denoising noise.
- `threshold`: method (`quantile`, `gaussian_fit`, `evt`) and decision logic (`min_consecutive`, `cooldown`, `min_attack_windows`).
- `infer`: batch size for scoring and top-K percentile reporting.

Adjust the YAML, re-run extraction/training, and the downstream pipelines will incorporate the changes automatically.

## Adding Your Own PCAPs
1. Copy PCAP files into `data/raw/` (use subdirectories to distinguish normal vs evaluation sets).
2. Re-run extraction with appropriate glob patterns – multiple patterns are supported per invocation.
3. Keep training datasets purely normal; evaluation PCAPs may contain attacks/non-stationary behaviour.

## Threshold Tuning
- Increase `threshold.quantile` (e.g., 0.999) or raise `gaussian_k` to reduce false positives.
- Tune `min_consecutive` and `min_attack_windows` to treat isolated spikes as benign.
- For heavy-tail distributions, experiment with `threshold.method: evt` and adjust `evt_tail_quantile`.
- Always inspect the CSV to validate whether top anomaly windows align with expectations.

## Performance Tips (2 vCPU VM)
- Leave `train.device` at `cpu` and `train.num_threads` at 2; PyTorch is configured accordingly.
- For memory pressure, lower `extract.batch_rows` (at the cost of slightly more Parquet row groups).
- Disable live inference or reduce `infer.batch_windows` when CPU time is constrained.
- Use pattern-based extraction to target smaller subsets during experimentation.

## Testing
Run the lightweight unit tests with:
```bash
pytest
```
The suite validates window alignment, entropy calculations, scaler persistence, and threshold selection logic.

## Live Inference (Experimental)
Use `scripts/infer_live.py` to sniff traffic (requires sufficient privileges):
```bash
sudo python scripts/infer_live.py \
  --config config.yaml \
  --iface eth0 \
  --duration 60 \
  --out data/reports/live.json
```
Captured packets are written to a temporary PCAP and evaluated using the same inference pipeline.

## Logging and Artifacts
All scripts use structured JSON logs. Artifacts (model/scaler/threshold) live under `data/artifacts/`. Reports are stored under `data/reports/` by default, but custom paths are accepted on the CLI.

---
This repository is ready for integration into automation workflows or further extension (e.g., GPU acceleration, additional features, or dashboarding). Contributions are welcome.
