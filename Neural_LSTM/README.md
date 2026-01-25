<<<<<<< HEAD
# DoS Detector (Neural_LSTM)

This module contains an end-to-end pipeline for detecting Distributed Denial of Service (DoS) activity from PCAP captures using an LSTM-based sequence classifier. The inference stage highlights the most suspicious MAC and IP addresses observed during an attack, making it easier to pivot from the prediction to actionable remediation.
=======
# DoS Detector (Neural_LSTM, baseline float32)

This module contains an end-to-end pipeline for detecting Distributed Denial of Service (DoS) activity from PCAP captures using an LSTM-based sequence classifier. The latest iteration shares the exact 1 s / 0.5 s sliding-window features used by the CNN (micro-bin histograms, protocol counts, SSDP tokens) and exposes a single binary head: every downstream score, gate, and attribution is derived from that attack-vs-normal probability. The inference stage highlights the most suspicious MAC and IP addresses observed during an attack, making it easier to pivot from the prediction to actionable remediation.

## Variant Notes (Baseline)

- Full-precision (float32) training and inference only.
- `training.supervised.precision_mode` in `configs/config.yaml` is ignored in this variant.
- No int8 quantization support; use `../Neural_LSTM_16_mixedP` (mixed precision) or `../Neural_LSTM_int8` (dynamic int8).
- Run the commands below from the `Neural_LSTM/` directory.
- These commands assume `configs/config.yaml` uses local paths under this directory (`models/`, `reports/`, `data/processed/`).

## Quick Commands (Baseline)

```bash
python3 -m dos_detector.cli extract-features "pcaps/*.pcap" --out data/processed --config-path configs/config.yaml
python3 -m dos_detector.cli train-supervised --config-path configs/config.yaml
python3 -m dos_detector.cli infer path/to/capture.pcap --out reports/prediction.json --config-path configs/config.yaml
python3 -m dos_detector.cli batch-infer "pcaps/*.pcap" --out-dir reports --config-path configs/config.yaml
```
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

## Environment Setup

1. Create a virtual environment (Python 3.10+):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project in editable mode along with the CLI:
   ```bash
<<<<<<< HEAD
   pip install -e ..
=======
   pip install -e ..[all]
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
   ```
   The root `pyproject.toml` exposes the `dos-detector` console entrypoint used below.

## Data Processing Workflow

1. Place your raw PCAP files anywhere on disk (e.g. under `pcaps/`).
<<<<<<< HEAD
2. Extract sliding-window features and persist them as Parquet files:
   ```bash
   dos-detector extract-features "pcaps/*.pcap" --out data/processed --config configs/config.yaml
=======
2. Extract sliding-window features (1 s window, 0.5 s hop, 8 micro-bins) and persist them as Parquet files:
   ```bash
   python3 -m dos_detector.cli extract-features "pcaps/*.pcap" --out data/processed --config-path configs/config.yaml
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
   ```
   - The manifest (`configs/config.yaml` ➜ `paths.manifest_path`) is updated with the feature column ordering used during training/inference.
   - Use the `DOS_LIMIT_PKTS` environment variable to cap packets per PCAP when debugging.

## Training

1. **Supervised detector (sequence classifier):**
   ```bash
<<<<<<< HEAD
   dos-detector train-supervised --config configs/config.yaml
   ```
Model artefacts, scalers, and metrics are written to the locations defined under `configs/config.yaml::paths`.
=======
   python3 -m dos_detector.cli train-supervised --config-path configs/config.yaml
   ```
Model artefacts, scalers, and metrics are written to the locations defined under `configs/config.yaml::paths`. Regenerate features and retrain whenever you update the shared feature definitions so the scaler stays in sync with the binary head.
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

## Inference

Run inference on a single PCAP:
```bash
<<<<<<< HEAD
dos-detector infer path/to/capture.pcap --out reports/prediction.json --config configs/config.yaml
```

The resulting JSON report includes:
- `final_decision`: `"attack"` or `"normal"`.
- `predicted_family`: dominant attack family among triggered windows.
=======
python3 -m dos_detector.cli infer path/to/capture.pcap --out reports/prediction.json --config-path configs/config.yaml
```

The resulting JSON report includes:
- `final_decision`: `"attack"` or `"normal"` from the LSTM binary head.
- `predicted_family`: currently mirrors the binary decision; kept for compatibility with earlier multi-head outputs.
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
- `host_activity`: ranked MAC/IP activity with per-window packet/score statistics.
- `most_suspicious_mac` / `most_suspicious_ip`: top-ranked source MAC/IP based on fused scores in attack windows.
The CLI also prints a one-line summary such as

```text
[mixed1.pcap] decision=attack max_prob=0.990044 num_attack_windows=6741 top_mac=14:cc:20:51:33:e9 top_ip=192.168.0.42
```
<<<<<<< HEAD
The CLI also supports batch mode via `dos-detector batch-infer "pcaps/*.pcap"`.
=======
The CLI also supports batch mode:
```bash
python3 -m dos_detector.cli batch-infer "pcaps/*.pcap" --out-dir reports --config-path configs/config.yaml
```
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

## Testing

Activate the virtual environment and run:
```bash
.venv/bin/python -m pytest
```
<<<<<<< HEAD
New tests cover the host activity aggregation logic introduced for MAC/IP attribution.
=======
Regression tests cover feature extraction, training, and host attribution to ensure the shared feature format and binary head remain consistent.
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7

## Notes on Suspicious Host Attribution

- MAC and IP statistics are derived from source fields per inference window and weighted by the supervised attack probability. Windows that pass the decision gate contribute additional weight, ensuring confirmed attack activity dominates the ranking.
- When no MAC/IP surpasses the attack threshold, the algorithm still reports the highest-scoring talkers to aid manual triage.
- All values in `host_activity` are JSON-friendly, enabling downstream tooling to consume the enriched telemetry directly.
