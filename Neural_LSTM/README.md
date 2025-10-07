# DoS Detector (Neural_LSTM)

This module contains an end-to-end pipeline for detecting Distributed Denial of Service (DoS) activity from PCAP captures using an LSTM-based sequence classifier. The inference stage highlights the most suspicious MAC and IP addresses observed during an attack, making it easier to pivot from the prediction to actionable remediation.

## Environment Setup

1. Create a virtual environment (Python 3.10+):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project in editable mode along with the CLI:
   ```bash
   pip install -e ..
   ```
   The root `pyproject.toml` exposes the `dos-detector` console entrypoint used below.

## Data Processing Workflow

1. Place your raw PCAP files anywhere on disk (e.g. under `pcaps/`).
2. Extract sliding-window features and persist them as Parquet files:
   ```bash
   dos-detector extract-features "pcaps/*.pcap" --out data/processed --config configs/config.yaml
   ```
   - The manifest (`configs/config.yaml` ➜ `paths.manifest_path`) is updated with the feature column ordering used during training/inference.
   - Use the `DOS_LIMIT_PKTS` environment variable to cap packets per PCAP when debugging.

## Training

1. **Supervised detector (sequence classifier):**
   ```bash
   dos-detector train-supervised --config configs/config.yaml
   ```
Model artefacts, scalers, and metrics are written to the locations defined under `configs/config.yaml::paths`.

## Inference

Run inference on a single PCAP:
```bash
dos-detector infer path/to/capture.pcap --out reports/prediction.json --config configs/config.yaml
```

The resulting JSON report includes:
- `final_decision`: `"attack"` or `"normal"`.
- `predicted_family`: dominant attack family among triggered windows.
- `host_activity`: ranked MAC/IP activity with per-window packet/score statistics.
- `most_suspicious_mac` / `most_suspicious_ip`: top-ranked source MAC/IP based on fused scores in attack windows.
The CLI also prints a one-line summary such as

```text
[mixed1.pcap] decision=attack max_prob=0.990044 num_attack_windows=6741 top_mac=14:cc:20:51:33:e9 top_ip=192.168.0.42
```
The CLI also supports batch mode via `dos-detector batch-infer "pcaps/*.pcap"`.

## Testing

Activate the virtual environment and run:
```bash
.venv/bin/python -m pytest
```
New tests cover the host activity aggregation logic introduced for MAC/IP attribution.

## Notes on Suspicious Host Attribution

- MAC and IP statistics are derived from source fields per inference window and weighted by the supervised attack probability. Windows that pass the decision gate contribute additional weight, ensuring confirmed attack activity dominates the ranking.
- When no MAC/IP surpasses the attack threshold, the algorithm still reports the highest-scoring talkers to aid manual triage.
- All values in `host_activity` are JSON-friendly, enabling downstream tooling to consume the enriched telemetry directly.
