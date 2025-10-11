# ARP Spoofing Detector (LSTM)

This project provides an end-to-end pipeline for spotting ARP spoofing activity in large PCAP captures. Traffic is aggregated into 1 s windows (0.5 s hop), transformed into ARP-centric statistical features, and passed through a bi-directional LSTM with a single binary head. During inference the detector reports the final attack probability together with the MAC address that most consistently advertises conflicting ARP claims, enabling quick remediation even when captures exceed hundreds of megabytes.

## Environment Setup

1. Create and activate a virtual environment (Python 3.10+):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package in editable mode so the CLI is available:
   ```bash
   pip install -e ..
   ```
   The root `pyproject.toml` exposes the `arp-detector` console entrypoint used below; you can also invoke the CLI with `python -m arp_detector.cli`.

## Data Preparation Workflow

1. Place ARP-focused PCAP files under any directory (for example `pcaps/`). Large captures (≈800 MB) are supported; the feature extractor streams packets and only retains lightweight per-window summaries.
2. Extract features (1 s window, 0.5 s hop) and persist them as Parquet:
   ```bash
   python -m arp_detector.cli extract-features "pcaps/*.pcap" --out data/processed --config-path configs/config.yaml
   ```
   - The manifest referenced under `configs/config.yaml::paths.manifest_path` is updated with the feature column ordering used for both training and inference.
   - Set `ARP_LIMIT_PKTS` (environment variable) to cap packets per capture during quick experiments.

## Training

Train the supervised LSTM head once feature files exist:
```bash
python -m arp_detector.cli train-supervised --config-path configs/config.yaml
```
Model weights, the feature scaler, and training metrics are written to the locations defined in the configuration (`paths.models_dir`, `paths.metrics_path`, …). Adjust `configs/fast.yaml` for a lower-resource run when experimenting on a 2 vCPU system.

## Inference

Run inference on a single capture:
```bash
python -m arp_detector.cli infer path/to/capture.pcap --out reports/prediction.json --config-path configs/config.yaml
```

The resulting JSON report contains:
- `final_decision`: `"attack"` or `"normal"` from the binary head.
- `max_prob`: highest window probability observed.
- `host_activity.macs`: ranked MAC statistics including the ARP IPs each MAC claimed (`claimed_ips`).
- `most_suspicious_mac` / `most_suspicious_ip`: top ranked attacker according to the weighted ARP claim conflicts.

Batch mode is also available:
```bash
python -m arp_detector.cli batch-infer "pcaps/*.pcap" --out-dir reports --config-path configs/config.yaml
```

## Testing

With the virtual environment active:
```bash
python -m pytest
```
The test suite covers feature extraction and window labelling to ensure the ARP-specific pipeline remains stable across refactors.

## Notes on Suspicious Host Attribution

- Only ARP frames contribute to the MAC/IP activity tables; non-ARP traffic is ignored to reduce noise on large captures.
- Conflicting ARP claims (multiple MAC addresses asserting ownership of the same IP, or a single MAC claiming many IPs) increase the suspicion score and drive the `claimed_ips` breakdown inside the report.
- When no window crosses the detection threshold, the detector still reports the highest-scoring MAC/IP pairs so analysts can follow up manually.
