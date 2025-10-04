# PCAP DoS Detector

This repository provides a production-ready pipeline for detecting distributed denial-of-service (DoS) activity in packet capture (PCAP) files. The detector focuses on sequence models (BiLSTM/GRU) and a sequence autoencoder to achieve high recall on known DoS behaviors (SSDP, TCP SYN, ICMP, UDP, HTTP floods) while aggressively controlling false positives on benign captures.

## Highlights

- **Sequence windowing**: fixed-duration packet windows (default 1.0 s) with configurable hop (default 0.5 s) aggregated into multi-minute sequences.
- **Rich feature engineering**: statistical, structural, and protocol-specific features, including SSDP multicast coverage, SYN surplus, ICMP TTL variance, and rolling change statistics.
- **Supervised detector**: configurable BiLSTM/GRU backbone with binary and multi-class attack heads plus optional temporal attention.
- **Sequence autoencoder**: LSTM encoder-decoder trained on normal traffic to surface anomalous behavior via reconstruction error.
- **Score fusion**: logistic combiner of supervised probabilities, autoencoder anomalies, and plausibility hints with configurable decision gating.
- **Explainability**: attention weights, SHAP insights for flagged sequences, and rich evaluation reports.
- **CLI**: Typer-powered workflow for feature extraction, training, and inference. Supports offline PCAP analysis and optional live sniffing via Scapy.

## Repository layout

```
├── configs/            # YAML configuration files
├── data/processed/     # Cached features, scalers, manifest files
├── models/             # Trained weights and calibrators
├── pcaps/              # Example PCAPs / user-provided captures
├── reports/            # Evaluation artifacts and inference outputs
├── scripts/            # Utility scripts (calibration, report helpers)
├── src/dos_detector/   # Source package
└── tests/              # Pytest-based regression and smoke tests
```

## Quick start

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install .[dev]
   ```

2. **Extract features** from PCAPs (labels optional):

   ```bash
   dos-detector extract-features --pcaps "pcaps/*.pcap" --out data/processed/
   ```

3. **Train models** using the default configuration:

   ```bash
   dos-detector train-supervised --config configs/config.yaml
   dos-detector train-ae --config configs/config.yaml
   dos-detector calibrate-fusion --config configs/config.yaml
   ```

4. **Infer on new captures**:

   ```bash
   dos-detector infer --pcap path/to/capture.pcap --out reports/prediction.json
   ```

5. **Batch inference** over a directory:

   ```bash
   dos-detector batch-infer --pcaps "pcaps/*.pcap" --out reports/
   ```

## Identify suspicious actors from feature CSVs

Use the PyTorch-based CLI to run anomaly scoring on pre-extracted features and surface the most suspicious MAC/IP actors:

```bash
python scripts/detect_and_identify.py \
  --csv samples/example_extracted_features.csv \
  --model-path models/dosnet_best.pt \
  --scaler-path models/scaler.joblib \
  --batch-size 8192 \
  --alpha-model 0.6 \
  --alpha-behavior 0.4 \
  --window-sec 5 \
  --min-windows 2 \
  --window-score-threshold 0.6 \
  --mac-score-thresh 0.7 \
  --device cpu
```

The command prints a ranked table of actors, saves CSV/JSON summaries under `results/`, and emits an additional `results/identify_summary.json` bundle for downstream automation.

## Configuration

All tunable parameters (window size, hop, feature toggles, model dimensions, training hyperparameters, gating thresholds) live in `configs/config.yaml`. Edit the YAML or supply overrides via environment variables to adapt to new datasets or operational requirements.

## Tests

Unit tests validate feature extraction, labeling, deterministic seeding, and smoke coverage for the CLI. Run them with:

```bash
pytest
```

## Reports

Evaluation summaries, calibration tables, and per-PCAP explanations are written into the `reports/` directory. This includes confusion matrices, per-family metrics, and top false-positive/false-negative analyses.

## License

This project is provided without a formal license for internal evaluation.
