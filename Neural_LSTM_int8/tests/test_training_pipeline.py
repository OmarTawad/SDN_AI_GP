#tests/test_training_pipeline.py
from pathlib import Path

from dos_detector.config import load_config
import pandas as pd

from dos_detector.inference.pipeline import InferencePipeline
from dos_detector.training.supervised_trainer import SupervisedTrainer
from dos_detector.utils.io import save_json


def test_end_to_end_training(tmp_path):
    config = load_config(Path("configs/config.yaml"))
    config.paths.processed_dir = tmp_path / "processed"
    config.paths.models_dir = tmp_path / "models"
    config.paths.reports_dir = tmp_path / "reports"
    config.paths.scaler_path = config.paths.models_dir / "scaler.joblib"
    config.paths.supervised_model_path = config.paths.models_dir / "supervised.pt"
    config.paths.metrics_path = config.paths.reports_dir / "metrics.json"
    config.paths.manifest_path = config.paths.processed_dir / "manifest.json"

    for path in [
        config.paths.processed_dir,
        config.paths.models_dir,
        config.paths.reports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    config.windowing.sequence_length = 1
    config.windowing.sequence_stride = 1
    config.windowing.max_windows = 50

    config.training.supervised.max_epochs = 1
    config.training.supervised.max_train_batches = 1
    config.training.supervised.max_val_batches = 1

    config.data.train_files = ["synthetic.pcap"]
    config.data.val_files = ["synthetic.pcap"]
    config.data.test_files = []
    feature_columns = ["pkts_per_s", "bytes_per_s", "udp_1900_fraction"]
    frame = pd.DataFrame(
        {
            "pcap": ["synthetic.pcap"] * 4,
            "window_index": list(range(4)),
            "window_start": [float(i) for i in range(4)],
            "window_end": [float(i + 1) for i in range(4)],
            "attack": [0, 0, 1, 1],
            "family": ["normal", "normal", "ssdp", "ssdp"],
            "family_index": [0, 0, 1, 1],
            "pkts_per_s": [0.1, 0.2, 5.0, 7.5],
            "bytes_per_s": [100.0, 120.0, 900.0, 1100.0],
            "udp_1900_fraction": [0.0, 0.0, 0.85, 0.92],
        }
    )
    target = config.paths.processed_dir / "synthetic.parquet"
    frame.to_parquet(target)
    save_json(
        config.paths.manifest_path,
        {
            "feature_columns": feature_columns,
            "frames": [
                {
                    "pcap": "synthetic.pcap",
                    "rows": int(len(frame)),
                    "packet_count": int(len(frame)),
                    "duration": float(len(frame)),
                }
            ],
        },
    )

    SupervisedTrainer(config).train()
    inference = InferencePipeline(config)
    report = inference.infer(Path("pcaps/ssdp_flood1.pcap"))
    assert report["pcap"] == "ssdp_flood1.pcap"
    assert report["final_decision"] in {"attack", "normal"}
    assert report["windows"]
