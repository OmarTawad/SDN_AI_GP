from pathlib import Path

from dos_detector.config import load_config
from dos_detector.data.processor import FeaturePipeline
from dos_detector.inference.pipeline import InferencePipeline
from dos_detector.training.autoencoder_trainer import AutoencoderTrainer
from dos_detector.training.supervised_trainer import SupervisedTrainer


def test_end_to_end_training(tmp_path):
    config = load_config(Path("configs/config.yaml"))
    config.paths.processed_dir = tmp_path / "processed"
    config.paths.models_dir = tmp_path / "models"
    config.paths.reports_dir = tmp_path / "reports"
    config.paths.scaler_path = config.paths.models_dir / "scaler.joblib"
    config.paths.supervised_model_path = config.paths.models_dir / "supervised.pt"
    config.paths.ae_model_path = config.paths.models_dir / "autoencoder.pt"
    config.paths.ae_scaler_path = config.paths.models_dir / "ae_scaler.joblib"
    config.paths.fusion_model_path = config.paths.models_dir / "fusion.joblib"
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
    config.training.autoencoder.max_epochs = 1
    config.training.autoencoder.max_train_batches = 1
    config.training.autoencoder.max_val_batches = 1

    config.data.train_files = ["example_ssdp_attack.pcap", "example_normal.pcap"]
    config.data.val_files = ["example_ssdp_attack.pcap"]
    config.data.test_files = []

    pipeline = FeaturePipeline(config)
    pipeline.process_files(
        [
            Path("pcaps/example_normal.pcap"),
            Path("pcaps/example_ssdp_attack.pcap"),
        ],
        config.paths.processed_dir,
    )

    SupervisedTrainer(config).train()
    AutoencoderTrainer(config).train()

    inference = InferencePipeline(config)
    report = inference.infer(Path("pcaps/example_ssdp_attack.pcap"))
    assert report["pcap"] == "example_ssdp_attack.pcap"
    assert report["final_decision"] in {"attack", "normal"}
    assert report["windows"]
