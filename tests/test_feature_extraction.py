from pathlib import Path

from dos_detector.config import load_config
from dos_detector.data.processor import FeaturePipeline


def test_feature_pipeline_detects_ssdp():
    config = load_config(Path("configs/config.yaml"))
    config.windowing.max_windows = 20
    pipeline = FeaturePipeline(config)
    frame, meta = pipeline.process_single(Path("pcaps/example_ssdp_attack.pcap"))
    assert not frame.empty
    assert "ssdp_share" in frame.columns
    assert frame["ssdp_share"].max() > 0.1
    assert frame["attack"].max() == 1
    assert meta.packet_count > 0
