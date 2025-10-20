"""Feature metadata and tensor preparation helpers for the gateway pipeline."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence

import joblib
import numpy as np
import yaml
from dosdet.data.packet_to_frame import scapy_pkt_to_row as dos_scapy_pkt_to_row
from dosdet.features.feature_slimming import StaticSlimmer
from dosdet.features.scaler import RobustScaler
from dosdet.features.seq_features import compute_sequence_features as compute_dos_sequence_features
from dosdet.features.static_features import compute_static_features as compute_dos_static_features
from Neural_LSTM.src.dos_detector.config import load_config as load_dos_lstm_config
from Neural_LSTM.src.dos_detector.features.feature_engineering import FeatureExtractor as DosFeatureExtractor
from Neural_LSTM.src.dos_detector.data.structures import PacketRecord as DosPacketRecord, Window as DosWindow
from ARP_LSTM.src.arp_detector.config import load_config as load_arp_lstm_config
from ARP_LSTM.src.arp_detector.features.feature_engineering import FeatureExtractor as ArpFeatureExtractor
from ARP_LSTM.src.arp_detector.data.structures import PacketRecord as ArpPacketRecord, Window as ArpWindow
from arpdet.data.packet_to_frame import scapy_pkt_to_row as arp_scapy_pkt_to_row
from arpdet.features.feature_slimming import StaticSlimmer as ArpStaticSlimmer
from arpdet.features.scaler import RobustScaler as ArpRobustScaler
from arpdet.features.seq_features import compute_sequence_features as compute_arp_sequence_features
from arpdet.features.static_features import compute_static_features as compute_arp_static_features

from .env import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Project artefacts and feature metadata
# ---------------------------------------------------------------------------

AUTO_ARTIFACT_DIR = PROJECT_ROOT / "autoencoder" / "data" / "artifacts"
DOS_CNN_ARTIFACT_DIR = PROJECT_ROOT / "dosdet" / "artifacts"
ARP_CNN_ARTIFACT_DIR = PROJECT_ROOT / "arpdet" / "artifacts"
DOS_LSTM_MODEL_DIR = PROJECT_ROOT / "Neural_LSTM" / "models"
ARP_LSTM_MODEL_DIR = PROJECT_ROOT / "ARP_LSTM" / "models"
DOS_LSTM_CONFIG_PATH = PROJECT_ROOT / "Neural_LSTM" / "configs" / "config.yaml"
ARP_LSTM_CONFIG_PATH = PROJECT_ROOT / "ARP_LSTM" / "configs" / "config.yaml"
DOS_CONFIG_PATH = PROJECT_ROOT / "dosdet" / "config.yaml"
ARP_CONFIG_PATH = PROJECT_ROOT / "arpdet" / "config.yaml"

AUTO_MODEL_CONFIG = json.loads((AUTO_ARTIFACT_DIR / "model_config.json").read_text())
AUTO_FEATURE_NAMES: List[str] = list(AUTO_MODEL_CONFIG.get("feature_names", []))
AUTO_LOG_FEATURES: List[str] = list(AUTO_MODEL_CONFIG.get("log_features", []))
AUTO_FEATURE_INDEX = {name: idx for idx, name in enumerate(AUTO_FEATURE_NAMES)}
AUTO_CLIP_BOUNDS = json.loads((AUTO_ARTIFACT_DIR / "clip_bounds.json").read_text())
AUTO_CLIP_LOWER: Dict[str, float] = {k: float(v) for k, v in AUTO_CLIP_BOUNDS.get("lower", {}).items()}
AUTO_CLIP_UPPER: Dict[str, float] = {k: float(v) for k, v in AUTO_CLIP_BOUNDS.get("upper", {}).items()}
AUTO_SCALER = joblib.load(AUTO_ARTIFACT_DIR / "scaler.pkl")

with DOS_CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    DOS_CONFIG = yaml.safe_load(cfg_file)
with ARP_CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    ARP_CONFIG = yaml.safe_load(cfg_file)

TOP_UDP_PORTS: List[int] = [int(p) for p in DOS_CONFIG["data"]["top_k_udp_ports"]]
SSDP_MULTICAST_V4: str = DOS_CONFIG["features"]["ssdp_multicast_ipv4"]
SSDP_MULTICAST_V6: str = DOS_CONFIG["features"]["ssdp_multicast_ipv6"]

DOS_CNN_SCALER = RobustScaler.load(str(DOS_CNN_ARTIFACT_DIR))
DOS_CNN_SLIMMER = StaticSlimmer()
DOS_CNN_SLIMMER.load(str(DOS_CNN_ARTIFACT_DIR))
DOS_META = json.loads((DOS_CNN_ARTIFACT_DIR / "feature_model_meta.json").read_text())
DOS_CNN_SEQ_IN_DIM = int(DOS_META.get("seq_in_dim", 14))
DOS_CNN_STATIC_DIM = int(DOS_META.get("static_dim", 40))
DOS_MICRO_BINS = int(DOS_META.get("micro_bins", DOS_CONFIG["windowing"].get("micro_bins", 8)))

ARP_CNN_SCALER = ArpRobustScaler.load(str(ARP_CNN_ARTIFACT_DIR))
ARP_CNN_SLIMMER = ArpStaticSlimmer()
ARP_CNN_SLIMMER.load(str(ARP_CNN_ARTIFACT_DIR))
ARP_META = json.loads((ARP_CNN_ARTIFACT_DIR / "feature_model_meta.json").read_text())
ARP_CNN_SEQ_IN_DIM = int(ARP_META.get("seq_in_dim", 12))
ARP_CNN_STATIC_DIM = int(ARP_META.get("static_dim", 31))
ARP_MICRO_BINS = int(ARP_META.get("micro_bins", ARP_CONFIG["windowing"].get("micro_bins", 8)))

DOS_LSTM_SCALER = joblib.load(DOS_LSTM_MODEL_DIR / "feature_scaler.joblib")
DOS_LSTM_CFG = load_dos_lstm_config(DOS_LSTM_CONFIG_PATH)
DOS_FEATURE_EXTRACTOR = DosFeatureExtractor(DOS_LSTM_CFG.feature, DOS_LSTM_CFG.windowing.window_size)
DOS_LSTM_SEQUENCE_LENGTH = int(DOS_LSTM_CFG.windowing.sequence_length)
DOS_WINDOW_SIZE = float(DOS_LSTM_CFG.windowing.window_size)
DOS_WINDOW_STRIDE = float(DOS_LSTM_CFG.windowing.hop_size)

ARP_LSTM_SCALER = joblib.load(ARP_LSTM_MODEL_DIR / "feature_scaler.joblib")
ARP_LSTM_CFG = load_arp_lstm_config(ARP_LSTM_CONFIG_PATH)
ARP_FEATURE_EXTRACTOR = ArpFeatureExtractor(ARP_LSTM_CFG.feature, ARP_LSTM_CFG.windowing.window_size)
ARP_LSTM_SEQUENCE_LENGTH = int(ARP_LSTM_CFG.windowing.sequence_length)
ARP_WINDOW_SIZE = float(ARP_LSTM_CFG.windowing.window_size)
ARP_WINDOW_STRIDE = float(ARP_LSTM_CFG.windowing.hop_size)

if abs(DOS_WINDOW_SIZE - ARP_WINDOW_SIZE) > 1e-6:
    raise ValueError("DoS and ARP window sizes differ; update the streaming logic to handle mismatched windows.")

WINDOW_SIZE = DOS_WINDOW_SIZE
WINDOW_STRIDE = DOS_WINDOW_STRIDE


def prepare_auto_tensor(stats: Dict[str, float]) -> np.ndarray:
    values = np.zeros(len(AUTO_FEATURE_NAMES), dtype=np.float32)
    for name, idx in AUTO_FEATURE_INDEX.items():
        values[idx] = float(stats.get(name, 0.0))
    for name in AUTO_LOG_FEATURES:
        idx = AUTO_FEATURE_INDEX.get(name)
        if idx is None:
            continue
        values[idx] = math.log1p(max(values[idx], 0.0))
    for name, idx in AUTO_FEATURE_INDEX.items():
        lower = AUTO_CLIP_LOWER.get(name)
        upper = AUTO_CLIP_UPPER.get(name)
        if lower is not None:
            values[idx] = max(values[idx], float(lower))
        if upper is not None:
            values[idx] = min(values[idx], float(upper))
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = AUTO_SCALER.transform(values.reshape(1, -1)).astype(np.float32)
    return scaled.squeeze(0)


def prepare_dos_static(static_vec: np.ndarray) -> np.ndarray:
    names_stub = [f"f_{i}" for i in range(static_vec.shape[0])]
    scaled = DOS_CNN_SCALER.transform(static_vec.reshape(1, -1), names_stub)
    slim = DOS_CNN_SLIMMER.transform(scaled)
    return slim.astype(np.float32).squeeze(0)


def prepare_arp_static(static_vec: np.ndarray) -> np.ndarray:
    names_stub = [f"f_{i}" for i in range(static_vec.shape[0])]
    scaled = ARP_CNN_SCALER.transform(static_vec.reshape(1, -1), names_stub)
    slim = ARP_CNN_SLIMMER.transform(scaled)
    return slim.astype(np.float32).squeeze(0)


__all__ = [
    "prepare_auto_tensor",
    "prepare_dos_static",
    "DOS_FEATURE_EXTRACTOR",
    "prepare_arp_static",
    "AUTO_FEATURE_INDEX",
    "AUTO_FEATURE_NAMES",
    "AUTO_LOG_FEATURES",
    "DOS_LSTM_SCALER",
    "DOS_LSTM_SEQUENCE_LENGTH",
    "DOS_MICRO_BINS",
    "DOS_WINDOW_SIZE",
    "DOS_WINDOW_STRIDE",
    "DOS_META",
    "DOS_CNN_SEQ_IN_DIM",
    "DOS_CNN_STATIC_DIM",
    "TOP_UDP_PORTS",
    "SSDP_MULTICAST_V4",
    "SSDP_MULTICAST_V6",
    "compute_dos_sequence_features",
    "compute_dos_static_features",
    "dos_scapy_pkt_to_row",
    "ARP_LSTM_SCALER",
    "ARP_LSTM_SEQUENCE_LENGTH",
    "ARP_MICRO_BINS",
    "ARP_WINDOW_SIZE",
    "ARP_WINDOW_STRIDE",
    "compute_arp_sequence_features",
    "compute_arp_static_features",
    "ARP_FEATURE_EXTRACTOR",
    "arp_scapy_pkt_to_row",
    "WINDOW_SIZE",
    "WINDOW_STRIDE",
]
