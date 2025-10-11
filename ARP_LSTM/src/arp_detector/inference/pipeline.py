"""Inference pipeline for ARP spoofing detection."""

from __future__ import annotations

from ..utils.progress import progress

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import SequenceDataset, collate_fn
from ..data.pcap_reader import PCAPMetadata
from ..data.processor import FeaturePipeline
from ..models.supervised import SequenceClassifier
from ..utils.io import load_dataframe, load_joblib, load_json
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import seed_everything
from .postprocessing import DecisionGate


def detect_conflicting_replies(host_activity: Dict[str, List[Dict[str, object]]]) -> Optional[Dict[str, object]]:
    """Return the MAC entry that responds with multiple IP claims, if any."""

    for entry in host_activity.get("macs", []) or []:
        if entry.get("reply_conflict_ip_count", 0) and entry["reply_conflict_ip_count"] >= 2:
            return entry
    return None


class InferencePipeline:
    """High-level orchestration for inference."""

    def __init__(self, config: Config) -> None:
        self.config = config
        configure_logging()
        seed_everything(config.seed, deterministic=False)
        self.logger = get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.manifest = load_json(config.paths.manifest_path)
        self.feature_columns: Sequence[str] = self.manifest.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("Feature manifest missing. Run extract-features first.")
        self.family_mapping = config.labels.family_mapping
        self.supervised_model = self._load_supervised_model()
        self.scaler = load_joblib(config.paths.scaler_path)
        self.gate = DecisionGate(config.postprocessing)
        self.feature_pipeline = FeaturePipeline(config)

    def _load_supervised_model(self) -> SequenceClassifier:
        model = SequenceClassifier(
            input_size=len(self.feature_columns),
            num_attack_types=len(self.family_mapping),
            config=self.config.model.supervised,
        ).to(self.device)
        state = torch.load(self.config.paths.supervised_model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def infer(self, path: Path) -> Dict[str, object]:
        cached = self._load_cached_features(path)
        if cached is None:
            frame, meta = self.feature_pipeline.process_single(path)
            host_maps = self.feature_pipeline.last_host_maps()
            self._maybe_cache_host_maps(path)
        else:
            frame, meta, host_maps = cached
        if frame.empty:
            return {
                "pcap": path.name,
                "windows": [],
                "final_decision": "normal",
                "predicted_family": "normal",
                "metadata": {
                    "packet_count": meta.packet_count,
                    "duration": meta.duration,
                },
                "most_suspicious_mac": None,
                "most_suspicious_ip": None,
                "host_activity": {"macs": [], "ips": []},
                "top_mac": None,
                "top_ip": None,
                "max_score": 0.0,
                "max_prob": 0.0,
                "attack_window_count": 0,
                "num_attack_windows": 0,
            }
        host_maps = host_maps or {"macs": {}, "ips": {}, "claims": {}, "reply_claims": {}}
        features_supervised = frame.copy()
        features_supervised[self.feature_columns] = self.scaler.transform(frame[self.feature_columns])

        sup_dataset = SequenceDataset([features_supervised], self.feature_columns, self.family_mapping, self.config.windowing)
        sup_loader = DataLoader(sup_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True)

        window_store = self._run_supervised(frame, sup_loader)
        window_results = self._assemble_results(frame, window_store)
        gate_input = [
            {
                "index": entry["index"],
                "score": entry["probability"],
            }
            for entry in window_results
        ]
        decisions, file_attack = self.gate.apply(gate_input)
        final_label = "arp_spoof" if file_attack else "normal"
        explanation = {}
        if window_results:
            top_window = max(window_results, key=lambda item: item["probability"])
            explanation = self._explain_sequence(sup_dataset, int(top_window["index"]))
        host_activity = self._compute_host_activity(host_maps, window_results, decisions)
        conflict_entry = detect_conflicting_replies(host_activity)
        if conflict_entry is not None:
            file_attack = True
        attack_window_count = sum(1 for decision in decisions if decision.is_attack)
        if conflict_entry is not None:
            attack_window_count = max(attack_window_count, int(conflict_entry.get("attack_windows", 0)) or 1)
        top_mac_entry = host_activity["macs"][0] if host_activity["macs"] else None
        if conflict_entry is not None:
            top_mac_entry = conflict_entry
        top_ip_entry = host_activity["ips"][0] if host_activity["ips"] else None
        max_score = max((entry["probability"] for entry in window_results), default=0.0)
        report = {
            "pcap": path.name,
            "metadata": {
                "packet_count": meta.packet_count,
                "duration": meta.duration,
                "start_time": meta.start_time,
                "end_time": meta.end_time,
            },
            "windows": [
                {
                    "index": entry["index"],
                    "start": entry["start"],
                    "end": entry["end"],
                    "probability": entry["probability"],
                    "decision": "attack" if any(dec.index == entry["index"] and dec.is_attack for dec in decisions) else "normal",
                }
                for entry in window_results
            ],
            "final_decision": "attack" if file_attack else "normal",
            "predicted_family": final_label,
            "max_score": max_score,
            "max_prob": max_score,
            "explanation": explanation,
            "most_suspicious_mac": top_mac_entry["mac"] if top_mac_entry else None,
            "most_suspicious_ip": (
                top_mac_entry.get("top_ip")
                if top_mac_entry and "top_ip" in top_mac_entry
                else (top_ip_entry["ip"] if top_ip_entry else None)
            ),
            "top_mac": top_mac_entry["mac"] if top_mac_entry else None,
            "top_ip": top_ip_entry["ip"] if top_ip_entry else None,
            "attack_window_count": attack_window_count,
            "num_attack_windows": attack_window_count,
            "host_activity": host_activity,
        }
        return report

    def _run_supervised(self, raw_frame: pd.DataFrame, loader: DataLoader) -> Dict[int, Dict[str, object]]:
        window_store: Dict[int, Dict[str, object]] = {}
        with torch.no_grad():
            for batch in progress(loader, desc="Inference", unit="batch", leave=False):
                features = batch["features"].to(self.device)
                outputs = self.supervised_model(features)
                probs = torch.sigmoid(outputs.window_logits).cpu().numpy()
                for i, meta in enumerate(batch["metadata"]):
                    start_index = meta["start_index"]
                    end_index = meta["end_index"]
                    for offset, window_index in enumerate(range(start_index, end_index + 1)):
                        info = window_store.setdefault(
                            window_index,
                            {
                                "supervised": [],
                                "features": raw_frame[self.feature_columns].iloc[window_index].to_dict(),
                                "start": float(raw_frame["window_start"].iloc[window_index]),
                                "end": float(raw_frame["window_end"].iloc[window_index]),
                            },
                        )
                        info["supervised"].append(float(probs[i, offset]))
        return window_store

    def _assemble_results(self, raw_frame: pd.DataFrame, window_store: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for index in sorted(window_store.keys()):
            info = window_store[index]
            supervised_prob = float(np.mean(info.get("supervised", [0.0])))
            results.append(
                {
                    "index": index,
                    "start": float(info.get("start", raw_frame["window_start"].iloc[index])),
                    "end": float(info.get("end", raw_frame["window_end"].iloc[index])),
                    "probability": supervised_prob,
                    "features": info.get("features", {}),
                }
            )
        return results

    def _explain_sequence(self, dataset: SequenceDataset, window_index: int) -> Dict[str, object]:
        top_k = self.config.explainability.top_features
        for sample in dataset.samples:
            start = sample.metadata.get("start_index")
            end = sample.metadata.get("end_index")
            if start is None or end is None:
                continue
            if int(start) <= window_index <= int(end):
                tensor = torch.tensor(sample.features, dtype=torch.float32, device=self.device).unsqueeze(0)
                tensor.requires_grad_(True)
                self.supervised_model.zero_grad(set_to_none=True)
                outputs = self.supervised_model(tensor)
                score = outputs.sequence_prob
                score.backward()
                gradients = tensor.grad.abs().squeeze(0)
                feature_scores = gradients.mean(dim=0).cpu().numpy()
                top_indices = np.argsort(feature_scores)[::-1][:top_k]
                top_features = [
                    {"name": self.feature_columns[idx], "score": float(feature_scores[idx])}
                    for idx in top_indices
                ]
                if outputs.attention is not None:
                    attention = outputs.attention.detach().cpu().squeeze(0).tolist()
                else:
                    attention = gradients.mean(dim=1).cpu().numpy().tolist()
                return {"top_features": top_features, "temporal_attention": attention}
        return {}

    def _load_cached_features(
        self, path: Path
    ) -> Optional[Tuple[pd.DataFrame, PCAPMetadata, Dict[str, Dict[int, Dict[str, int]]]]]:
        processed_dir = getattr(self.config.paths, "processed_dir", None)
        if processed_dir is None:
            return None
        feature_path = Path(processed_dir) / f"{path.stem}.parquet"
        if not feature_path.exists():
            return None
        frame = load_dataframe(feature_path)
        host_maps = self.feature_pipeline.load_host_maps(path)
        if host_maps is None:
            return None
        meta = self._metadata_from_frame(path, frame)
        return frame, meta, host_maps

    def _metadata_from_frame(self, path: Path, frame: pd.DataFrame) -> PCAPMetadata:
        entry = self._lookup_manifest_entry(path.name)
        packet_count = int(entry.get("packet_count", 0)) if entry else 0
        duration = float(entry.get("duration", 0.0)) if entry else 0.0
        start_time = float(frame["window_start"].min()) if "window_start" in frame.columns and not frame.empty else 0.0
        end_time = float(frame["window_end"].max()) if "window_end" in frame.columns and not frame.empty else start_time
        if entry is None and end_time > start_time:
            duration = end_time - start_time
        return PCAPMetadata(path=path, packet_count=packet_count, duration=duration, start_time=start_time, end_time=end_time)

    def _lookup_manifest_entry(self, name: str) -> Optional[Dict[str, object]]:
        for entry in self.manifest.get("frames", []):
            if entry.get("pcap") == name:
                return entry
        return None

    def _maybe_cache_host_maps(self, path: Path) -> None:
        processed_dir = getattr(self.config.paths, "processed_dir", None)
        if processed_dir is None:
            return
        self.feature_pipeline.save_last_host_maps(path)

    def _compute_host_activity(
        self,
        host_maps: Dict[str, object],
        window_results: Sequence[Dict[str, object]],
        decisions: Sequence[WindowDecision],
    ) -> Dict[str, List[Dict[str, object]]]:
        """Aggregate MAC/IP activity aligned with window-level scores."""

        if not window_results:
            return {"macs": [], "ips": []}

        mac_map = {
            int(idx): {mac: int(count) for mac, count in counts.items()}
            for idx, counts in (host_maps or {}).get("macs", {}).items()
        }
        ip_map = {
            int(idx): {ip: int(count) for ip, count in counts.items()}
            for idx, counts in (host_maps or {}).get("ips", {}).items()
        }
        claim_map = {
            int(idx): {
                mac: {ip: int(count) for ip, count in mac_claims.items()}
                for mac, mac_claims in counts.items()
            }
            for idx, counts in (host_maps or {}).get("claims", {}).items()
        }
        reply_claim_map = {
            int(idx): {
                mac: {ip: int(count) for ip, count in mac_claims.items()}
                for mac, mac_claims in counts.items()
            }
            for idx, counts in (host_maps or {}).get("reply_claims", {}).items()
        }
        decision_map = {decision.index: decision for decision in decisions}

        def _default_stats() -> Dict[str, float | int]:
            return {
                "total_packets": 0,
                "total_windows": 0,
                "score_sum": 0.0,
                "max_window_score": 0.0,
                "attack_packets": 0,
                "attack_windows": 0,
                "attack_score": 0.0,
                "max_attack_score": 0.0,
                "claimed_ips": {},
                "reply_claims": {},
            }

        mac_stats: defaultdict[str, Dict[str, float | int]] = defaultdict(_default_stats)
        ip_stats: defaultdict[str, Dict[str, float | int]] = defaultdict(_default_stats)

        for entry in window_results:
            index = int(entry["index"])
            probability = float(entry["probability"])
            decision = decision_map.get(index)
            is_attack = bool(decision.is_attack) if decision is not None else False
            mac_counts = mac_map.get(index, {})
            ip_counts = ip_map.get(index, {})
            window_claims = claim_map.get(index, {})
            window_reply_claims = reply_claim_map.get(index, {})

            for mac, count in mac_counts.items():
                stats = mac_stats[mac]
                stats["total_packets"] = int(stats["total_packets"]) + int(count)
                stats["total_windows"] = int(stats["total_windows"]) + 1
                stats["score_sum"] = float(stats["score_sum"]) + probability * int(count)
                stats["max_window_score"] = max(float(stats["max_window_score"]), probability)
                if is_attack:
                    stats["attack_packets"] = int(stats["attack_packets"]) + int(count)
                    stats["attack_windows"] = int(stats["attack_windows"]) + 1
                    stats["attack_score"] = float(stats["attack_score"]) + probability * int(count)
                    stats["max_attack_score"] = max(float(stats["max_attack_score"]), probability)
                claims = window_claims.get(mac, {})
                if claims:
                    claimed_ips = stats.setdefault("claimed_ips", {})
                    for ip, claim_count in claims.items():
                        claimed_ips[ip] = int(claimed_ips.get(ip, 0)) + int(claim_count)
                reply_claims = window_reply_claims.get(mac, {})
                if reply_claims:
                    reply_map = stats.setdefault("reply_claims", {})
                    for ip, claim_count in reply_claims.items():
                        reply_map[ip] = int(reply_map.get(ip, 0)) + int(claim_count)

            for ip, count in ip_counts.items():
                stats = ip_stats[ip]
                stats["total_packets"] = int(stats["total_packets"]) + int(count)
                stats["total_windows"] = int(stats["total_windows"]) + 1
                stats["score_sum"] = float(stats["score_sum"]) + probability * int(count)
                stats["max_window_score"] = max(float(stats["max_window_score"]), probability)
                if is_attack:
                    stats["attack_packets"] = int(stats["attack_packets"]) + int(count)
                    stats["attack_windows"] = int(stats["attack_windows"]) + 1
                    stats["attack_score"] = float(stats["attack_score"]) + probability * int(count)
                    stats["max_attack_score"] = max(float(stats["max_attack_score"]), probability)

        def _finalize(stats_dict: defaultdict[str, Dict[str, float | int]], key_name: str) -> List[Dict[str, object]]:
            ranked = sorted(
                stats_dict.items(),
                key=lambda item: (
                    float(item[1]["attack_score"]),
                    int(item[1]["attack_packets"]),
                    float(item[1]["score_sum"]),
                ),
                reverse=True,
            )
            results: List[Dict[str, object]] = []
            for key, stats in ranked:
                total_packets = int(stats["total_packets"])
                attack_packets = int(stats["attack_packets"])
                suspicion = float(stats["attack_score"])
                if suspicion <= 0.0:
                    suspicion = float(stats["score_sum"]) * 0.5
                suspicion += float(attack_packets) * 0.01
                total_windows = int(stats["total_windows"])
                attack_windows = int(stats["attack_windows"])
                claimed_ips = stats.get("claimed_ips", {})
                reply_claims = stats.get("reply_claims", {})
                ranked_claims = sorted(
                    claimed_ips.items(),
                    key=lambda item: (item[1],),
                    reverse=True,
                )
                top_ip = ranked_claims[0][0] if ranked_claims else None
                reply_claim_list = sorted(
                    reply_claims.items(),
                    key=lambda item: (item[1],),
                    reverse=True,
                )
                reply_conflict_ip_count = len([ip for ip, count in reply_claim_list if count])
                suspicion += reply_conflict_ip_count * 0.5
                if reply_claim_list and reply_claim_list[0][0]:
                    top_ip = reply_claim_list[0][0]
                results.append(
                    {
                        key_name: key,
                        "total_packets": total_packets,
                        "attack_packets": attack_packets,
                        "attack_windows": attack_windows,
                        "total_windows": total_windows,
                        "max_window_score": float(stats["max_window_score"]),
                        "max_attack_window_score": float(stats["max_attack_score"]),
                        "mean_probability": float(stats["score_sum"]) / max(total_packets, 1),
                        "mean_attack_score": float(stats["attack_score"]) / max(attack_packets, 1) if attack_packets else 0.0,
                        "attack_packet_ratio": attack_packets / max(total_packets, 1),
                        "suspicion_score": suspicion,
                        "claimed_ips": [
                            {"ip": ip, "count": int(count)}
                            for ip, count in ranked_claims[:5]
                        ],
                        "reply_claims": [
                            {"ip": ip, "count": int(count)}
                            for ip, count in reply_claim_list[:5]
                        ],
                        "reply_conflict_ip_count": reply_conflict_ip_count,
                        "top_ip": top_ip,
                    }
                )
            return results

        return {
            "macs": _finalize(mac_stats, "mac"),
            "ips": _finalize(ip_stats, "ip"),
        }


__all__ = ["InferencePipeline", "detect_conflicting_replies"]
