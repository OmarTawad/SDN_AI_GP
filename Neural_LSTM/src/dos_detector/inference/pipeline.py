from __future__ import annotations
import os
"""Inference pipeline for PCAP DoS detection."""


from ..utils.progress import progress

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import SequenceDataset, collate_fn
from ..data.processor import FeaturePipeline
from ..fusion.calibrator import ScoreFusion
from ..models.autoencoder import LSTMAutoencoder
from ..models.supervised import SequenceClassifier
from ..utils.io import load_joblib, load_json
from ..utils.logging import configure_logging, get_logger
from ..utils.seed import seed_everything
from .postprocessing import DecisionGate


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
        self.index_to_family = {index: name for name, index in self.family_mapping.items()}
        self.supervised_model = self._load_supervised_model()
        self.autoencoder = self._load_autoencoder()
        self.scaler = load_joblib(config.paths.scaler_path)
        self.ae_scaler = load_joblib(config.paths.ae_scaler_path)
        self.fusion = self._load_fusion()
        self.metrics = load_json(config.paths.metrics_path) if config.paths.metrics_path.exists() else {}
        self.ae_baseline = self._extract_ae_baseline()
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

    def _load_autoencoder(self) -> LSTMAutoencoder:
        model = LSTMAutoencoder(len(self.feature_columns), self.config.model.autoencoder).to(self.device)
        state = torch.load(self.config.paths.ae_model_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def _load_fusion(self) -> ScoreFusion | None:
        if self.config.paths.fusion_model_path.exists():
            return ScoreFusion.load(self.config.paths.fusion_model_path)
        return None

    def _extract_ae_baseline(self) -> Dict[str, float]:
        history = self.metrics.get("autoencoder_history", [])
        if history:
            return history[-1]
        return {"ae_error_mean": 0.0, "ae_error_std": 1.0}

    def infer(self, path: Path) -> Dict[str, object]:
        frame, meta = self.feature_pipeline.process_single(path)
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
            }
        features_supervised = frame.copy()
        features_autoencoder = frame.copy()
        features_supervised[self.feature_columns] = self.scaler.transform(frame[self.feature_columns])
        features_autoencoder[self.feature_columns] = self.ae_scaler.transform(frame[self.feature_columns])

        sup_dataset = SequenceDataset([features_supervised], self.feature_columns, self.family_mapping, self.config.windowing)
        ae_dataset = SequenceDataset([features_autoencoder], self.feature_columns, self.family_mapping, self.config.windowing)
        sup_loader = DataLoader(sup_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True)
        ae_loader = DataLoader(ae_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True)

        window_store = self._run_supervised(frame, sup_loader)
        self._run_autoencoder(ae_loader, window_store)
        window_results = self._assemble_results(frame, window_store)
        gate_input = [
            {
                "index": entry["index"],
                "score": entry["fused_score"],
                "family": entry["family"],
                "features": entry.get("features", {}),
            }
            for entry in window_results
        ]
        decisions, file_attack = self.gate.apply(gate_input)
        final_family = self._dominant_family(decisions)
        explanation = {}
        if window_results:
            top_window = max(window_results, key=lambda item: item["fused_score"])
            explanation = self._explain_sequence(sup_dataset, int(top_window["index"]))
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
                    "supervised_prob": entry["supervised_prob"],
                    "ae_error": entry["ae_error"],
                    "ae_anomaly": entry["ae_anomaly"],
                    "fused_score": entry["fused_score"],
                    "family": entry["family"],
                    "decision": "attack" if any(dec.index == entry["index"] and dec.is_attack for dec in decisions) else "normal",
                }
                for entry in window_results
            ],
            "final_decision": "attack" if file_attack else "normal",
            "predicted_family": final_family,
            "max_score": max((entry["fused_score"] for entry in window_results), default=0.0),
            "explanation": explanation,
        }
        return report

    def _run_supervised(self, raw_frame: pd.DataFrame, loader: DataLoader) -> Dict[int, Dict[str, object]]:
        window_store: Dict[int, Dict[str, object]] = {}
        with torch.no_grad():
            for batch in progress(loader, desc="Inference", unit="batch", leave=False):
                features = batch["features"].to(self.device)
                outputs = self.supervised_model(features)
                probs = torch.sigmoid(outputs.window_logits).cpu().numpy()
                type_logits = F.softmax(outputs.type_logits, dim=-1).cpu().numpy()
                for i, meta in enumerate(batch["metadata"]):
                    start_index = meta["start_index"]
                    end_index = meta["end_index"]
                    for offset, window_index in enumerate(range(start_index, end_index + 1)):
                        info = window_store.setdefault(
                            window_index,
                            {
                                "supervised": [],
                                "type_probs": [],
                                "features": raw_frame[self.feature_columns].iloc[window_index].to_dict(),
                                "start": float(raw_frame["window_start"].iloc[window_index]),
                                "end": float(raw_frame["window_end"].iloc[window_index]),
                            },
                        )
                        info["supervised"].append(float(probs[i, offset]))
                        info["type_probs"].append(type_logits[i, offset])
        return window_store

    def _run_autoencoder(self, loader: DataLoader, window_store: Dict[int, Dict[str, object]]) -> None:
        errors: Dict[int, List[float]] = defaultdict(list)
        with torch.no_grad():
            for batch in progress(loader, desc="Inference", unit="batch", leave=False):
                features = batch["features"].to(self.device)
                reconstruction, _ = self.autoencoder(features)
                window_errors = LSTMAutoencoder.reconstruction_error(features, reconstruction).cpu().numpy()
                for i, meta in enumerate(batch["metadata"]):
                    start_index = meta["start_index"]
                    end_index = meta["end_index"]
                    for offset, window_index in enumerate(range(start_index, end_index + 1)):
                        errors[window_index].append(float(window_errors[i, offset]))
        for index, vals in errors.items():
            window_store.setdefault(index, {})["ae_errors"] = vals

    def _assemble_results(self, raw_frame: pd.DataFrame, window_store: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        baseline_mean = self.ae_baseline.get("ae_error_mean", 0.0)
        baseline_std = max(self.ae_baseline.get("ae_error_std", 1.0), 1e-6)
        for index in sorted(window_store.keys()):
            info = window_store[index]
            supervised_prob = float(np.mean(info.get("supervised", [0.0])))
            type_probs = np.mean(np.array(info.get("type_probs", [[0.0] * len(self.family_mapping)])), axis=0)
            if np.isnan(type_probs).any():
                type_probs = np.zeros(len(self.family_mapping))
                type_probs[0] = 1.0
            family_index = int(np.argmax(type_probs)) if type_probs.size else 0
            family_name = self.index_to_family.get(family_index, "normal")
            ae_error = float(np.mean(info.get("ae_errors", [0.0])))
            normalized = max(0.0, (ae_error - baseline_mean) / baseline_std)
            ae_anomaly = float(np.clip(normalized / 3.0, 0.0, 1.0))
            fusion_features = [supervised_prob, normalized, float(info.get("features", {}).get("ssdp_share", 0.0))]
            if self.fusion is not None:
                fused = self.fusion.predict_proba(fusion_features)
            else:
                fused = 0.6 * supervised_prob + 0.4 * ae_anomaly
            results.append(
                {
                    "index": index,
                    "start": float(info.get("start", raw_frame["window_start"].iloc[index])),
                    "end": float(info.get("end", raw_frame["window_end"].iloc[index])),
                    "supervised_prob": supervised_prob,
                    "ae_error": ae_error,
                    "ae_anomaly": ae_anomaly,
                    "fused_score": float(fused),
                    "family": family_name,
                    "features": info.get("features", {}),
                }
            )
        return results

    def _dominant_family(self, decisions: Sequence) -> str:
        counts = defaultdict(int)
        for decision in decisions:
            if decision.is_attack:
                counts[decision.family] += 1
        if not counts:
            return "normal"
        return max(counts.items(), key=lambda item: item[1])[0]

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
                score = torch.sigmoid(outputs.file_logits)
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


__all__ = ["InferencePipeline"]