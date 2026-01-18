"""Cache discovery utilities for MoE datasets.


"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from gateway.core import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, class_id_to_name
from gateway.data.datasets.labels import infer_label_from_metadata, resolve_label_id
from gateway.data.structures.pcap import PcapInfo
from gateway.env import CACHE_ROOT, SAMPLES_DIR

CLASS_SUBDIRS: Dict[int, str] = {label: name for label, name in CLASS_ID_TO_NAME.items()}


def _load_labels_from_csv(base_dir: Path) -> Dict[Path, Dict[str, Any]]:
    manifest_path = base_dir / "labels.csv"
    if not manifest_path.exists():
        return {}
    entries: Dict[Path, Dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("labels.csv must include a header row with at least 'filename' and 'label'.")
        name_fields = [field for field in ("filename", "file", "pcap", "path") if field in reader.fieldnames]
        if not name_fields:
            raise ValueError("labels.csv header must include one of: filename, file, pcap, path.")
        label_fields = [field for field in ("label", "class", "target") if field in reader.fieldnames]
        if not label_fields:
            raise ValueError("labels.csv header must include a 'label' (or class/target) column.")
        for row in reader:
            raw_name = next((row.get(field) for field in name_fields if row.get(field)), None)
            if not raw_name:
                continue
            raw_label = next((row.get(field) for field in label_fields if row.get(field) is not None), None)
            if raw_label is None:
                raise ValueError(f"Row for '{raw_name}' missing label column in labels.csv.")
            label_id = resolve_label_id(raw_label)
            candidate = Path(raw_name)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            if not candidate.exists():
                candidate = (base_dir / Path(raw_name).name).resolve()
            if not candidate.exists():
                print(f"[Manifest] Skipping '{raw_name}' because the PCAP file was not found.")
                continue
            entries[candidate] = {
                "label": label_id,
                "meta": {
                    "label_source": "labels.csv",
                    "label_name": class_id_to_name(label_id),
                    "manifest_path": str(manifest_path),
                },
            }
    return entries


def _load_labels_from_subdirs(base_dir: Path) -> Dict[Path, Dict[str, Any]]:
    entries: Dict[Path, Dict[str, Any]] = {}
    for label_id, folder_name in CLASS_SUBDIRS.items():
        folder = base_dir / folder_name
        if not folder.exists():
            continue
        for pcap in sorted(folder.rglob("*.pcap")):
            path = pcap.resolve()
            entries.setdefault(
                path,
                {
                    "label": label_id,
                    "meta": {
                        "label_source": "folder",
                        "folder": str(folder),
                        "label_name": class_id_to_name(label_id),
                    },
                },
            )
    return entries


def discover_pcaps(tasks: Optional[Sequence[str]] = None) -> List[PcapInfo]:
    """Discover labelled PCAP captures in the samples directory."""

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_labels_from_csv(SAMPLES_DIR)
    folder_records = _load_labels_from_subdirs(SAMPLES_DIR)
    for path, payload in folder_records.items():
        records.setdefault(path, payload)

    infos: List[PcapInfo] = []
    for path, payload in records.items():
        label = int(payload["label"])
        meta = dict(payload.get("meta", {}))
        infos.append(PcapInfo(path=path, label=label, meta=meta))

    infos.sort(key=lambda info: (info.path.name.lower(), str(info.path)))
    return infos


def tasks_slug(tasks: Sequence[str]) -> str:
    """Return a deterministic slug for a sequence of task identifiers."""

    return "-".join(sorted(tasks))


def load_cache_entries(
    cache_base: Path,
    files: Sequence[PcapInfo],
    tasks: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Path], List[Tuple[Path, str]]]:
    """Read cached tensor manifests for the requested tasks."""

    slug = tasks_slug(tasks)
    cache_dir = cache_base / slug
    entries: List[Dict[str, Any]] = []
    missing: List[Path] = []
    mismatched: List[Tuple[Path, str]] = []

    if not cache_dir.exists():
        missing.extend([info.path for info in files])
        return entries, missing, mismatched

    task_signature = tuple(sorted(tasks))
    if not files:
        for cache_path in sorted(cache_dir.glob("*.pt")):
            try:
                data = torch.load(cache_path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - IO failure
                mismatched.append((cache_path, f"failed to load cache ({exc})"))
                continue
            cache_tasks = tuple(sorted(data.get("tasks", [])))
            if cache_tasks and cache_tasks != task_signature:
                mismatched.append((cache_path, "cache tasks mismatch"))
                continue
            features = data.get("features")
            if not isinstance(features, dict) or not features:
                mismatched.append((cache_path, "cache missing feature tensors"))
                continue
            meta = data.get("meta", {})
            source_path = Path(meta.get("source_path", cache_path.stem))
            cache_label = infer_label_from_metadata(meta)
            if cache_label is None:
                cache_label = CLASS_NAME_TO_ID["normal"]
            info = PcapInfo(path=source_path, label=int(cache_label), meta=dict(meta))
            entries.append({"data": data, "path": source_path, "cache_path": cache_path, "info": info})
        return entries, missing, mismatched

    for info in files:
        cache_path = cache_dir / f"{info.path.name}.pt"
        if not cache_path.exists():
            missing.append(info.path)
            continue
        try:
            data = torch.load(cache_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - IO failure
            mismatched.append((info.path, f"failed to load cache ({exc})"))
            continue
        cache_tasks = tuple(sorted(data.get("tasks", [])))
        if cache_tasks and cache_tasks != task_signature:
            mismatched.append((info.path, "cache tasks mismatch"))
            continue
        features = data.get("features")
        if not isinstance(features, dict) or not features:
            mismatched.append((info.path, "cache missing feature tensors"))
            continue
        meta = data.get("meta", {})
        cache_label = infer_label_from_metadata(meta)
        if cache_label is not None:
            updated_meta = dict(info.meta)
            updated_meta.update(meta)
            info = PcapInfo(path=info.path, label=int(cache_label), meta=updated_meta)
        entries.append({"data": data, "path": info.path, "cache_path": cache_path, "info": info})
    return entries, missing, mismatched


__all__ = [
    "CACHE_ROOT",
    "CLASS_SUBDIRS",
    "discover_pcaps",
    "load_cache_entries",
    "tasks_slug",
]
