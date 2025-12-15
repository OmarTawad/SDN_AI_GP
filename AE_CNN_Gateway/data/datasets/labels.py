"""Label utilities used by dataset loaders.


"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from gateway.core import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID


def resolve_label_id(value: Any) -> int:
    """Coerce arbitrary label representations into canonical IDs."""

    if isinstance(value, (int, np.integer)):
        label = int(value)
        if label in CLASS_ID_TO_NAME:
            return label
        raise ValueError(f"Unsupported label id '{label}'.")
    if isinstance(value, (float, np.floating)):
        return resolve_label_id(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in CLASS_NAME_TO_ID:
            return CLASS_NAME_TO_ID[token]
        try:
            numeric = int(token)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported label name '{value}'.") from exc
        return resolve_label_id(numeric)
    raise TypeError(f"Cannot resolve label from value of type {type(value)}.")


def infer_label_from_tasks(task_labels: Dict[str, Any]) -> Optional[int]:
    """Infer label ID from task flags stored in cached metadata."""

    if not isinstance(task_labels, dict):
        return None
    try:
        arp_flag = int(round(float(task_labels.get("arp", 0))))
        dos_flag = int(round(float(task_labels.get("dos", 0))))
    except Exception:
        return None
    if arp_flag > 0:
        return CLASS_NAME_TO_ID["arp"]
    if dos_flag > 0:
        return CLASS_NAME_TO_ID["dos"]
    return CLASS_NAME_TO_ID["normal"]


def infer_label_from_metadata(meta: Dict[str, Any]) -> Optional[int]:
    """Best-effort label inference from cached metadata."""

    if not isinstance(meta, dict):
        return None
    if "class_label" in meta:
        try:
            return resolve_label_id(meta["class_label"])
        except (TypeError, ValueError):
            pass
    label_name = meta.get("label_name")
    if label_name is not None:
        try:
            return resolve_label_id(label_name)
        except (TypeError, ValueError):
            pass
    task_labels = meta.get("labels")
    if isinstance(task_labels, dict):
        inferred = infer_label_from_tasks(task_labels)
        if inferred is not None:
            return inferred
    return None


def coerce_targets_from_cache(raw_labels: Any, default_label: int, window_count: int) -> Tensor:
    """Standardise cached labels into a target tensor."""

    if isinstance(raw_labels, Tensor):
        tensor = raw_labels.detach().clone()
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
        tensor = tensor.reshape(-1).to(torch.long)
        if tensor.shape[0] < window_count:
            pad = torch.full((window_count - tensor.shape[0],), default_label, dtype=torch.long)
            tensor = torch.cat([tensor, pad], dim=0)
        elif tensor.shape[0] > window_count:
            tensor = tensor[:window_count]
        return tensor
    if isinstance(raw_labels, dict):
        targets = torch.full((window_count,), default_label, dtype=torch.long)
        if "dos" in raw_labels:
            dos_tensor = torch.as_tensor(raw_labels["dos"]).reshape(-1)
            limit = min(window_count, dos_tensor.shape[0])
            mask = dos_tensor[:limit].round().to(torch.long) > 0
            targets[:limit][mask] = CLASS_NAME_TO_ID["dos"]
        if "arp" in raw_labels:
            arp_tensor = torch.as_tensor(raw_labels["arp"]).reshape(-1)
            limit = min(window_count, arp_tensor.shape[0])
            mask = arp_tensor[:limit].round().to(torch.long) > 0
            targets[:limit][mask] = CLASS_NAME_TO_ID["arp"]
        return targets
    return torch.full((window_count,), default_label, dtype=torch.long)


__all__ = [
    "CLASS_ID_TO_NAME",
    "CLASS_NAME_TO_ID",
    "coerce_targets_from_cache",
    "infer_label_from_metadata",
    "infer_label_from_tasks",
    "resolve_label_id",
]

