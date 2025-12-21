"""Reporting helpers for training runs."""

from __future__ import annotations

from typing import Dict

from gateway.core import class_id_to_name


def summarise(dataset) -> None:
    print("\nPCAP summary:")
    stats = getattr(dataset, "stats", {})
    skipped = getattr(dataset, "skipped", {})
    if not stats:
        print(" (no statistics available)")
        return

    label_totals: Dict[int, int] = {}
    for path, info in stats.items():
        skip_reason = info.get("skipped_reason") or skipped.get(path)
        if skip_reason:
            print(f" - {path}: skipped ({skip_reason})")
            continue
        label_id = int(info.get("label", 0))
        label_name = info.get("label_name", class_id_to_name(label_id))
        windows = int(info.get("windows", 0))
        batches = int(info.get("batches", 0))
        truncated = int(info.get("truncated_windows", 0))
        line = f" - {path}: label={label_name}, windows={windows}, batches={batches}"
        if truncated:
            line += f", truncated={truncated}"
        print(line)
        label_totals[label_id] = label_totals.get(label_id, 0) + windows

    print("\nWindow counts by class:")
    for label_id, count in sorted(label_totals.items()):
        print(f" * {class_id_to_name(label_id)}: {count} windows")


__all__ = ["summarise"]

