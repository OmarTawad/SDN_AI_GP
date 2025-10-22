"""Inference utility for the unified Mixture-of-Experts model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.data_pipeline import MoEDataset, PcapInfo, class_id_to_name  # noqa: E402
from gateway.unified_moe_model import (  # noqa: E402
    UNIFIED_EXPERT_SPECS,
    UNIFIED_GATING_INPUT_DIM,
    build_unified_moe,
)

NUM_CLASSES = 3
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "unified_moe.pt"
CLASS_LABELS: Tuple[str, ...] = ("normal", "dos", "arp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified MoE inference over a PCAP file.")
    parser.add_argument("pcap", type=str, help="Path to the PCAP file to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to a trained unified MoE checkpoint (default: gateway/unified_moe.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of windows per inference batch.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Optional cap on processed windows (0 disables the cap).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Torch thread pool size for feature extraction (default: 1).",
    )
    parser.add_argument(
        "--gating-hidden",
        type=int,
        default=None,
        help="Override the gating hidden size when loading the checkpoint (normally auto-detected).",
    )
    parser.add_argument(
        "--attack-threshold",
        type=float,
        default=0.15,
        help="Minimum fraction of windows required for a class to take the final verdict.",
    )
    parser.add_argument(
        "--attack-prob-threshold",
        type=float,
        default=0.30,
        help="Average class probability required to promote an attack when fractions are small.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.85,
        help="Per-window probability considered high confidence for escalation.",
    )
    parser.add_argument(
        "--min-high-confidence-windows",
        type=int,
        default=25,
        help="Minimum high-confidence windows required to promote an attack verdict.",
    )
    parser.add_argument(
        "--min-attack-windows",
        type=int,
        default=50,
        help="Minimum windows predicted for an attack class before escalation is considered.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, gating_hidden_override: int | None) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        gating_hidden = gating_hidden_override or int(payload.get("gating_hidden", 128))
        state_dict: Dict[str, torch.Tensor] = payload["state_dict"]
    else:
        gating_hidden = gating_hidden_override or 128
        state_dict = payload

    model = build_unified_moe(device=torch.device("cpu"), gating_hidden_dim=gating_hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _aggregate_predictions(predictions: Iterable[int]) -> Tuple[Dict[str, int], Dict[str, float], str]:
    counts: Dict[str, int] = {cls: 0 for cls in CLASS_LABELS}
    total = 0
    for pred in predictions:
        label_name = class_id_to_name(int(pred))
        counts[label_name] = counts.get(label_name, 0) + 1
        total += 1
    for cls in CLASS_LABELS:
        counts.setdefault(cls, 0)
    if total == 0:
        fractions = {cls: 0.0 for cls in CLASS_LABELS}
        verdict = "normal"
    else:
        fractions = {cls: counts.get(cls, 0) / total for cls in CLASS_LABELS}
        verdict = max(fractions.items(), key=lambda item: item[1])[0]
        if fractions.get(verdict, 0.0) < 0.15:
            verdict = "normal"
    return counts, fractions, verdict


def _decide_verdict(
    window_counts: Dict[str, int],
    fractions: Dict[str, float],
    probability_means: Dict[str, float],
    high_confidence_windows: Dict[str, int],
    attack_threshold: float,
    attack_prob_threshold: float,
    min_attack_windows: int,
    min_high_conf_windows: int,
) -> str:
    total_windows = sum(window_counts.values())
    if total_windows <= 0:
        return "normal"

    majority_class, majority_fraction = max(fractions.items(), key=lambda item: item[1])
    verdict = majority_class
    if majority_fraction < attack_threshold:
        verdict = "normal"

    if verdict == "normal":
        attack_candidates: List[Tuple[str, float, float, int]] = []
        for cls in CLASS_LABELS:
            if cls == "normal":
                continue
            attack_candidates.append(
                (
                    cls,
                    fractions.get(cls, 0.0),
                    probability_means.get(cls, 0.0),
                    high_confidence_windows.get(cls, 0),
                )
            )
        if attack_candidates:
            attack_candidates.sort(key=lambda item: (item[1], item[2], item[3]))
            candidate_class, candidate_fraction, candidate_prob, candidate_high_conf = attack_candidates[-1]
            candidate_count = window_counts.get(candidate_class, 0)

            meets_fraction = candidate_fraction >= attack_threshold
            meets_prob_windows = candidate_prob >= attack_prob_threshold and candidate_count >= min_attack_windows
            meets_high_conf = candidate_prob >= attack_prob_threshold and candidate_high_conf >= min_high_conf_windows

            if meets_fraction or meets_prob_windows or meets_high_conf:
                verdict = candidate_class

    # Ensure promotion criteria are still satisfied when verdict is an attack class.
    if verdict != "normal":
        fractional_support = fractions.get(verdict, 0.0)
        mean_prob = probability_means.get(verdict, 0.0)
        high_conf = high_confidence_windows.get(verdict, 0)
        window_support = window_counts.get(verdict, 0)
        if not (
            fractional_support >= attack_threshold
            or (
                mean_prob >= attack_prob_threshold
                and (window_support >= min_attack_windows or high_conf >= min_high_conf_windows)
            )
        ):
            verdict = "normal"

    return verdict


def _top_entries(counter: Counter[str], limit: int = 3) -> List[str]:
    candidates = [(key, float(value)) for key, value in counter.items() if key]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in candidates[:limit]]


def _extract_suspicious_entities(
    pcap_path: Path,
    metas: Iterable[Dict[str, Any]] | None,
    confidences: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    ip_counter: Counter[str] = Counter()
    mac_counter: Counter[str] = Counter()

    if metas is not None:
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            suspect_ip = meta.get("suspect_ip")
            if suspect_ip:
                ip_counter[str(suspect_ip)] += float(meta.get("score", 1.0))
            suspect_mac = meta.get("suspect_mac")
            if suspect_mac:
                mac_counter[str(suspect_mac).lower()] += float(meta.get("score", 1.0))

    try:
        from scapy.layers.inet import IP, TCP
        from scapy.layers.l2 import ARP
        from scapy.utils import PcapReader
    except Exception:
        return _top_entries(ip_counter), _top_entries(mac_counter)

    mac_to_ips: Dict[str, set[str]] = defaultdict(set)
    ip_to_macs: Dict[str, set[str]] = defaultdict(set)
    syn_sources: Counter[str] = Counter()
    packet_sources: Counter[str] = Counter()

    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                # MAC handling is deferred to ARP parsing below.

                ip_layer = pkt.getlayer(IP)
                src_ip = getattr(ip_layer, "src", None) if ip_layer is not None else None
                if src_ip:
                    src_ip_str = str(src_ip)
                    packet_sources[src_ip_str] += 1.0
                else:
                    src_ip_str = None

                tcp_layer = pkt.getlayer(TCP)
                if tcp_layer is not None and src_ip_str:
                    flags = int(getattr(tcp_layer, "flags", 0))
                    if flags & 0x02:  # SYN flag
                        if not (flags & 0x10):  # avoid counting SYN/ACK
                            syn_sources[src_ip_str] += 1.0

                arp_layer = pkt.getlayer(ARP)
                if arp_layer is not None:
                    sender_ip = str(getattr(arp_layer, "psrc", "") or "")
                    sender_mac = str(getattr(arp_layer, "hwsrc", "") or "").lower()
                    target_ip = str(getattr(arp_layer, "pdst", "") or "")
                    target_mac_raw = str(getattr(arp_layer, "hwdst", "") or "")
                    target_mac = target_mac_raw.lower() if target_mac_raw else ""

                    if sender_ip and sender_mac:
                        mac_to_ips[sender_mac].add(sender_ip)
                        ip_to_macs[sender_ip].add(sender_mac)
                    if target_ip and target_mac and target_mac not in {
                        "ff:ff:ff:ff:ff:ff",
                        "00:00:00:00:00:00",
                    }:
                        mac_to_ips[target_mac].add(target_ip)
                        ip_to_macs[target_ip].add(target_mac)
    except Exception:
        pass

    dos_weight = float(confidences.get("dos", 0.0))
    arp_weight = float(confidences.get("arp", 0.0))
    dos_factor = dos_weight if dos_weight > 0 else (0.1 if syn_sources else 0.0)
    arp_factor = arp_weight if arp_weight > 0 else (0.1 if mac_to_ips or ip_to_macs else 0.0)

    for ip_addr, count in syn_sources.items():
        ip_counter[ip_addr] += count * max(dos_factor, 1.0)
    if dos_weight > 0:
        for ip_addr, count in packet_sources.items():
            ip_counter[ip_addr] += 0.1 * count * dos_weight

    for mac_addr, ips in mac_to_ips.items():
        if mac_addr and len(ips) > 1:
            mac_counter[mac_addr] += float(len(ips)) * max(arp_factor, 1.0)
            for ip_addr in ips:
                if ip_addr:
                    ip_counter[ip_addr] += max(arp_factor, 1.0)

    for ip_addr, macs in ip_to_macs.items():
        if ip_addr and len(macs) > 1:
            ip_counter[ip_addr] += float(len(macs)) * max(arp_factor, 1.0)
            for mac_addr in macs:
                if mac_addr:
                    mac_counter[mac_addr] += 0.5 * max(arp_factor, 1.0)

    # Clean up placeholders
    for token in ("", None, "ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"):
        ip_counter.pop(token, None)
        mac_counter.pop(token, None)

    top_ips = _top_entries(ip_counter)
    top_macs = _top_entries(mac_counter)
    return top_ips, top_macs


def _save_window_csv(pcap_path: Path, headers: List[str], records: List[Dict[str, Any]]) -> Path:
    csv_path = pcap_path.with_suffix(".windows.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in records:
            writer.writerow(record["row"])
    return csv_path


def _save_summary_json(
    pcap_path: Path,
    verdict: str,
    confidences: Dict[str, float],
    probability_means: Dict[str, float],
    high_confidence_windows: Dict[str, int],
    window_counts: Dict[str, int],
    top_ips: List[str],
    top_macs: List[str],
    attack_threshold: float,
    attack_prob_threshold: float,
    high_confidence_threshold: float,
    min_attack_windows: int,
    min_high_conf_windows: int,
) -> Path:
    summary_payload: Dict[str, Any] = {
        "pcap": pcap_path.name,
        "verdict": verdict,
        "confidence": {cls: float(confidences.get(cls, 0.0)) for cls in CLASS_LABELS},
        "top_suspicious_ips": top_ips,
        "top_suspicious_macs": top_macs,
        "window_counts": {cls: int(window_counts.get(cls, 0)) for cls in CLASS_LABELS},
        "probability_mean": {cls: float(probability_means.get(cls, 0.0)) for cls in CLASS_LABELS},
        "high_confidence_windows": {cls: int(high_confidence_windows.get(cls, 0)) for cls in CLASS_LABELS},
        "high_confidence_threshold": float(high_confidence_threshold),
        "attack_threshold": float(attack_threshold),
        "attack_prob_threshold": float(attack_prob_threshold),
        "min_attack_windows": int(min_attack_windows),
        "min_high_confidence_windows": int(min_high_conf_windows),
    }
    for cls in CLASS_LABELS:
        summary_payload[f"confidence_{cls}"] = float(confidences.get(cls, 0.0))
    summary_payload["total_windows"] = int(sum(window_counts.get(cls, 0) for cls in CLASS_LABELS))

    json_path = pcap_path.with_suffix(".summary.json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
        handle.write("\n")
    return json_path


def _verdict_message(verdict: str) -> str:
    verdict_lower = verdict.lower()
    if verdict_lower == "dos":
        return "DOS attack detected"
    if verdict_lower == "arp":
        return "ARP spoofing detected"
    return "Traffic appears normal"


def run_inference(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, args.num_threads))

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = load_model(checkpoint_path, gating_hidden_override=args.gating_hidden)

    pcap_path = Path(args.pcap).resolve()
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    info = PcapInfo(path=pcap_path, label=0)
    max_windows = args.max_windows if args.max_windows > 0 else None
    dataset = MoEDataset(
        files=[info],
        tasks=("dos", "arp"),
        batch_size=args.batch_size,
        shuffle=False,
        seed=42,
        max_windows_per_file=max_windows,
        max_total_windows=max_windows,
        status_interval=None,
        max_file_size=None,
        max_packets_per_file=None,
        file_timeout=None,
        max_packets_per_window=None,
    )
    dataset.set_window_budget(max_windows)
    dataset.set_log_fn(None)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    expert_names: List[str] = [spec.name for spec in UNIFIED_EXPERT_SPECS]

    print(
        f"[Info] Loaded checkpoint '{checkpoint_path.name}' | gating_input={UNIFIED_GATING_INPUT_DIM} "
        f"experts={len(expert_names)}"
    )
    print(f"[Info] Processing {pcap_path}")

    headers = ["Window", "Prediction", "P(normal)", "P(dos)", "P(arp)"]
    headers.extend(f"gate_{name}" for name in expert_names)
    print("\t".join(headers))

    window_offset = 0
    window_records: List[Dict[str, Any]] = []
    prediction_ids: List[int] = []
    metas: List[Dict[str, Any]] = []
    prob_sums: Dict[str, float] = {cls: 0.0 for cls in CLASS_LABELS}
    high_conf_counts: Dict[str, int] = {cls: 0 for cls in CLASS_LABELS}
    meta_keys = ("meta", "meta_dict", "metadata", "window_meta")
    with torch.no_grad():
        for batch_features, _ in dataloader:
            logits, attention = model(batch_features, return_attention=True)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            weights = attention["weights"]

            batch_meta_source: Iterable[Any] | None = None
            for key in meta_keys:
                candidate = batch_features.get(key) if isinstance(batch_features, dict) else None
                if isinstance(candidate, list):
                    batch_meta_source = candidate
                    break

            for idx in range(predictions.size(0)):
                window_index = window_offset + idx + 1
                pred_label = int(predictions[idx].item())
                probs = probabilities[idx]
                gates = weights[idx]
                row = [
                    f"{window_index:06d}",
                    class_id_to_name(pred_label),
                    f"{probs[0].item():.4f}",
                    f"{probs[1].item():.4f}",
                    f"{probs[2].item():.4f}",
                ]
                row.extend(f"{value.item():.4f}" for value in gates)
                print("\t".join(row))
                prob_vector = [float(probs[i].item()) for i in range(NUM_CLASSES)]
                for cls_idx, cls_name in enumerate(CLASS_LABELS):
                    prob_value = prob_vector[cls_idx]
                    prob_sums[cls_name] += prob_value
                    if prob_value >= args.high_confidence_threshold:
                        high_conf_counts[cls_name] += 1
                window_records.append(
                    {
                        "row": row,
                        "prediction_id": pred_label,
                        "probabilities": prob_vector,
                    }
                )
                prediction_ids.append(pred_label)
                if batch_meta_source is not None:
                    try:
                        meta_candidate = batch_meta_source[idx]
                    except (IndexError, TypeError):
                        meta_candidate = None
                    if isinstance(meta_candidate, dict):
                        metas.append(meta_candidate)

            window_offset += predictions.size(0)

    total_windows = len(window_records)
    probability_means: Dict[str, float] = {
        cls: (prob_sums[cls] / total_windows) if total_windows > 0 else 0.0 for cls in CLASS_LABELS
    }

    window_counts, confidences, _ = _aggregate_predictions(prediction_ids)
    high_conf_counts = {cls: high_conf_counts.get(cls, 0) for cls in CLASS_LABELS}
    attack_threshold = max(0.0, min(1.0, float(args.attack_threshold)))
    attack_prob_threshold = max(0.0, min(1.0, float(args.attack_prob_threshold)))
    min_attack_windows = max(0, int(args.min_attack_windows))
    min_high_conf_windows = max(0, int(args.min_high_confidence_windows))
    verdict = _decide_verdict(
        window_counts,
        confidences,
        probability_means,
        high_conf_counts,
        attack_threshold,
        attack_prob_threshold,
        min_attack_windows,
        min_high_conf_windows,
    )
    print("\nPrediction counts:")
    for cls in CLASS_LABELS:
        print(f" - {cls:<7}: {window_counts.get(cls, 0)}")

    csv_path = _save_window_csv(pcap_path, headers, window_records)
    top_ips, top_macs = _extract_suspicious_entities(
        pcap_path,
        metas if metas else None,
        confidences,
    )
    summary_path = _save_summary_json(
        pcap_path,
        verdict,
        confidences,
        probability_means,
        high_conf_counts,
        window_counts,
        top_ips,
        top_macs,
        attack_threshold,
        attack_prob_threshold,
        float(args.high_confidence_threshold),
        min_attack_windows,
        min_high_conf_windows,
    )

    print(f"[Info] Saved per-window report to {csv_path}")
    print(f"[Info] Saved summary report to {summary_path}")

    verdict_fraction = float(confidences.get(verdict, 0.0))
    verdict_prob_mean = float(probability_means.get(verdict, 0.0))
    if verdict == "normal":
        verdict_line = f"{_verdict_message(verdict)} (confidence {verdict_fraction:.2f})"
    else:
        support_windows = int(window_counts.get(verdict, 0))
        verdict_line = (
            f"{_verdict_message(verdict)} "
            f"(fraction {verdict_fraction:.2f}, mean probability {verdict_prob_mean:.2f}, windows {support_windows})"
        )
    ips_display = ", ".join(top_ips) if top_ips else "None"
    macs_display = ", ".join(top_macs) if top_macs else "None"
    banner = f"=== Final Verdict for {pcap_path.name} ==="
    closing = "=" * len(banner)
    print(banner)
    print(verdict_line)
    print(f"Top suspicious IPs: {ips_display}")
    print(f"Top suspicious MACs: {macs_display}")
    print(closing)


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
