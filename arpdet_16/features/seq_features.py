from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def compute_sequence_features(
    win_rows: List[Dict],
    bin_idx: List[int],
    micro_bins: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      seq_feats: shape [M, K_seq]
      extras: dict with per-bin aggregates useful for downstream logic.
    Per-bin features:
      - total packets and bytes
      - ARP packet/request/reply counts
      - unique ARP sender/target IPs and MACs
      - conflicting claims per sender MAC / target IP
      - gratuitous ARP count
    """
    M = int(micro_bins)
    names = [
        "bin_total_pkts",
        "bin_total_bytes",
        "bin_arp_pkts",
        "bin_arp_requests",
        "bin_arp_replies",
        "bin_unique_sender_ips",
        "bin_unique_sender_macs",
        "bin_unique_target_ips",
        "bin_unique_target_macs",
        "bin_conflicting_sender_claims",
        "bin_conflicting_target_claims",
        "bin_gratuitous_arp",
    ]
    K = len(names)
    X = np.zeros((M, K), dtype=np.float32)

    sender_ip_sets = [set() for _ in range(M)]
    sender_mac_sets = [set() for _ in range(M)]
    target_ip_sets = [set() for _ in range(M)]
    target_mac_sets = [set() for _ in range(M)]
    mac_to_ips = [dict() for _ in range(M)]  # type: ignore[var-annotated]
    ip_to_macs = [dict() for _ in range(M)]  # type: ignore[var-annotated]

    col = {n: i for i, n in enumerate(names)}

    for row, b in zip(win_rows, bin_idx):
        if b < 0 or b >= M:
            continue
        X[b, col["bin_total_pkts"]] += 1.0
        X[b, col["bin_total_bytes"]] += float(row.get("len") or 0.0)

        if int(row.get("is_arp", 0)) != 1:
            continue

        X[b, col["bin_arp_pkts"]] += 1.0
        opcode = int(row.get("arp_opcode") or 0)
        if opcode == 1:
            X[b, col["bin_arp_requests"]] += 1.0
        elif opcode == 2:
            X[b, col["bin_arp_replies"]] += 1.0

        if int(row.get("arp_is_gratuitous", 0)) == 1:
            X[b, col["bin_gratuitous_arp"]] += 1.0

        sender_ip = row.get("arp_sender_ip")
        sender_mac = row.get("arp_sender_mac")
        target_ip = row.get("arp_target_ip")
        target_mac = row.get("arp_target_mac")

        if sender_ip:
            sender_ip_sets[b].add(sender_ip)
        if sender_mac:
            sender_mac_sets[b].add(sender_mac)
        if target_ip:
            target_ip_sets[b].add(target_ip)
        if target_mac:
            target_mac_sets[b].add(target_mac)

        if sender_mac and sender_ip:
            claims = mac_to_ips[b].setdefault(sender_mac, set())
            claims.add(sender_ip)
        if target_ip and sender_mac:
            reverse_claims = ip_to_macs[b].setdefault(target_ip, set())
            reverse_claims.add(sender_mac)

    for idx in range(M):
        X[idx, col["bin_unique_sender_ips"]] = float(len(sender_ip_sets[idx]))
        X[idx, col["bin_unique_sender_macs"]] = float(len(sender_mac_sets[idx]))
        X[idx, col["bin_unique_target_ips"]] = float(len(target_ip_sets[idx]))
        X[idx, col["bin_unique_target_macs"]] = float(len(target_mac_sets[idx]))

        conflict_sender = sum(1 for claims in mac_to_ips[idx].values() if len(claims) > 1)
        conflict_target = sum(1 for claims in ip_to_macs[idx].values() if len(claims) > 1)
        X[idx, col["bin_conflicting_sender_claims"]] = float(conflict_sender)
        X[idx, col["bin_conflicting_target_claims"]] = float(conflict_target)

    extras = {
        "per_bin_total_pkts": X[:, col["bin_total_pkts"]].copy(),
        "per_bin_arp_pkts": X[:, col["bin_arp_pkts"]].copy(),
        "per_bin_conflicting_sender_claims": X[:, col["bin_conflicting_sender_claims"]].copy(),
        "per_bin_conflicting_target_claims": X[:, col["bin_conflicting_target_claims"]].copy(),
        "feature_names": np.array(names),
    }
    return X, extras
