import numpy as np
from features.static_features import compute_static_features


def _arp(ts, opcode, sender_ip, sender_mac, target_ip=None, target_mac=None, length=60, gratuitous=0):
    return {
        "ts": ts,
        "len": length,
        "is_arp": 1,
        "arp_opcode": opcode,
        "arp_sender_ip": sender_ip,
        "arp_sender_mac": sender_mac,
        "arp_target_ip": target_ip,
        "arp_target_mac": target_mac,
        "arp_is_gratuitous": gratuitous,
    }


def test_static_features_reflect_arp_conflicts():
    rows = [
        _arp(0.00, 1, "10.0.0.10", "aa:aa:aa:aa:aa:01", "10.0.0.254", "ff:ff:ff:ff:ff:ff", gratuitous=1),
        _arp(0.12, 2, "10.0.0.10", "aa:aa:aa:aa:aa:01", "10.0.0.1", "00:11:22:33:44:55"),
        _arp(0.20, 2, "10.0.0.20", "aa:aa:aa:aa:aa:01", "10.0.0.2", "00:11:22:33:44:55"),
    ]
    extras = {
        "per_bin_total_pkts": np.array([3] + [0] * 7, dtype=float),
        "per_bin_conflicting_sender_claims": np.array([1] + [0] * 7, dtype=float),
        "per_bin_conflicting_target_claims": np.zeros(8, dtype=float),
    }
    vec, names, snaps = compute_static_features(rows, micro_bins=8, extras=extras, window_sec=1.0)

    assert "sender_conflict_ratio" in names
    idx_conflict = names.index("sender_conflict_ratio")
    assert vec[idx_conflict] > 0.0
    assert snaps["arp_replies"] == 2.0
    assert snaps["sender_conflict_ratio"] == vec[idx_conflict]
