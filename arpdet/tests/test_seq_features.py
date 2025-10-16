import numpy as np
from features.seq_features import compute_sequence_features


def _arp_row(ts, opcode, sender_ip, sender_mac, target_ip=None, target_mac=None, length=60, gratuitous=0):
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


def test_sequence_features_capture_conflicts():
    rows = [
        _arp_row(0.01, 1, "10.0.0.1", "aa:aa:aa:aa:aa:01", "10.0.0.254", "ff:ff:ff:ff:ff:ff", gratuitous=1),
        _arp_row(0.12, 2, "10.0.0.1", "aa:aa:aa:aa:aa:01", "10.0.0.100", "00:11:22:33:44:55"),
        _arp_row(0.18, 2, "10.0.0.2", "aa:aa:aa:aa:aa:01", "10.0.0.101", "00:11:22:33:44:55"),
        {"ts": 0.25, "len": 80, "is_arp": 0},  # non-ARP packet should be ignored
    ]
    bins = [0, 0, 0, 1]
    X, extras = compute_sequence_features(rows, bins, micro_bins=8)
    names = extras["feature_names"]
    idx_conflicting = list(names).index("bin_conflicting_sender_claims")
    idx_replies = list(names).index("bin_arp_replies")
    idx_unique_ips = list(names).index("bin_unique_sender_ips")

    # All ARP replies fall in bin 0
    assert X[0, idx_replies] == 2.0
    # Same MAC claimed two IP addresses in the window
    assert X[0, idx_conflicting] == 1.0
    # Unique sender IPs tracked correctly
    assert X[0, idx_unique_ips] == 2.0

    # Non-ARP packet does not contribute to ARP counts
    assert np.allclose(X[1:, idx_replies], 0.0)
