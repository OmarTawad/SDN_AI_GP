import numpy as np
from dosdet.features.static_features import compute_static_features

def _mk(ts, proto="udp", dst_port=None, tcp_syn=0, tcp_synack=0, ssdp_method="NONE", length=60, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_mac="aa:bb:cc:dd:ee:ff", ttl=64):
    return {
        "ts": ts, "len": length,
        "is_udp": 1 if proto=="udp" else 0,
        "is_tcp": 1 if proto=="tcp" else 0,
        "is_icmp": 1 if proto=="icmp" else 0,
        "dst_port": dst_port,
        "tcp_syn": tcp_syn, "tcp_synack": tcp_synack,
        "ssdp_method": ssdp_method,
        "src_ip": src_ip, "dst_ip": dst_ip,
        "src_mac": src_mac, "ttl": ttl
    }

def test_static_core():
    rows = [
        _mk(0.00,"udp",1900, ssdp_method="M-SEARCH"),
        _mk(0.10,"udp",1900, ssdp_method="M-SEARCH"),
        _mk(0.20,"udp",1900, ssdp_method="200-OK"),
        _mk(0.30,"tcp",80, tcp_syn=1),
        _mk(0.40,"icmp"),
    ]
    per_bin = np.array([2,1,0,0,0,0,0,0,0,0,0,0], dtype=float)
    vec, names, snaps = compute_static_features(rows, micro_bins=12, per_bin_total_pkts=per_bin, top_k_udp_ports=[1900,53], window_sec=1.0)
    assert len(vec)==len(names)
    assert "udp_1900_fraction" in names
    assert snaps["pkts"] == 5.0
    idx = names.index("tcp_syn_over_synack")
    assert vec[idx] >= 1.0  # at least one SYN, zero SYN-ACK → ratio >= 1
