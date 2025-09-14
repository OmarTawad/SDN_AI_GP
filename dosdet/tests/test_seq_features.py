import numpy as np
from dosdet.features.seq_features import compute_sequence_features

def _mk(ts, proto="udp", dst_port=None, tcp_syn=0, tcp_synack=0, ssdp_method="NONE", length=60):
    r = {
        "ts": ts, "len": length,
        "is_udp": 1 if proto=="udp" else 0,
        "is_tcp": 1 if proto=="tcp" else 0,
        "is_icmp": 1 if proto=="icmp" else 0,
        "dst_port": dst_port,
        "tcp_syn": tcp_syn, "tcp_synack": tcp_synack,
        "ssdp_method": ssdp_method
    }
    return r

def test_seq_counts():
    rows = [
        _mk(0.01,"udp",1900, ssdp_method="M-SEARCH"),
        _mk(0.02,"udp",1900, ssdp_method="200-OK"),
        _mk(0.03,"tcp", dst_port=80, tcp_syn=1),
        _mk(0.40,"icmp"),
    ]
    bins = [0,0,1,8]
    X, extras = compute_sequence_features(rows, bins, micro_bins=12, top_k_udp_ports=[1900,53])
    names = extras["feature_names"].tolist()
    # sanity checks
    assert X.sum() > 0
    assert "bin_udp_pkts_dst_1900" in names
    i_msearch = names.index("bin_ssdp_msearch")
    i_ssdpok = names.index("bin_ssdp_response")
    assert X[:, i_msearch].sum() == 1
    assert X[:, i_ssdpok].sum() == 1
