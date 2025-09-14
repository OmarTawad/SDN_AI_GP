# tests/test_decision.py
from decision import DecisionConfig, WindowObs, decide_file

def W(p, s, t):
    # helper to construct WindowObs with minimal snaps; 't' index is seconds
    snaps = {
        "pkts_per_s": s.get("pkts_per_s", 100.0),
        "max_bin_pkts": s.get("max_bin_pkts", 10.0),
        "median_bin_pkts": s.get("median_bin_pkts", 5.0),
        "udp_ssdp_req_over_resp": s.get("udp_ssdp_req_over_resp", 1.0),
        "udp_1900_fraction": s.get("udp_1900_fraction", 0.0),
        "ssdp_multicast_hit": s.get("ssdp_multicast_hit", 0.0),
        "ssdp_header_repetition": s.get("ssdp_header_repetition", 0.0),
        "H_src_ip": s.get("H_src_ip", 6.0),
        "H_dst_port": s.get("H_dst_port", 6.0),
        "H_ttl": s.get("H_ttl", 6.0),
        "tcp_syn_over_synack": s.get("tcp_syn_over_synack", 1.0),
        "tcp_syn_rate": s.get("tcp_syn_rate", 10.0),
        "tcp_synack_completion": s.get("tcp_synack_completion", 0.8),
    }
    return WindowObs(prob=p, snaps=snaps, t0=float(t), t1=float(t+1))

def cfg_defaults():
    return DecisionConfig(
        tau_high=0.70,
        tau_low=0.55,
        min_attack_windows=3,
        consecutive_required=2,
        cooldown_windows=1,
        enable_gate=True,
        warmup_windows=10,   # keep tests small
        abs_pkts_per_s_cap=1200.0,
        burstiness_multiple=3.0,
        syn_completion_max=0.10
    )

def test_no_attack_all_normal():
    cfg = cfg_defaults()
    wins = [W(0.2, {"pkts_per_s": 100}, t) for t in range(30)]
    out = decide_file("normal.pcap", wins, cfg)
    assert out.decision == "normal"
    assert out.first_attack_timestamp is None
    assert out.num_attack_windows == 0

def test_clear_attack_sustained_high_ssdp():
    cfg = cfg_defaults()
    # Warmup 10 normalish windows
    wins = [W(0.3, {"pkts_per_s": 100}, t) for t in range(10)]
    # Attack cluster: high prob and strong SSDP gate (dominance + asymmetry + rate spike)
    for t in range(10, 16):
        wins.append(W(
            0.92,
            {
                "pkts_per_s": 2200,               # rate spike
                "udp_1900_fraction": 0.9,         # SSDP dominance
                "udp_ssdp_req_over_resp": 25.0,   # asymmetry
                "ssdp_multicast_hit": 1.0,
                "H_dst_port": 2.0,                # entropy collapse
                "max_bin_pkts": 90.0,
                "median_bin_pkts": 20.0
            },
            t
        ))
    # Then some trailing lows
    for t in range(16, 25):
        wins.append(W(0.2, {"pkts_per_s": 120}, t))
    out = decide_file("attack_ssdp.pcap", wins, cfg)
    assert out.decision == "attack"
    assert out.first_attack_timestamp is not None
    assert out.num_attack_windows >= cfg.min_attack_windows
    assert any("SSDP" in r or "rate spike" in r for r in out.gate_reasons)

def test_mixed_bursty_with_cooldown():
    cfg = cfg_defaults()
    wins = [W(0.3, {"pkts_per_s": 150}, t) for t in range(10)]
    # small burst that should NOT reach cluster size
    wins += [
        W(0.75, {"pkts_per_s": 1300, "max_bin_pkts": 100, "median_bin_pkts": 20, "H_dst_port": 2.0}, 10),
        W(0.40, {"pkts_per_s": 200}, 11),  # cooldown path triggers but not attack enter
        W(0.78, {"pkts_per_s": 1350, "max_bin_pkts": 90, "median_bin_pkts": 30, "H_dst_port": 2.2}, 12),
    ]
    wins += [W(0.3, {"pkts_per_s": 150}, t) for t in range(13, 20)]
    out = decide_file("mixed.pcap", wins, cfg)
    assert out.decision == "normal"  # not enough consecutive + cluster size

def test_benign_ssdp_discovery_not_attack():
    cfg = cfg_defaults()
    wins = [W(0.35, {"pkts_per_s": 180}, t) for t in range(10)]
    # Benign discovery: short, balanced, moderate rate
    for t in range(10, 14):
        wins.append(W(
            0.65,  # even if model is a bit jumpy
            {
                "pkts_per_s": 400,                 # not a big spike
                "udp_1900_fraction": 0.35,         # not dominant
                "udp_ssdp_req_over_resp": 1.5,     # balanced
                "ssdp_multicast_hit": 1.0,         # may hit multicast
                "H_dst_port": 5.0,
                "max_bin_pkts": 30.0,
                "median_bin_pkts": 15.0,
            },
            t
        ))
    wins += [W(0.3, {"pkts_per_s": 150}, t) for t in range(14, 20)]
    out = decide_file("benign_ssdp.pcap", wins, cfg)
    assert out.decision == "normal"
    assert out.first_attack_timestamp is None