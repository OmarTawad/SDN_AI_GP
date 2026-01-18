from decision import DecisionConfig, WindowObs, decide_file


def make_window(prob: float, snaps: dict, index: int) -> WindowObs:
    defaults = {
        "arp_replies": 0.0,
        "reply_conflict_macs": 0.0,
    }
    defaults.update(snaps)
    return WindowObs(prob=prob, snaps=defaults, t0=float(index), t1=float(index + 1))


def base_config() -> DecisionConfig:
    return DecisionConfig(
        tau_high=0.7,
        tau_low=0.5,
        min_attack_windows=2,
        consecutive_required=1,
        cooldown_windows=1,
        enable_gate=True,
        min_arp_replies=2.0,
        min_conflict_macs=1,
    )


def test_normal_file_remains_normal():
    cfg = base_config()
    windows = [make_window(0.3, {"arp_replies": 0.0}, i) for i in range(10)]
    result = decide_file("normal.pcap", windows, cfg)
    assert result.decision == "normal"
    assert result.first_attack_timestamp is None
    assert result.num_attack_windows == 0


def test_conflicting_mac_triggers_attack():
    cfg = base_config()
    windows = [make_window(0.3, {"arp_replies": 0.0}, i) for i in range(5)]
    windows += [
        make_window(
            0.85,
            {
                "arp_replies": 3.0,
                "reply_conflict_macs": 1.0,
            },
            5,
        ),
        make_window(
            0.82,
            {
                "arp_replies": 2.0,
                "reply_conflict_macs": 1.0,
            },
            6,
        ),
    ]
    result = decide_file("spoof.pcap", windows, cfg)
    assert result.decision == "attack"
    assert result.first_attack_timestamp == 5.0
    assert result.num_attack_windows >= cfg.min_attack_windows


def test_high_prob_without_conflict_is_blocked_by_gate():
    cfg = base_config()
    windows = [
        make_window(
            0.9,
            {
                "arp_replies": 5.0,
                "reply_conflict_macs": 0.0,
            },
            0,
        ),
        make_window(
            0.88,
            {
                "arp_replies": 4.0,
                "reply_conflict_macs": 0.0,
            },
            1,
        ),
    ]
    result = decide_file("benign_high_prob.pcap", windows, cfg)
    assert result.decision == "normal"


def test_cooldown_resets_state():
    cfg = base_config()
    windows = [
        make_window(
            0.8,
            {
                "arp_replies": 2.0,
                "reply_conflict_macs": 1.0,
            },
            0,
        ),
        make_window(0.2, {"arp_replies": 0.0}, 1),
        make_window(
            0.82,
            {
                "arp_replies": 3.0,
                "reply_conflict_macs": 1.0,
            },
            2,
        ),
    ]
    result = decide_file("cooldown.pcap", windows, cfg)
    # Only one attack window cluster qualifies once because cooldown resets
    assert result.decision == "attack"
    assert result.num_attack_windows >= 1
