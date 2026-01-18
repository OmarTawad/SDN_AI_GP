#tests/test_conflict_detection.py

from arp_detector.inference.pipeline import detect_conflicting_replies


def test_detect_conflicting_replies_flags_multi_ip_mac():
    host_activity = {
        "macs": [
            {"mac": "00:11:22:33:44:55", "reply_conflict_ip_count": 1},
            {
                "mac": "aa:bb:cc:dd:ee:ff",
                "reply_conflict_ip_count": 2,
                "reply_claims": [
                    {"ip": "192.168.0.10", "count": 3},
                    {"ip": "192.168.0.11", "count": 2},
                ],
            },
        ]
    }
    entry = detect_conflicting_replies(host_activity)
    assert entry is not None
    assert entry["mac"] == "aa:bb:cc:dd:ee:ff"


def test_detect_conflicting_replies_returns_none_without_conflict():
    host_activity = {
        "macs": [
            {"mac": "00:00:00:00:00:01", "reply_conflict_ip_count": 1},
            {"mac": "00:00:00:00:00:02", "reply_conflict_ip_count": 0},
        ]
    }
    assert detect_conflicting_replies(host_activity) is None
