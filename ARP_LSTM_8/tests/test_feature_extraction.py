#tests/test_feature_extraction.py

from arp_detector.features.feature_engineering import FeatureExtractor
from arp_detector.config.types import FeatureConfig
from arp_detector.data.structures import PacketRecord, Window


def _arp_packet(
    timestamp: float,
    opcode: int,
    sender_ip: str,
    sender_mac: str,
    target_ip: str,
    target_mac: str,
    gratuitous: bool = False,
) -> PacketRecord:
    return PacketRecord(
        timestamp=timestamp,
        src_mac=sender_mac,
        dst_mac=target_mac,
        src_ip=sender_ip,
        dst_ip=target_ip,
        src_port=None,
        dst_port=None,
        protocol="arp",
        length=42,
        ttl=None,
        tcp_flags=None,
        payload_len=28,
        info={},
        arp_opcode=opcode,
        arp_sender_ip=sender_ip,
        arp_sender_mac=sender_mac,
        arp_target_ip=target_ip,
        arp_target_mac=target_mac,
        arp_is_gratuitous=gratuitous,
    )


def test_feature_extractor_flags_conflicting_claims():
    window = Window(
        index=0,
        start_time=0.0,
        end_time=1.0,
        packets=[
            _arp_packet(0.1, 1, "192.168.0.10", "aa:aa:aa:aa:aa:aa", "192.168.0.1", "ff:ff:ff:ff:ff:ff"),
            _arp_packet(0.2, 2, "192.168.0.1", "11:11:11:11:11:11", "192.168.0.10", "aa:aa:aa:aa:aa:aa"),
            _arp_packet(0.3, 2, "192.168.0.1", "22:22:22:22:22:22", "192.168.0.11", "bb:bb:bb:bb:bb:bb"),
            _arp_packet(0.6, 2, "192.168.0.1", "22:22:22:22:22:22", "192.168.0.12", "cc:cc:cc:cc:cc:cc"),
            _arp_packet(0.8, 2, "192.168.0.1", "22:22:22:22:22:22", "192.168.0.13", "dd:dd:dd:dd:dd:dd", gratuitous=True),
            _arp_packet(0.9, 2, "192.168.0.2", "22:22:22:22:22:22", "192.168.0.14", "ee:ee:ee:ee:ee:ee"),
        ],
    )
    extractor = FeatureExtractor(FeatureConfig(), window_size=1.0)
    frame = extractor.extract([window])
    row = frame.iloc[0]

    assert row["arp_fraction"] == 1.0
    assert row["arp_reply_rate"] > 0.0
    assert row["max_claims_per_ip"] == 2
    assert row["conflict_ip_ratio"] > 0.0
    assert row["max_ips_per_mac"] >= 2  # attacker MAC claims multiple IPs
    assert row["sender_conflict_ratio"] > 0.0
    assert row["arp_gratuitous_fraction"] > 0.0
