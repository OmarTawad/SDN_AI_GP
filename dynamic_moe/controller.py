"""Ryu application tying OpenFlow events to the unified MoE."""

from __future__ import annotations

import concurrent.futures
import copy
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import arp, ethernet, ether_types, icmp, ipv4, ipv6, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3
from scapy.layers.l2 import Ether

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.dynamic_moe_adapter import DynamicMoEGateway
from dynamic_moe.config import load_runtime_config
from dynamic_moe.device_map import DEVICE_MAP, normalize_mac
from dynamic_moe.feature_extractor import FeatureWindow, StreamingFeatureExtractor
from dynamic_moe.policy import PolicyConfig, SdnPolicyEngine
from dynamic_moe.runtime import RuntimeLogger


class DynamicMoeController(app_manager.RyuApp):
    """Learning-switch style controller extended with MoE inspection."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger.setLevel(logging.INFO)
        self.mac_to_port: Dict[int, Dict[str, int]] = {}
        self.moe = DynamicMoEGateway(config_path="gateway/config_dynamic.yaml")
        self.runtime_config = load_runtime_config()
        self.runtime_logger = RuntimeLogger(self.runtime_config)
        self.policy = SdnPolicyEngine(
            PolicyConfig(
                min_alert_confidence=self.runtime_config.min_alert_confidence,
                min_rate_limit_confidence=self.runtime_config.min_rate_limit_confidence,
                min_block_confidence=self.runtime_config.min_block_confidence,
                arp_isolate_confidence=self.runtime_config.arp_isolate_confidence,
                mitigation_mode=self.runtime_config.mitigation,
                allow_automatic_blocking=self.runtime_config.allow_automatic_blocking,
                action_expiry_seconds=self.runtime_config.action_expiry_seconds,
            )
        )
        self.extractor = StreamingFeatureExtractor()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.blocked_macs: set[str] = set()

    # ------------------------------------------------------------------ Flow management helpers
    def _add_flow(self, datapath, priority, match, actions, buffer_id=None) -> None:
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id is not None:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                buffer_id=buffer_id,
                match=match,
                instructions=inst,
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst,
            )
        datapath.send_msg(mod)

    def _apply_mitigation(self, datapath, src_mac: Optional[str], action: str) -> None:
        if not src_mac or src_mac in self.blocked_macs or action not in {"drop", "isolate", "quarantine"}:
            return
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_src=src_mac)
        self._add_flow(datapath, priority=20, match=match, actions=[])
        self.blocked_macs.add(src_mac)
        self.logger.warning("Installed %s flow for %s", action, src_mac)
        self.runtime_logger.log_flow_event(datapath.id, 0, 0, src_mac, "*", action, notes="MoE mitigation")

    # ------------------------------------------------------------------ Ryu event hooks
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.mac_to_port.setdefault(datapath.id, {})
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)
        self.logger.info("Configured table-miss on switch %s", datapath.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = normalize_mac(eth.dst)
        src = normalize_mac(eth.src)
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, 1, match, actions, msg.buffer_id)
            else:
                self._add_flow(datapath, 1, match, actions)
            self.runtime_logger.log_flow_event(dpid, in_port, out_port, src, dst, "forward")

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

        packet_meta = self._build_packet_metadata(pkt, dpid, in_port, out_port)
        packet_meta["raw_frame"] = bytes(msg.data)
        try:
            scapy_pkt = Ether(msg.data)
        except Exception:
            self.logger.warning("Failed to decode Ethernet frame for feature extraction.")
            return
        scapy_pkt.time = time.time()
        windows = self.extractor.process_packet(scapy_pkt)
        for window in windows:
            self.executor.submit(self._classify_window, window, copy.deepcopy(packet_meta), datapath)

    # ------------------------------------------------------------------ Helpers
    def _build_packet_metadata(self, pkt: packet.Packet, switch_id: int, in_port: int, out_port: int) -> Dict[str, object]:
        ip_pkt = pkt.get_protocol(ipv4.ipv4) or pkt.get_protocol(ipv6.ipv6)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        arp_pkt = pkt.get_protocol(arp.arp)
        protocol = "ether"
        src_ip = dst_ip = None
        src_port = dst_port = None
        if ip_pkt is not None:
            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            protocol = "ipv4" if isinstance(ip_pkt, ipv4.ipv4) else "ipv6"
        elif arp_pkt is not None:
            src_ip = arp_pkt.src_ip
            dst_ip = arp_pkt.dst_ip
            protocol = "arp"
        if tcp_pkt is not None:
            src_port = tcp_pkt.src_port
            dst_port = tcp_pkt.dst_port
            protocol = "tcp"
        elif udp_pkt is not None:
            src_port = udp_pkt.src_port
            dst_port = udp_pkt.dst_port
            protocol = "udp"
        elif pkt.get_protocol(icmp.icmp) is not None:
            protocol = "icmp"

        src_mac = normalize_mac(pkt.get_protocols(ethernet.ethernet)[0].src)
        dst_mac = normalize_mac(pkt.get_protocols(ethernet.ethernet)[0].dst)
        meta = {
            "switch": switch_id,
            "in_port": in_port,
            "out_port": out_port if out_port != ofproto_v1_3.OFPP_FLOOD else None,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "src_device": DEVICE_MAP.get(src_mac, {}).get("name"),
            "dst_device": DEVICE_MAP.get(dst_mac, {}).get("name"),
        }
        return meta

    def _classify_window(self, window: FeatureWindow, packet_meta: Dict[str, object], datapath) -> None:
        try:
            result = self.moe.predict(window.features)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("MoE inference failed: %s", exc)
            return
        context = window.context
        context.setdefault("packet_count", 0)
        self.runtime_logger.log_decision(context, packet_meta, result)
        self.runtime_logger.log_packet_metadata(context, packet_meta, result)
        decision = self.policy.decide(result)
        self.runtime_logger.log_mitigation(
            packet_meta=packet_meta,
            inference=result,
            action=decision.action,
            expiry=decision.expiry,
            reason=decision.reason,
        )
        if result.get("is_attack"):
            self.logger.warning(
                "Attack detected | type=%s score=%.3f action=%s src=%s dst=%s",
                result.get("attack_type"),
                float(result.get("score", 0.0)),
                decision.action,
                packet_meta.get("src_mac"),
                packet_meta.get("dst_mac"),
            )
            self.runtime_logger.log_attack(context, packet_meta, result, raw_frame=packet_meta.get("raw_frame"))
            if decision.install_flow:
                self._apply_mitigation(datapath, packet_meta.get("src_mac"), decision.action)

    def close(self) -> None:  # pragma: no cover - not triggered in tests
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.runtime_logger.close()
        except Exception:
            pass
        super().close()


__all__ = ["DynamicMoeController"]
