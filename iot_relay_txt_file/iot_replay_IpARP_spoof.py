from ryu.base import app_manager
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.controller import ofp_event
from ryu.lib.packet import packet, ethernet, arp, ipv4, udp
from ryu.lib.packet import ether_types
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
import csv
import time
import random

class IoTTrafficReplayer(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IoTTrafficReplayer, self).__init__(*args, **kwargs)

        self.mac_to_device = {
            "d0:52:a8:00:67:5e": "Smart Things",
            "44:65:0d:56:cc:d3": "Amazon Echo",
            "70:ee:50:18:34:43": "Netatmo Welcome",
            "f4:f2:6d:93:51:f1": "TP-Link Camera",
            "00:16:6c:ab:6b:88": "Samsung SmartCam",
            "30:8c:fb:2f:e4:b2": "Dropcam",
            "00:62:6e:51:27:2e": "Insteon Cam (Wired)",
            "e8:ab:fa:19:de:4f": "Insteon Cam (Wireless)",
            "00:24:e4:11:18:a8": "Withings Baby Monitor",
            "ec:1a:59:79:f4:89": "Belkin Wemo Switch",
            "50:c7:bf:00:56:39": "TP-Link Smart Plug",
            "74:c6:3b:29:d7:1d": "iHome",
            "ec:1a:59:83:28:11": "Belkin Motion Sensor",
            "18:b4:30:25:be:e4": "Nest Protect Alarm",
            "70:ee:50:03:b8:ac": "Netatmo Weather",
            "00:24:e4:1b:6f:96": "Withings Scale",
            "74:6a:89:00:2e:25": "Blipcare BP Meter",
            "00:24:e4:20:28:c6": "Withings Aura Sensor",
            "d0:73:d5:01:83:08": "LiFX Smart Bulb",
            "18:b7:9e:02:20:44": "Triby Speaker",
            "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",
            "70:5a:0f:e4:9b:c0": "HP Printer",
            "08:21:ef:3b:fc:e3": "Samsung Tab",
            "30:8c:fb:b6:ea:45": "Nest Dropcam",
            "40:f3:08:ff:1e:da": "Android Phone 1",
            "74:2f:68:81:69:42": "Laptop",
            "ac:bc:32:d4:6f:2f": "MacBook",
            "b4:ce:f6:a7:a3:c2": "Android Phone 2",
            "d0:a6:37:df:a1:e1": "iPhone",
            "f4:5c:89:93:cc:85": "MacBook/iPhone",
            "14:cc:20:51:33:ea": "TPLink Router (Gateway)",
            "00:00:00:00:00:14": "Malicious Host"
        }

        self.datapath = None
        self.packet_interval = 1
        self.attack_interval = 3
        self.csv_file = "iot_traffic.csv"
        self.logger.info("Starting IoT replay + ARP/IP spoofing simulation.")
        self.thread = hub.spawn(self._replay_traffic_loop)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        self.datapath = ev.msg.datapath
        ofproto = self.datapath.ofproto
        parser = self.datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(self.datapath, 0, match, actions)

    def _add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    def _replay_traffic_loop(self):
        time.sleep(5)
        packet_counter = 0

        while True:
            try:
                with open(self.csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        packet_counter += 1

                        src_mac = row.get("eth.src", "").strip().lower()
                        dst_mac = row.get("eth.dst", "").strip().lower()
                        src_ip = row.get("IP.src", "").strip()
                        dst_ip = row.get("IP.dst", "").strip()

                        src_device = self.mac_to_device.get(src_mac, f"Unknown({src_mac})")
                        dst_device = self.mac_to_device.get(dst_mac, f"Unknown({dst_mac})")

                        print(f"[Normal] {src_device} ({src_mac}) → {dst_device} ({dst_mac})")

                        # Every attack_interval packets, send spoof attacks:
                        if packet_counter % self.attack_interval == 0:
                            self._send_arp_spoof()
                            self._send_ip_spoof()

                        hub.sleep(self.packet_interval)

            except FileNotFoundError:
                self.logger.error(f"CSV file '{self.csv_file}' not found.")
                break

    def _send_arp_spoof(self):
        if not self.datapath:
            return

        parser = self.datapath.ofproto_parser
        ofproto = self.datapath.ofproto

        devices = list(self.mac_to_device.keys())
        malicious_mac = "00:00:00:00:00:14"
        candidates = [mac for mac in devices if mac != malicious_mac]
        if len(candidates) < 2:
            self.logger.error("Not enough devices for ARP spoofing.")
            return

        victim_mac = random.choice(candidates)
        spoofed_mac = random.choice([mac for mac in candidates if mac != victim_mac])

        def mac_to_ip(mac):
            last_byte = int(mac.split(":")[-1], 16)
            return f"192.168.1.{last_byte}"

        victim_ip = mac_to_ip(victim_mac)
        spoofed_ip = mac_to_ip(spoofed_mac)

        p = packet.Packet()
        p.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            src=malicious_mac,
            dst="ff:ff:ff:ff:ff:ff"
        ))
        p.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=malicious_mac,
            src_ip=spoofed_ip,
            dst_mac="ff:ff:ff:ff:ff:ff",
            dst_ip=victim_ip
        ))
        p.serialize()

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=self.datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=p.data
        )
        self.datapath.send_msg(out)

        print(f"\033[91m[!! ARP Spoof] Malicious Host ({malicious_mac}) claims to be {spoofed_ip} ({spoofed_mac}) → Victim IP {victim_ip} ({victim_mac})\033[0m")

    def _send_ip_spoof(self):
        if not self.datapath:
            return

        parser = self.datapath.ofproto_parser
        ofproto = self.datapath.ofproto

        devices = list(self.mac_to_device.keys())
        malicious_mac = "00:00:00:00:00:14"
        candidates = [mac for mac in devices if mac != malicious_mac]
        if len(candidates) < 2:
            self.logger.error("Not enough devices for IP spoofing.")
            return

        victim_mac = random.choice(candidates)
        spoofed_mac = random.choice([mac for mac in candidates if mac != victim_mac])

        def mac_to_ip(mac):
            last_byte = int(mac.split(":")[-1], 16)
            return f"192.168.1.{last_byte}"

        victim_ip = mac_to_ip(victim_mac)
        spoofed_ip = mac_to_ip(spoofed_mac)

        # Construct Ethernet + IPv4 + UDP packet with spoofed source IP
        p = packet.Packet()
        p.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_IP,
            src=malicious_mac,
            dst="ff:ff:ff:ff:ff:ff"
        ))
        p.add_protocol(ipv4.ipv4(
            total_length=20 + 8,
            proto=17,  # UDP
            src=spoofed_ip,
            dst=victim_ip
        ))
        # Minimal UDP payload for completeness
        p.add_protocol(udp.udp(
            src_port=12345,
            dst_port=80,
            total_length=8
        ))
        p.serialize()

        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=self.datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=p.data
        )
        self.datapath.send_msg(out)

        print(f"\033[93m[!! IP Spoof] Malicious Host ({malicious_mac}) sends packet with spoofed IP {spoofed_ip} → Victim IP {victim_ip} ({victim_mac})\033[0m")