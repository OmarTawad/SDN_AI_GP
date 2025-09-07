# iot_replay_packet_out.py
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub

from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP
import logging

# Terminal color codes
RED = "\033[91m"
ENDC = "\033[0m"
YELLOW = "\033[93m"

class IoTTrafficReplayer(app_manager.RyuApp):
    # Use OpenFlow 1.3
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IoTTrafficReplayer, self).__init__(*args, **kwargs)
        self.logger.setLevel(logging.INFO)

        # PCAP file path
        self.pcap_file = '/home/omar/18-10-27.pcap'

        # Mapping of MAC addresses to friendly names (same as your map)
        self.device_map = self._load_device_map()

        # Attack MACs (for log highlighting)
        self.attack_macs = {
            "00:17:88:2b:9a:25": "Huebulb Attack",
            "f4:f5:d8:8f:0a:3c": "Chromecast Flood",
        }

        # Datapath (switch) reference once connected
        self.datapath = None
        self.ofproto = None
        self.parser = None

        # Optional: MAC -> port learnt mapping (if you implement learning)
        # Example: {'aa:bb:cc:...': 1}
        self.mac_to_port = {}

        # Start the traffic replay thread
        self.thread = hub.spawn(self.replay_traffic)

    def _load_device_map(self):
        return {
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
            "40:f3:08:ff:1e:da": "Android Phone",
            "74:2f:68:81:69:42": "Laptop",
            "ac:bc:32:d4:6f:2f": "MacBook",
            "b4:ce:f6:a7:a3:c2": "Android Phone",
            "d0:a6:37:df:a1:e1": "iPhone",
            "f4:5c:89:93:cc:85": "MacBook/iPhone",
            "14:cc:20:51:33:ea": "TPLink Router (Gateway)",
            "00:17:88:2b:9a:25": "Huebulb",
            "f4:f5:d8:8f:0a:3c": "Chromecast"
        }

    # This handler runs when a switch connects and negotiates features.
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapath = dp
        self.ofproto = dp.ofproto
        self.parser = dp.ofproto_parser
        self.logger.info("Switch connected. Datapath stored for packet_out injection (OFP 1.3).")

        # OPTIONAL: install a simple table-miss flow so the switch sends unknown packets to controller
        match = self.parser.OFPMatch()
        actions = [self.parser.OFPActionOutput(self.ofproto.OFPP_CONTROLLER, self.ofproto.OFPCML_NO_BUFFER)]
        inst = [self.parser.OFPInstructionActions(self.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = self.parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)
        self.logger.info("Installed table-miss flow to send unknown packets to controller (optional).")

    def _choose_out_port(self, src_mac, dst_mac):
        """
        Choose an output port for a given dst mac.
        If we have a learning table entry, use it; otherwise flood.
        """
        dst = dst_mac.lower()
        if dst in self.mac_to_port:
            return self.mac_to_port[dst]
        return self.ofproto.OFPP_FLOOD

    def replay_traffic(self):
        # Wait until datapath is connected
        wait_seconds = 0
        while self.datapath is None and wait_seconds < 30:
            self.logger.info("Waiting for switch datapath to connect...")
            hub.sleep(1)
            wait_seconds += 1

        if self.datapath is None:
            self.logger.error("No datapath connected after waiting — aborting live injection.")
            return

        self.logger.info(f"Streaming IoT traffic from PCAP and injecting via PACKET_OUT: {self.pcap_file}")
        try:
            with PcapReader(self.pcap_file) as pcap_reader:
                for pkt in pcap_reader:
                    hub.sleep(0.001)  # slight delay

                    # we only care about Ethernet/IP packets for now
                    if not pkt.haslayer(Ether) or not pkt.haslayer(IP):
                        continue

                    # lower-case macs
                    src_mac = pkt[Ether].src.lower()
                    dst_mac = pkt[Ether].dst.lower()
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst

                    # protocol detection
                    if pkt.haslayer(TCP):
                        proto = "TCP"
                    elif pkt.haslayer(UDP):
                        proto = "UDP"
                    elif pkt.haslayer(ICMP):
                        proto = "ICMP"
                    else:
                        proto = f"Proto{pkt[IP].proto}"

                    src_device = self.device_map.get(src_mac, f"Unknown({src_mac})")
                    dst_device = self.device_map.get(dst_mac, f"Unknown({dst_mac})")

                    # Logging with highlights
                    if src_mac in self.attack_macs or dst_mac in self.attack_macs:
                        attack_name = self.attack_macs.get(src_mac, self.attack_macs.get(dst_mac, "Unknown Attack"))
                        log_msg = f"{RED}[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip}) [ATTACK: {attack_name}]{ENDC}"
                    elif "Unknown" in src_device or "Unknown" in dst_device:
                        log_msg = f"{YELLOW}[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip}){ENDC}"
                    else:
                        log_msg = f"[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip})"

                    self.logger.info(log_msg)

                    # Build packet-out: use the raw bytes from Scapy
                    raw = bytes(pkt)

                    # choose port (flood if unknown)
                    out_port = self._choose_out_port(src_mac, dst_mac)
                    actions = [self.parser.OFPActionOutput(out_port)]

                    # Create and send the packet_out message
                    out = self.parser.OFPPacketOut(
                        datapath=self.datapath,
                        buffer_id=self.ofproto.OFP_NO_BUFFER,
                        in_port=self.ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=raw
                    )
                    self.datapath.send_msg(out)

        except FileNotFoundError:
            self.logger.error(f"PCAP file '{self.pcap_file}' not found.")
        except Exception as e:
            self.logger.error(f"Error reading PCAP: {e}")

    # OPTIONAL: If you want the controller to learn MAC->port from PacketIn events,
    # implement a PacketIn handler that populates self.mac_to_port. That will allow
    # _choose_out_port to send directly to target ports instead of flooding.
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        in_port = msg.match['in_port']
        pkt = None
        try:
            # msg.data raw bytes -> parse with scapy if needed
            from scapy.all import Ether
            pkt = Ether(msg.data)
        except Exception:
            pass

        if pkt and pkt.haslayer(Ether):
            src = pkt[Ether].src.lower()
            dst = pkt[Ether].dst.lower()
            # learn src mac -> in_port
            self.mac_to_port[src] = in_port
            # optional: log
            self.logger.debug(f"Learned MAC {src} -> port {in_port}")
