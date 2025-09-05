from ryu.base import app_manager
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.controller import ofp_event
from ryu.lib import hub
from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP
import logging

# Terminal color codes
RED = "\033[91m"
ENDC = "\033[0m"
YELLOW = "\033[93m"

class IoTTrafficReplayer(app_manager.RyuApp):
    def __init__(self, *args, **kwargs):
        super(IoTTrafficReplayer, self).__init__(*args, **kwargs)
        self.logger.setLevel(logging.INFO)

        # PCAP file path
        self.pcap_file = '/home/omar/SDN_IoT_Simulated/18-10-27.pcap'

        # Mapping of MAC addresses to friendly names
        self.device_map = self._load_device_map()

        # Define attack MACs and attack names
        self.attack_macs = {
            "00:17:88:2b:9a:25": "Huebulb Attack",
            "f4:f5:d8:8f:0a:3c": "Chromecast Flood",
            # Add more attack MACs and types here
        }

        # Start the traffic replay thread
        self.thread = hub.spawn(self.replay_traffic)

    def _load_device_map(self):
        """All IoT devices mapped by MAC"""
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

    def replay_traffic(self):
        self.logger.info(f"Streaming IoT traffic from PCAP: {self.pcap_file}")
        try:
            with PcapReader(self.pcap_file) as pcap_reader:
                for pkt in pcap_reader:
                    hub.sleep(0.001)  # Slight delay for logging readability

                    if not pkt.haslayer(IP):
                        continue

                    src_mac = pkt[Ether].src.lower() if pkt.haslayer(Ether) else "N/A"
                    dst_mac = pkt[Ether].dst.lower() if pkt.haslayer(Ether) else "N/A"
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst

                    # Determine protocol
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

                    # Highlight attack devices in red and show attack type
                    if src_mac in self.attack_macs or dst_mac in self.attack_macs:
                        attack_name = self.attack_macs.get(src_mac, self.attack_macs.get(dst_mac, "Unknown Attack"))
                        log_msg = f"{RED}[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip}) [ATTACK: {attack_name}]{ENDC}"
                    # Highlight unknown devices in yellow
                    elif "Unknown" in src_device or "Unknown" in dst_device:
                        log_msg = f"{YELLOW}[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip}){ENDC}"
                    else:
                        log_msg = f"[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip})"

                    self.logger.info(log_msg)

        except FileNotFoundError:
            self.logger.error(f"PCAP file '{self.pcap_file}' not found.")
        except Exception as e:
            self.logger.error(f"Error reading PCAP: {e}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, MAIN_DISPATCHER)
    def switch_features_handler(self, ev):
        self.logger.info("Switch connected. Ready to replay IoT traffic.")
