from ryu.base import app_manager
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.controller import ofp_event
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet as ethernet_pkt, ipv4, udp
from ryu.lib.packet import ether_types
from ryu.ofproto import ofproto_v1_3
import csv
import random
import time


class IoTDoSReplayer(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IoTDoSReplayer, self).__init__(*args, **kwargs)

        # CSV path (change if needed)
        self.csv_file = '/home/omar/SDN_IoT_Simulated/iot_traffic.csv'
        self.traffic_data = []
        self.device_map = self._load_device_map()

        # Replay timing state
        self.start_time = None

        # DoS attack config (tweak as you like)
        self.malicious_mac = "aa:bb:cc:dd:ee:ff"
        self.malicious_ip = "192.168.1.250"
        self.attack_start_time = 6   # seconds after replay thread starts
        self.attack_duration = 10    # seconds of flood
        self.attack_rate = 100       # packets per second

        # datapath (set once switch connects)
        self.dp = None

        # Flags
        self.attack_launched = False

        # Spawn replay and scheduler in background
        hub.spawn(self.replay_traffic)
        hub.spawn(self.attack_scheduler)

    def _load_device_map(self):
        """Maps lowercase MAC addresses to device names"""
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
            "40:f3:08:ff:1e:da": "Android Phone 1",
            "74:2f:68:81:69:42": "Laptop",
            "ac:bc:32:d4:6f:2f": "MacBook",
            "b4:ce:f6:a7:a3:c2": "Android Phone 2",
            "d0:a6:37:df:a1:e1": "iPhone",
            "f4:5c:89:93:cc:85": "MacBook/iPhone",
            "14:cc:20:51:33:ea": "TPLink Router (Gateway)",
            # include malicious if you want it named
            "aa:bb:cc:dd:ee:ff": "Malicious Host (sim)",
        }

    def replay_traffic(self):
        """Read CSV into memory, then replay rows with delay & log normal packets."""
        self.logger.info("Loading IoT traffic data from CSV...")
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.traffic_data.append(row)
        except FileNotFoundError:
            self.logger.error(f"CSV file '{self.csv_file}' not found.")
            return

        self.logger.info("Traffic loaded. Beginning simulation...")
        replay_start = time.time()
        self.start_time = None

        for row in self.traffic_data:
            # If the datapath isn't ready yet, still produce logs; sending packets happens only if dp exists.
            # Respect CSV TIME field for delay if present
            try:
                if self.start_time is None:
                    self.start_time = float(row.get('TIME', 0) or 0)
                    delay = 0
                else:
                    now_time = float(row.get('TIME', 0) or 0)
                    delay = now_time - self.start_time
                    self.start_time = now_time
                if delay > 0:
                    hub.sleep(delay)
            except Exception:
                # faulty or missing time -> small spacing to avoid tight loop
                hub.sleep(0.01)

            # Log the normal packet (no color)
            self.log_packet(row, attack=False)

            # Optionally: actually send the packet into the datapath (if you want replayed packets on wire)
            # If desired, build and send here like in run_dos_attack() using self.dp
            # For now, we only log normal traffic to match your previous behavior.

    def log_packet(self, row, attack=False):
        """Logs packet with device names and protocol tag.
        Attack logs are printed in red to stand out (ANSI)."""
        src_mac = row.get('eth.src', '').strip().lower()
        dst_mac = row.get('eth.dst', '').strip().lower()
        src_ip = row.get('IP.src', '').strip()
        dst_ip = row.get('IP.dst', '').strip()

        proto_num = row.get('IP.proto', '').strip()
        if proto_num == '6':
            proto = 'TCP'
        elif proto_num == '17':
            proto = 'UDP'
        elif proto_num == '1':
            proto = 'ICMP'
        else:
            proto = f"Proto{proto_num}" if proto_num else "IP"

        src_device = self.device_map.get(src_mac, f"Unknown({src_mac})")
        dst_device = self.device_map.get(dst_mac, f"Unknown({dst_mac})")

        tag = "[ATTACK]" if attack else "[NORMAL]"
        message = f"{tag} [{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip})"

        if attack:
            # red highlight for attack
            print(f"\033[91m{message}\033[0m")
        else:
            self.logger.info(message)

    def attack_scheduler(self):
        """Wait until datapath ready (or timeout) then wait attack_start_time and spawn the attack thread."""
        # Wait up to a short while for datapath to be set (non-blocking)
        waited = 0
        while self.dp is None and waited < 10:
            hub.sleep(0.5)
            waited += 0.5

        # Wait attack_start_time seconds (relative to scheduler start)
        hub.sleep(self.attack_start_time)
        # Spawn real attack in background so replay continues
        if not self.attack_launched:
            hub.spawn(self.run_dos_attack)
            self.attack_launched = True

    def run_dos_attack(self):
        """Actual DoS flood: pick random victims from traffic_data, send UDP packets, and log each packet."""
        self.logger.info(f"*** DoS attack starting for {self.attack_duration} seconds at {self.attack_rate} pps ***")
        end_time = time.time() + self.attack_duration
        packets_per_sec = max(1, int(self.attack_rate))

        # compute total packets to send (approx)
        total_packets = int(self.attack_duration * packets_per_sec)

        sent = 0
        while time.time() < end_time:
            loop_start = time.time()
            for _ in range(packets_per_sec):
                # choose victim row (if no traffic_data, pick random mac from map)
                if self.traffic_data:
                    victim = random.choice(self.traffic_data)
                    dst_mac = victim.get('eth.dst', '').strip().lower()
                    dst_ip = victim.get('IP.dst', '').strip()
                else:
                    dst_mac = random.choice(list(self.device_map.keys()))
                    dst_ip = ""  # unknown

                # build malicious packet (if datapath exists, send it)
                pkt = packet.Packet()
                pkt.add_protocol(ethernet_pkt.ethernet(
                    ethertype=ether_types.ETH_TYPE_IP,
                    src=self.malicious_mac,
                    dst=dst_mac
                ))
                pkt.add_protocol(ipv4.ipv4(
                    src=self.malicious_ip,
                    dst=dst_ip or "0.0.0.0",
                    proto=17  # UDP
                ))
                pkt.add_protocol(udp.udp(
                    src_port=random.randint(1024, 65535),
                    dst_port=80
                ))
                pkt.serialize()

                # If datapath available, send the packet out (flood)
                if self.dp:
                    parser = self.dp.ofproto_parser
                    ofproto = self.dp.ofproto
                    actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                    out = parser.OFPPacketOut(
                        datapath=self.dp,
                        buffer_id=ofproto.OFP_NO_BUFFER,
                        in_port=ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=pkt.data
                    )
                    try:
                        self.dp.send_msg(out)
                    except Exception as e:
                        # sending failed but continue logging
                        self.logger.debug(f"Failed to send packet_out: {e}")

                # Log the malicious packet exactly like normal logs but tagged and red
                row = {
                    'eth.src': self.malicious_mac,
                    'eth.dst': dst_mac,
                    'IP.src': self.malicious_ip,
                    'IP.dst': dst_ip,
                    'IP.proto': '17'
                }
                self.log_packet(row, attack=True)

                sent += 1

            # maintain approximate rate per second
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0, 1.0 - loop_elapsed)
            if sleep_time > 0:
                hub.sleep(sleep_time)

        self.logger.info(f"*** DoS attack finished - packets sent (approx): {sent} ***")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Set datapath when switch connects and install simple table-miss flow (optional)."""
        self.dp = ev.msg.datapath
        ofproto = self.dp.ofproto
        parser = self.dp.ofproto_parser

        # optional: add table-miss to send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=self.dp, priority=0, match=match, instructions=inst)
        try:
            self.dp.send_msg(mod)
        except Exception:
            pass

        self.logger.info("Switch connected. Ready to replay IoT traffic and launch DoS.")
