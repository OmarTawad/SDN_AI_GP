from ryu.base import app_manager
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.controller import ofp_event
from ryu.lib import hub
import csv

class IoTTrafficReplayer(app_manager.RyuApp):
    def __init__(self, *args, **kwargs):
        super(IoTTrafficReplayer, self).__init__(*args, **kwargs)
        self.csv_file = '/home/omar/SDN_IoT_Simulated/iot_traffic.csv'
        self.traffic_data = []
        self.device_map = self._load_device_map()
        self.start_time = None
        self.thread = hub.spawn(self.replay_traffic)

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
            "14:cc:20:51:33:ea": "TPLink Router (Gateway)"
        }

    def replay_traffic(self):
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
        self.start_time = None

        for row in self.traffic_data:
            try:
                current_time = float(row.get('TIME', 0))
                if self.start_time is None:
                    self.start_time = current_time
                    delay = 0
                else:
                    delay = current_time - self.start_time
                    self.start_time = current_time

                if delay > 0:
                    hub.sleep(delay)
            except Exception as e:
                self.logger.error(f"Timestamp parsing error: {e}")
                hub.sleep(1)

            # Normalize MAC addresses and IPs
            src_mac = row.get('eth.src', '').strip().lower()
            dst_mac = row.get('eth.dst', '').strip().lower()
            src_ip = row.get('IP.src', '').strip()
            dst_ip = row.get('IP.dst', '').strip()

            # Map IP.proto number to protocol name
            proto_num = row.get('IP.proto', '').strip()
            if proto_num == '6':
                proto = 'TCP'
            elif proto_num == '17':
                proto = 'UDP'
            elif proto_num == '1':
                proto = 'ICMP'
            else:
                proto = f"Proto{proto_num}"

            # Look up device names
            src_device = self.device_map.get(src_mac, f"Unknown({src_mac})")
            dst_device = self.device_map.get(dst_mac, f"Unknown({dst_mac})")

            self.logger.info(
                f"[{proto}] {src_device} ({src_ip}) --> {dst_device} ({dst_ip})"
            )

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, MAIN_DISPATCHER)
    def switch_features_handler(self, ev):
        self.logger.info("Switch connected. Ready to replay IoT traffic.")
