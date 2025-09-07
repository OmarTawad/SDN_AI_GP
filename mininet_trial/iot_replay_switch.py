#!/usr/bin/env python3
import json, ipaddress, logging, time
from datetime import datetime, timezone
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP, ARP

# bring your attacker windows
from attackers_inventory import build_indexes

RED = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"; ENDC = "\033[0m"

class IoTTrafficReplayer(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(IoTTrafficReplayer, self).__init__(*args, **kwargs)
        self.logger.setLevel(logging.INFO)  # set to DEBUG for more detail

        # === Paths ===
        self.pcap_file  = "/home/omar/18-10-27.pcap"
        self.hosts_file = "/home/omar/SDN_IoT_Simulated/mininet_trial/iot_hostmap.json"

        # === Replay timing ===
        self.use_capture_timing = True     # use pcap timestamps
        self.speed_factor       = 50.0      # 1.0 real-time; >1 faster; <1 slower

        # === Live traffic visibility (optional) ===
        self.mirror_icmp_to_controller = False

        # === Enforcement during attack windows ===
        # mode: "meter" (throttle), "drop_icmp" (only ICMP), "drop_all", or "none"
        self.enforce_mode        = "none"
        self.block_seconds       = 15            # duration per detection
        self.meter_rate_kbps     = 128           # throttle rate for attacker MAC
        self.meter_burst_kbytes  = 32            # small burst
        # internal meter tracking
        self.next_meter_id = 1
        self.meter_for_mac = {}  # mac -> {"id": int, "expires": float}

        # === Optional rewrite: map private IPs to Mininet MACs from host map ===
        self.rewrite_to_mininet = True
        self.external_sink_ip   = None  # e.g. "192.168.1.108" to collapse externals

        # === Labels and rewrites ===
        self.mac_to_name = {}
        self.ip_to_mac   = {}
        try:
            hosts = json.load(open(self.hosts_file))
            for h in hosts:
                mac = h["mac"].lower()
                name = h.get("name", mac)
                self.mac_to_name[mac] = name
                if "ip" in h and h["ip"]:
                    self.ip_to_mac[h["ip"]] = mac
        except Exception as e:
            self.logger.warning(f"Could not load host map '{self.hosts_file}': {e}. Proceeding without names/rewrites.")

        # === Attacker windows (from your inventory) ===
        self.att_by_mac, self.att_by_ip, self.att_windows = build_indexes()

        # === Learning-switch state ===
        self.mac_to_port = {}

        # === OF handles ===
        self.datapath = None
        self.ofproto  = None
        self.parser   = None
        self.started  = False

    # ---------- helpers ----------
    def _is_private(self, ip):
        try:
            o = ipaddress.ip_address(ip)
            return o.is_private and not (o.is_loopback or o.is_unspecified or o.is_link_local or o.is_multicast)
        except ValueError:
            return False

    def _pkt_date(self, pkt):
        try:
            ts = float(pkt.time)
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except Exception:
            return None

    def _match_attacker(self, pkt):
        d = self._pkt_date(pkt)
        if d is None:
            return None
        mac_src = mac_dst = ip_src = ip_dst = None
        if Ether in pkt:
            mac_src = pkt[Ether].src.lower()
            mac_dst = pkt[Ether].dst.lower()
        if IP in pkt:
            ip_src = pkt[IP].src
            ip_dst = pkt[IP].dst
        # MAC windows
        for m in (mac_src, mac_dst):
            if not m: continue
            for start, end, dev, ip in self.att_by_mac.get(m, []):
                if start <= d <= end:
                    return (dev, f"mac={m}")
        # IP windows
        for ip in (ip_src, ip_dst):
            if not ip: continue
            for start, end, dev, mac in self.att_by_ip.get(ip, []):
                if start <= d <= end:
                    return (dev, f"ip={ip}")
        return None

    def _name_for_mac(self, mac):
        mac = mac.lower()
        return self.mac_to_name.get(mac, f"Unknown-{mac}")

    def _rewrite(self, pkt):
        if IP not in pkt or Ether not in pkt:
            return pkt
        eth = pkt[Ether]; ip = pkt[IP]
        # collapse externals if desired
        if not self._is_private(ip.dst) and self.external_sink_ip:
            ip.dst = self.external_sink_ip
        # align to Mininet MACs
        if self._is_private(ip.src) and ip.src in self.ip_to_mac:
            eth.src = self.ip_to_mac[ip.src]
        if self._is_private(ip.dst) and ip.dst in self.ip_to_mac:
            eth.dst = self.ip_to_mac[ip.dst]
        # recompute checksums
        if hasattr(ip, 'chksum'): del ip.chksum
        if TCP in pkt and hasattr(pkt[TCP], 'chksum'): del pkt[TCP].chksum
        if UDP in pkt and hasattr(pkt[UDP], 'chksum'): del pkt[UDP].chksum
        return pkt

    def _choose_out_port(self, dst_mac):
        return self.mac_to_port.get(dst_mac.lower(), self.ofproto.OFPP_FLOOD)

    # ---------- enforcement primitives ----------
    def _install_icmp_drop(self, mac, seconds=None, priority=220):
        if seconds is None: seconds = self.block_seconds
        p, dp, ofp = self.parser, self.datapath, self.ofproto
        mac = mac.lower()
        # src -> any (ICMP)
        m = p.OFPMatch(eth_type=0x0800, ip_proto=1, eth_src=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m,
                                 instructions=[], idle_timeout=seconds, hard_timeout=0))
        # any -> dst (ICMP)
        m = p.OFPMatch(eth_type=0x0800, ip_proto=1, eth_dst=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m,
                                 instructions=[], idle_timeout=seconds, hard_timeout=0))

    def _install_all_drop(self, mac, seconds=None, priority=220):
        if seconds is None: seconds = self.block_seconds
        p, dp = self.parser, self.datapath
        mac = mac.lower()
        m = p.OFPMatch(eth_src=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m,
                                 instructions=[], idle_timeout=seconds, hard_timeout=0))
        m = p.OFPMatch(eth_dst=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m,
                                 instructions=[], idle_timeout=seconds, hard_timeout=0))

    def _ensure_meter(self, mac):
        """Create a meter if not present."""
        mac = mac.lower()
        if mac in self.meter_for_mac:
            return self.meter_for_mac[mac]["id"]
        p, dp, ofp = self.parser, self.datapath, self.ofproto
        meter_id = self.next_meter_id; self.next_meter_id += 1
        band = p.OFPMeterBandDrop(rate=self.meter_rate_kbps, burst_size=self.meter_burst_kbytes * 1024 // 8)
        flags = ofp.OFPMF_KBPS | ofp.OFPMF_BURST
        dp.send_msg(p.OFPMeterMod(datapath=dp, command=ofp.OFPMC_ADD,
                                  flags=flags, meter_id=meter_id, bands=[band]))
        self.meter_for_mac[mac] = {"id": meter_id, "expires": 0}
        return meter_id

    def _apply_meter_flows(self, mac, seconds=None, priority=50):
        """Apply meter to traffic to/from MAC using NORMAL forwarding (so switching still works)."""
        if seconds is None: seconds = self.block_seconds
        mac = mac.lower()
        meter_id = self._ensure_meter(mac)
        self.meter_for_mac[mac]["expires"] = time.time() + seconds

        p, dp, ofp = self.parser, self.datapath, self.ofproto
        inst = lambda: [
            p.OFPInstructionMeter(meter_id),
            p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(ofp.OFPP_NORMAL)])
        ]
        # src -> any
        m1 = p.OFPMatch(eth_src=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m1,
                                 instructions=inst(), idle_timeout=seconds, hard_timeout=0))
        # any -> dst
        m2 = p.OFPMatch(eth_dst=mac)
        dp.send_msg(p.OFPFlowMod(datapath=dp, priority=priority, match=m2,
                                 instructions=inst(), idle_timeout=seconds, hard_timeout=0))

        # start/refresh a cleanup loop per MAC
        if "cleaner" not in self.meter_for_mac[mac]:
            self.meter_for_mac[mac]["cleaner"] = hub.spawn(self._meter_cleanup_loop, mac)

    def _meter_cleanup_loop(self, mac):
        mac = mac.lower()
        p, dp, ofp = self.parser, self.datapath, self.ofproto
        while True:
            hub.sleep(1.0)
            if mac not in self.meter_for_mac:
                return
            if time.time() < self.meter_for_mac[mac]["expires"]:
                continue
            # delete flows
            for fld in (("eth_src", mac), ("eth_dst", mac)):
                if fld[0] == "eth_src":
                    match = p.OFPMatch(eth_src=fld[1])
                else:
                    match = p.OFPMatch(eth_dst=fld[1])
                dp.send_msg(p.OFPFlowMod(datapath=dp,
                                         command=ofp.OFPFC_DELETE,
                                         out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY,
                                         priority=50, match=match))
            # delete meter
            meter_id = self.meter_for_mac[mac]["id"]
            dp.send_msg(p.OFPMeterMod(datapath=dp, command=ofp.OFPMC_DELETE,
                                      flags=0, meter_id=meter_id, bands=[]))
            del self.meter_for_mac[mac]
            return

    # ---------- switch connect ----------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.datapath = dp
        self.ofproto  = dp.ofproto
        self.parser   = dp.ofproto_parser

        # Drop LLDP
        match = self.parser.OFPMatch(eth_type=0x88cc)
        dp.send_msg(self.parser.OFPFlowMod(datapath=dp, priority=100, match=match, instructions=[]))

        # ARP: mirror to controller + flood (fast learning)
        match = self.parser.OFPMatch(eth_type=0x0806)
        actions = [
            self.parser.OFPActionOutput(self.ofproto.OFPP_CONTROLLER, self.ofproto.OFPCML_NO_BUFFER),
            self.parser.OFPActionOutput(self.ofproto.OFPP_FLOOD),
        ]
        inst = [self.parser.OFPInstructionActions(self.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(self.parser.OFPFlowMod(datapath=dp, priority=90, match=match, instructions=inst))

        # Table-miss to controller
        match = self.parser.OFPMatch()
        actions = [self.parser.OFPActionOutput(self.ofproto.OFPP_CONTROLLER, self.ofproto.OFPCML_NO_BUFFER)]
        inst = [self.parser.OFPInstructionActions(self.ofproto.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(self.parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst))

        self.logger.info("Switch connected. LLDP drop + ARP mirror/flood + table-miss installed.")

        if not self.started:
            self.started = True
            hub.spawn(self.replay_traffic)

    # ---------- learning switch ----------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto; parser = dp.ofproto_parser
        in_port = msg.match.get('in_port')

        eth = Ether(msg.data)
        src = eth.src.lower(); dst = eth.dst.lower()

        # learn
        self.mac_to_port[src] = in_port

        # log the first ICMP we see between any pair
        try:
            if IP in eth and ICMP in eth:
                s = eth[IP].src; d = eth[IP].dst
                self.logger.info(f"{GRN}[PING(first)] {s} -> {d}  ({src} -> {dst}){ENDC}")
        except Exception:
            pass

        out_port = self.mac_to_port.get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if self.mirror_icmp_to_controller and IP in eth and ICMP in eth:
            actions.append(parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER))

        # install unicast flow
        if out_port != ofp.OFPP_FLOOD:
            if IP in eth and ICMP in eth:
                match = parser.OFPMatch(eth_type=0x0800, ip_proto=1, eth_src=src, eth_dst=dst)
                prio = 2
            else:
                match = parser.OFPMatch(eth_src=src, eth_dst=dst)
                prio = 1
            inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
            dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=prio, match=match,
                                          instructions=inst, idle_timeout=60, hard_timeout=0))

        # forward this packet now
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=ofp.OFP_NO_BUFFER,
                                        in_port=in_port, actions=actions, data=msg.data))

    # ---------- replay thread (capture-timed) ----------
    def replay_traffic(self):
        if not self.datapath:
            self.logger.error("No datapath; aborting replay.")
            return

        self.logger.info(f"Replaying PCAP at x{self.speed_factor} timing: {self.pcap_file}")

        cap_t0 = None
        wall_t0 = None
        n = 0
        ofp = self.ofproto

        try:
            with PcapReader(self.pcap_file) as r:
                for pkt in r:
                    if Ether not in pkt:
                        continue
                    # ignore LLDP frames in PCAP
                    if pkt[Ether].type == 0x88cc:
                        continue

                    # pace by capture timestamp
                    if self.use_capture_timing:
                        t_cap = float(pkt.time)
                        if cap_t0 is None:
                            cap_t0 = t_cap
                            wall_t0 = time.time()
                        desired = wall_t0 + (t_cap - cap_t0) / max(self.speed_factor, 1e-6)
                        while True:
                            now = time.time()
                            if now >= desired: break
                            hub.sleep(min(0.05, desired - now))
                    else:
                        hub.sleep(0.001)

                    # rewrite
                    if self.rewrite_to_mininet:
                        pkt = self._rewrite(pkt)

                    # attack match & enforcement
                    hit = self._match_attacker(pkt)
                    sm = pkt[Ether].src.lower()
                    dm = pkt[Ether].dst.lower()
                    src_label = self._name_for_mac(sm)
                    dst_label = self._name_for_mac(dm)

                    if hit:
                        dev, why = hit
                        # choose the MAC to enforce on
                        mac_to_act = None
                        if "mac=" in why:
                            mac_to_act = why.split("mac=")[1].strip().lower()
                        elif IP in pkt:
                            for ip_ in (pkt[IP].src, pkt[IP].dst):
                                if ip_ in self.ip_to_mac:
                                    mac_to_act = self.ip_to_mac[ip_]; break
                        if mac_to_act:
                            if self.enforce_mode == "drop_icmp":
                                self._install_icmp_drop(mac_to_act, seconds=self.block_seconds)
                            elif self.enforce_mode == "drop_all":
                                self._install_all_drop(mac_to_act, seconds=self.block_seconds)
                            elif self.enforce_mode == "meter":
                                self._apply_meter_flows(mac_to_act, seconds=self.block_seconds)

                    # logs
                    if IP in pkt:
                        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "ICMP" if ICMP in pkt else f"IP{pkt[IP].proto}"
                        s, d = pkt[IP].src, pkt[IP].dst
                        if hit:
                            dev, why = hit
                            self.logger.info(f"{RED}[{proto}] {src_label}({s}) -> {dst_label}({d})  [ATTACK: {dev}; {why}]{ENDC}")
                        else:
                            if (n % 1000) == 0:
                                col = "" if (self._is_private(s) and self._is_private(d)) else YEL
                                self.logger.info(f"{col}[{proto}] {src_label}({s}) -> {dst_label}({d}){ENDC}")
                    else:
                        if hit:
                            dev, why = hit
                            self.logger.info(f"{RED}[L2] {src_label} -> {dst_label}  [ATTACK: {dev}; {why}]{ENDC}")
                        else:
                            if (n % 1000) == 0:
                                self.logger.info(f"[L2] {src_label} -> {dst_label}")

                    # inject to dataplane
                    out_port = self._choose_out_port(pkt[Ether].dst)
                    actions  = [self.parser.OFPActionOutput(out_port)]
                    out = self.parser.OFPPacketOut(
                        datapath=self.datapath,
                        buffer_id=self.ofproto.OFP_NO_BUFFER,
                        in_port=self.ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=bytes(pkt)
                    )
                    self.datapath.send_msg(out)
                    n += 1

            self.logger.info("Replay finished.")
        except FileNotFoundError:
            self.logger.error(f"PCAP '{self.pcap_file}' not found.")
        except Exception as e:
            self.logger.error(f"Replay error: {e}")
