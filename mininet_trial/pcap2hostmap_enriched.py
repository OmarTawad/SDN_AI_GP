#!/usr/bin/env python3
import json, ipaddress
from scapy.all import PcapReader, Ether, IP, ARP
from devices_inventory import DEVICES

PCAP = "/home/omar/18-10-27.pcap"
OUT  = "iot_hostmap.json"  # list of host dicts for easier consumption

def is_private_iot(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        return ip_obj.is_private and not (ip_obj.is_loopback or ip_obj.is_unspecified or ip_obj.is_link_local or ip_obj.is_multicast)
    except ValueError:
        return False

# Learn IP->MAC (prefer ARP)
ip2mac = {}
with PcapReader(PCAP) as r:
    for pkt in r:
        if ARP in pkt:
            if is_private_iot(pkt[ARP].psrc):
                ip2mac.setdefault(pkt[ARP].psrc, pkt[ARP].hwsrc.lower())
        elif IP in pkt and Ether in pkt:
            if is_private_iot(pkt[IP].src):
                ip2mac.setdefault(pkt[IP].src, pkt[Ether].src.lower())

# Emit list of hosts: {ip, mac, name, conn}
hosts = []
seen = set()
for ip, mac in sorted(ip2mac.items(), key=lambda kv: kv[0]):
    name = DEVICES.get(mac, {}).get("name", f"Unknown-{mac}")
    conn = DEVICES.get(mac, {}).get("conn", "Unknown")
    key = (ip, mac)
    if key in seen: 
        continue
    hosts.append({"ip": ip, "mac": mac, "name": name, "conn": conn})
    seen.add(key)

print(f"Found {len(hosts)} private hosts")
with open(OUT, "w") as f:
    json.dump(hosts, f, indent=2)
print(f"Wrote {OUT}")
