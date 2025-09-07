# attackers_inventory.py
# All MACs/IPs normalized lower-case; date format: DD/MM/YYYY
ATTACKERS = [
    # From your table (May–Jun 2018 campaign)
    {"mac": "ec:1a:59:83:28:11", "ip": "192.168.1.165", "start": "28/05/2018", "end": "17/06/2018", "device": "WEMO Motion Sensor"},
    {"mac": "ec:1a:59:79:f4:89", "ip": "192.168.1.223", "start": "28/05/2018", "end": "17/06/2018", "device": "WEMO Power Switch"},
    {"mac": "00:16:6c:ab:6b:88", "ip": "192.168.1.248", "start": "28/05/2018", "end": "17/06/2018", "device": "Samsung Camera"},
    {"mac": "50:c7:bf:00:56:39", "ip": "192.168.1.227", "start": "28/05/2018", "end": "17/06/2018", "device": "TP-Link Plug"},
    {"mac": "70:ee:50:18:34:43", "ip": "192.168.1.241", "start": "28/05/2018", "end": "17/06/2018", "device": "Netatmo Camera"},

    # From your table (Sep–Oct 2018 campaign)
    {"mac": "00:17:88:2b:9a:25", "ip": "192.168.1.129", "start": "24/09/2018", "end": "26/10/2018", "device": "Huebulb"},
    {"mac": "44:65:0d:56:cc:d3", "ip": "192.168.1.239", "start": "24/09/2018", "end": "26/10/2018", "device": "Amazon Echo"},
    {"mac": "f4:f5:d8:8f:0a:3c", "ip": "192.168.1.119", "start": "24/09/2018", "end": "26/10/2018", "device": "Chromecast"},
    {"mac": "74:c6:3b:29:d7:1d", "ip": "192.168.1.163", "start": "24/09/2018", "end": "26/10/2018", "device": "iHome"},
    {"mac": "d0:73:d5:01:83:08", "ip": "192.168.1.118", "start": "24/09/2018", "end": "26/10/2018", "device": "LiFX"},
]

# Optional convenience: build quick lookups
def build_indexes():
    from datetime import datetime
    def parse(d): return datetime.strptime(d, "%d/%m/%Y").date()

    by_mac = {}
    by_ip  = {}
    windows = []  # [(start_date, end_date, mac, ip, device)]
    for a in ATTACKERS:
        start = parse(a["start"]); end = parse(a["end"])
        mac = a["mac"].lower(); ip = a["ip"]; dev = a["device"]
        by_mac.setdefault(mac, []).append((start, end, dev, ip))
        by_ip.setdefault(ip, []).append((start, end, dev, mac))
        windows.append((start, end, mac, ip, dev))
    return by_mac, by_ip, windows
