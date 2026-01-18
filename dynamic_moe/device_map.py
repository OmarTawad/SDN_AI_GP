"""IoT device inventory used by the dynamic MoE Mininet topology."""

from __future__ import annotations

from typing import Dict

DEVICE_MAP: Dict[str, Dict[str, str]] = {
    "d0:52:a8:00:67:5e": {"name": "SmartThings", "type": "iot", "link": "wired"},
    "44:65:0d:56:cc:d3": {"name": "Amazon Echo", "type": "iot", "link": "wireless"},
    "70:ee:50:18:34:43": {"name": "Netatmo Welcome", "type": "iot", "link": "wireless"},
    "f4:f2:6d:93:51:f1": {"name": "TP-Link Day Night camera", "type": "camera", "link": "wireless"},
    "00:16:6c:ab:6b:88": {"name": "Samsung SmartCam", "type": "camera", "link": "wired"},
    "30:8c:fb:2f:e4:b2": {"name": "Dropcam", "type": "camera", "link": "wireless"},
    "00:62:6e:51:27:2e": {"name": "Insteon Camera (wired)", "type": "camera", "link": "wired"},
    "e8:ab:fa:19:de:4f": {"name": "Insteon Camera (wireless)", "type": "camera", "link": "wireless"},
    "00:24:e4:11:18:a8": {"name": "Withings Smart Baby Monitor", "type": "iot", "link": "wireless"},
    "ec:1a:59:79:f4:89": {"name": "Belkin Wemo switch", "type": "iot", "link": "wireless"},
    "50:c7:bf:00:56:39": {"name": "TP-Link Smart plug", "type": "iot", "link": "wireless"},
    "74:c6:3b:29:d7:1d": {"name": "iHome", "type": "iot", "link": "wireless"},
    "ec:1a:59:83:28:11": {"name": "Belkin Wemo motion sensor", "type": "sensor", "link": "wireless"},
    "18:b4:30:25:be:e4": {"name": "NEST Protect smoke alarm", "type": "sensor", "link": "wireless"},
    "70:ee:50:03:b8:ac": {"name": "Netatmo weather station", "type": "iot", "link": "wireless"},
    "00:24:e4:1b:6f:96": {"name": "Withings Smart scale", "type": "iot", "link": "wireless"},
    "74:6a:89:00:2e:25": {"name": "Blipcare BP meter", "type": "medical", "link": "wireless"},
    "00:24:e4:20:28:c6": {"name": "Withings Aura sleep sensor", "type": "medical", "link": "wireless"},
    "d0:73:d5:01:83:08": {"name": "LiFX Smart Bulb", "type": "iot", "link": "wireless"},
    "18:b7:9e:02:20:44": {"name": "Triby Speaker", "type": "iot", "link": "wireless"},
    "e0:76:d0:33:bb:85": {"name": "PIX-STAR Photo-frame", "type": "iot", "link": "wireless"},
    "70:5a:0f:e4:9b:c0": {"name": "HP Printer", "type": "printer", "link": "wired"},
    "08:21:ef:3b:fc:e3": {"name": "Samsung Galaxy Tab", "type": "mobile", "link": "wireless"},
    "30:8c:fb:b6:ea:45": {"name": "Nest Dropcam", "type": "camera", "link": "wireless"},
    "40:f3:08:ff:1e:da": {"name": "Android Phone 1", "type": "mobile", "link": "wireless"},
    "74:2f:68:81:69:42": {"name": "Laptop", "type": "laptop", "link": "wireless"},
    "ac:bc:32:d4:6f:2f": {"name": "MacBook", "type": "laptop", "link": "wireless"},
    "b4:ce:f6:a7:a3:c2": {"name": "Android Phone 2", "type": "mobile", "link": "wireless"},
    "d0:a6:37:df:a1:e1": {"name": "iPhone", "type": "mobile", "link": "wireless"},
    "f4:5c:89:93:cc:85": {"name": "MacBook/iPhone", "type": "mobile", "link": "wireless"},
    "14:cc:20:51:33:ea": {"name": "HomeGateway", "type": "router", "link": "wired"},
}


def normalize_mac(mac: str) -> str:
    """Return a lower-cased MAC string for consistent dictionary lookups."""

    return mac.lower()


def device_lookup(mac: str) -> Dict[str, str] | None:
    """Lookup helper that returns device metadata for a MAC address."""

    return DEVICE_MAP.get(normalize_mac(mac))


__all__ = ["DEVICE_MAP", "device_lookup", "normalize_mac"]
