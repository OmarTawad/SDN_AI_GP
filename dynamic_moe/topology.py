"""Mininet topology helpers for the IoT home network."""

from __future__ import annotations

import re
from typing import Dict

from .device_map import DEVICE_MAP


MAX_MININET_NAME = 10  # leave room for '-ethX' suffix within kernel ifname limits


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"h_{slug or 'device'}"


def _shorten(name: str, existing: set[str]) -> str:
    base = name[:MAX_MININET_NAME]
    if len(base) < 3:
        base = (name + "_iot")[:MAX_MININET_NAME]
    candidate = base
    counter = 1
    while candidate in existing:
        suffix = f"{counter:02d}"
        candidate = f"{base[:MAX_MININET_NAME - len(suffix)]}{suffix}"
        counter += 1
    existing.add(candidate)
    return candidate


def build_dynamic_moe_topology(
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6633,
) -> "Mininet":
    """Instantiate a Mininet containing all IoT devices plus the home gateway."""

    try:
        from mininet.net import Mininet
        from mininet.node import OVSKernelSwitch, RemoteController
        from mininet.topo import Topo
    except ImportError as exc:  # pragma: no cover - dependency missing during linting
        raise ImportError(
            "Mininet is required to build the dynamic MoE topology. "
            "Install it from https://github.com/mininet/mininet."
        ) from exc

    class IoTHomeTopo(Topo):
        def build(self) -> None:
            switch = self.addSwitch("s1", protocols="OpenFlow13")
            host_metadata: Dict[str, Dict[str, str]] = {}
            alias_map: Dict[str, str] = {}
            assigned_names: set[str] = set()
            host_index = 2
            for mac, meta in DEVICE_MAP.items():
                mac_norm = mac.lower()
                friendly_name = _slug(meta["name"])
                host_name = _shorten(friendly_name, assigned_names)
                alias_map[friendly_name] = host_name
                is_router = meta.get("type") == "router"
                if is_router:
                    ip_cidr = "10.0.0.1/24"
                    default_route = None
                else:
                    ip_cidr = f"10.0.0.{host_index}/24"
                    default_route = "via 10.0.0.1"
                self.addHost(
                    host_name,
                    mac=mac_norm,
                    ip=ip_cidr,
                    defaultRoute=default_route,
                )
                self.addLink(host_name, switch)
                host_metadata[host_name] = {
                    "mac": mac_norm,
                    "ip": ip_cidr,
                    "device_name": meta.get("name"),
                    "device_type": meta.get("type"),
                    "link": meta.get("link"),
                    "friendly_name": friendly_name,
                }
                host_index += 1
            self.host_metadata = host_metadata
            self.alias_map = alias_map

    topo = IoTHomeTopo()
    controller_factory = lambda name: RemoteController(name, ip=controller_ip, port=controller_port)
    net = Mininet(
        topo=topo,
        controller=controller_factory,
        switch=OVSKernelSwitch,
        autoSetMacs=False,
        autoStaticArp=False,
    )
    net.host_metadata = getattr(topo, "host_metadata", {})  # type: ignore[attr-defined]
    alias_map = getattr(topo, "alias_map", {})  # type: ignore[attr-defined]
    for alias, actual in alias_map.items():
        if alias in net.nameToNode or actual not in net.nameToNode:
            continue
        net.nameToNode[alias] = net.nameToNode[actual]
    net.host_aliases = alias_map
    return net


__all__ = ["build_dynamic_moe_topology"]
