"""Utilities for replaying PCAP captures inside Mininet hosts."""

from __future__ import annotations

import logging
import shutil
import textwrap
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("dynamic_moe.pcap_replay")


def _scapy_script(path: Path, iface: str, loop: bool, rate: Optional[str]) -> str:
    inter = 0.0
    if rate:
        try:
            value = float(rate)
            if value > 0:
                inter = 1.0 / value
        except ValueError:
            pass
    return textwrap.dedent(
        f"""
        from scapy.all import rdpcap, sendp
        pkts = rdpcap(r"{path}")
        sendp(pkts, iface="{iface}", loop={1 if loop else 0}, inter={inter}, verbose=False)
        """
    ).strip()


def replay_pcap_on_host(host, pcap_path: str | Path, loop: bool = False, rate: Optional[str] = None):
    """Replay a capture file through the namespace of ``host``."""

    path = Path(pcap_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {path}")
    iface = host.defaultIntf().name  # type: ignore[attr-defined]
    LOGGER.info("Replaying %s via %s (%s)", path.name, host.name, iface)
    if shutil.which("tcpreplay"):
        cmd = ["tcpreplay", "--intf1", iface]
        if loop:
            cmd.extend(["--loop", "0"])
        if rate:
            cmd.extend(["--pps", str(rate)])
        cmd.append(str(path))
        return host.popen(cmd)
    LOGGER.warning("tcpreplay missing; falling back to scapy sender.")
    script = _scapy_script(path, iface, loop, rate)
    return host.popen(["python3", "-c", script])


__all__ = ["replay_pcap_on_host"]
