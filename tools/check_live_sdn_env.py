"""Check whether the live Mininet/Ryu/OpenFlow demo can run."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys


def _version(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=5, check=False)
    except Exception as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return text[0] if text else f"exit={result.returncode}"


def main() -> int:
    status = {
        "is_root": os.geteuid() == 0,
        "mininet": {
            "path": shutil.which("mn"),
            "version": _version(["mn", "--version"]) if shutil.which("mn") else None,
        },
        "ovs_vsctl": {
            "path": shutil.which("ovs-vsctl"),
            "version": _version(["ovs-vsctl", "--version"]) if shutil.which("ovs-vsctl") else None,
        },
        "ovs_ofctl": {
            "path": shutil.which("ovs-ofctl"),
            "version": _version(["ovs-ofctl", "--version"]) if shutil.which("ovs-ofctl") else None,
        },
        "ryu_manager": {
            "path": shutil.which("ryu-manager"),
            "python_import": importlib.util.find_spec("ryu") is not None,
            "python_module": _version([sys.executable, "-m", "ryu.cmd.manager", "--version"])
            if importlib.util.find_spec("ryu") is not None
            else None,
        },
    }
    missing: list[str] = []
    if not status["is_root"]:
        missing.append("root privileges")
    if not status["mininet"]["path"]:
        missing.append("Mininet executable 'mn'")
    if not status["ovs_vsctl"]["path"]:
        missing.append("Open vSwitch executable 'ovs-vsctl'")
    if not status["ovs_ofctl"]["path"]:
        missing.append("OpenFlow executable 'ovs-ofctl'")
    if not status["ryu_manager"]["python_import"]:
        missing.append("Python module 'ryu'")
    status["can_run_live_demo"] = not missing
    status["missing_items"] = missing
    if missing:
        status["next_steps"] = [
            "Run from a root shell or prefix the live command with sudo.",
            "Install Mininet/Open vSwitch: sudo apt install mininet openvswitch-switch tcpreplay.",
            "Install Ryu in the Python environment used by sudo, preferably Python 3.10/3.11: pip install ryu.",
            "Verify: sudo python3 -c \"import ryu; print('ryu import ok')\" && which ryu-manager.",
        ]
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["can_run_live_demo"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
