from __future__ import annotations
from typing import Optional, Tuple, Dict

# SSDP constants
SSDP_METHODS = ("M-SEARCH", "NOTIFY")
SSDP_RESPONSE_PREFIX = "HTTP/1.1 200 OK"

def parse_ssdp_payload(raw_bytes: bytes, max_len: int = 2048) -> Tuple[str, Dict[str, str]]:
    """
    Return (method, tokens) where method ∈ {"M-SEARCH","NOTIFY","200-OK","OTHER","NONE"}.
    tokens may include ST, MAN, USER-AGENT (upper-cased keys).
    """
    if not raw_bytes:
        return "NONE", {}
    try:
        text = raw_bytes[:max_len].decode("utf-8", errors="ignore")
    except Exception:
        return "OTHER", {}
    if not text:
        return "NONE", {}

    first = text.split("\r\n", 1)[0].strip().upper()
    if first.startswith(SSDP_RESPONSE_PREFIX):
        method = "200-OK"
    elif any(first.startswith(m) for m in SSDP_METHODS):
        method = "M-SEARCH" if first.startswith("M-SEARCH") else "NOTIFY"
    else:
        method = "OTHER"

    tokens = {}
    # Simple header parse
    for line in text.split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip().upper()
            v = v.strip()
            if k in ("ST", "MAN", "USER-AGENT"):
                tokens[k] = v
    return method, tokens
