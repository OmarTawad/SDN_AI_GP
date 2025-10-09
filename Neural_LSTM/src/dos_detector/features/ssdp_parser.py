"""Utilities for parsing SSDP payloads."""

from __future__ import annotations

from typing import Dict, Tuple

SSDP_METHODS = ("M-SEARCH", "NOTIFY")
SSDP_RESPONSE_PREFIX = "HTTP/1.1 200 OK"


def parse_ssdp_payload(raw_bytes: bytes, max_len: int = 2048) -> Tuple[str, Dict[str, str]]:
    """Parse SSDP control payloads into a method label and selected headers."""

    if not raw_bytes:
        return "NONE", {}
    try:
        text = raw_bytes[:max_len].decode("utf-8", errors="ignore")
    except Exception:
        return "OTHER", {}
    if not text:
        return "NONE", {}

    first_line = text.split("\r\n", 1)[0].strip().upper()
    if first_line.startswith(SSDP_RESPONSE_PREFIX):
        method = "200-OK"
    elif any(first_line.startswith(m) for m in SSDP_METHODS):
        method = "M-SEARCH" if first_line.startswith("M-SEARCH") else "NOTIFY"
    else:
        method = "OTHER"

    tokens: Dict[str, str] = {}
    for line in text.split("\r\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in {"ST", "MAN", "USER-AGENT"}:
            tokens[key] = value
    return method, tokens


__all__ = ["parse_ssdp_payload"]
