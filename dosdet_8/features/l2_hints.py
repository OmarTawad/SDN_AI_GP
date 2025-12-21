from __future__ import annotations
from typing import Dict, List
from collections import Counter

def mac_dominance(rows: List[Dict]) -> float:
    cnt = Counter([r.get("src_mac") for r in rows if r.get("src_mac")])
    total = sum(cnt.values())
    return (max(cnt.values())/total) if total>0 else 0.0
