"""Label utilities shared across gateway components."""

from __future__ import annotations

from typing import Dict

CLASS_ID_TO_NAME: Dict[int, str] = {0: "normal", 1: "dos", 2: "arp"}
CLASS_NAME_TO_ID: Dict[str, int] = {
    "0": 0,
    "normal": 0,
    "benign": 0,
    "background": 0,
    "1": 1,
    "dos": 1,
    "dos_attack": 1,
    "flood": 1,
    "attack_dos": 1,
    "2": 2,
    "arp": 2,
    "arp_spoof": 2,
    "spoof": 2,
    "poison": 2,
    "attack_arp": 2,
}


def class_id_to_name(class_id: int) -> str:
    """Return the textual label for a class identifier.

    Args:
        class_id: Numeric label.

    Returns:
        str: Label name; falls back to ``class_{id}`` if unknown.
    """

    return CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")


__all__ = ["CLASS_ID_TO_NAME", "CLASS_NAME_TO_ID", "class_id_to_name"]
