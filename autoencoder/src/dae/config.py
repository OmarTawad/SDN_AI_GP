from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


@dataclass
class Config:
    """Container for application configuration."""

    data: Dict[str, Any]
    path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.data
        for key in keys:
            if not isinstance(cursor, dict):
                return default
            cursor = cursor.get(key, default)
        return cursor

    def as_dict(self) -> Dict[str, Any]:
        return self.data

    def hash(self) -> str:
        canonical = json.dumps(self.data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_config(path: Path | str) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data=data, path=config_path)


def ensure_directories(config: Config, keys: Iterable[Iterable[str]]) -> None:
    for key_path in keys:
        path_value = config.get(*key_path)
        if path_value is None:
            continue
        Path(path_value).mkdir(parents=True, exist_ok=True)
