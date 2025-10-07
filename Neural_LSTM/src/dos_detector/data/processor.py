from __future__ import annotations
import gc, json, os, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..config.types import Config
from ..features.feature_engineering import FeatureExtractor
from ..utils.io import ensure_dir, save_dataframe, save_json
from .labels import load_attack_intervals, label_windows
from .pcap_reader import read_pcap, summarize_packets
from .structures import Window
from .windowing import WindowBuilder, WindowingParams
from ..utils.progress import progress
from collections import Counter

class FeaturePipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.extractor = FeatureExtractor(config.feature, config.windowing.window_size)
        self._last_windows: List[Window] = []
        self._last_host_maps: Dict[str, Dict[int, Dict[str, int]]] = {"macs": {}, "ips": {}}

    def process_single(self, pcap_path: Path) -> Tuple[pd.DataFrame, object]:
        limit_env = os.getenv("DOS_LIMIT_PKTS")
        limit = int(limit_env) if (limit_env and limit_env.isdigit()) else None

        t0 = time.perf_counter()
        packets = read_pcap(pcap_path, limit=limit)
        t1 = time.perf_counter()

        builder = WindowBuilder(WindowingParams(
            window_size=self.config.windowing.window_size,
            hop_size=self.config.windowing.hop_size,
            max_windows=self.config.windowing.max_windows,
        ))
        windows = builder.build(packets)
        self._last_windows = list(windows)
        self._last_host_maps = self._build_host_maps(self._last_windows)
        t2 = time.perf_counter()

        frame = self.extractor.extract(windows)
        frame.insert(0, "pcap", pcap_path.name)
        t3 = time.perf_counter()

        intervals_csv = self.config.labels.intervals_csv
        if intervals_csv and Path(intervals_csv).exists():
            intervals_map = load_attack_intervals(Path(intervals_csv), self.config.labels)
            intervals = intervals_map.get(pcap_path.name, [])
            labs = label_windows(windows, intervals, self.config.labels)
            frame["attack"] = [x.attack for x in labs]
            frame["family"] = [x.family for x in labs]
        else:
            frame["attack"] = 0
            frame["family"] = self.config.labels.default_family

        fmap = self.config.labels.family_mapping
        frame["family_index"] = frame["family"].map(lambda f: fmap.get(str(f).lower(), 0)).astype("int64")
        t4 = time.perf_counter()

        # lightweight, explicit stage logs
        print(f"[{pcap_path.name}] read={t1-t0:.1f}s  windows={t2-t1:.1f}s  features={t3-t2:.1f}s  label={t4-t3:.1f}s  rows={len(frame)}", flush=True)
        return frame, summarize_packets(packets, pcap_path)

    def last_windows(self) -> List[Window]:
        """Return the windows generated during the most recent processing call."""

        return list(self._last_windows)

    def last_host_maps(self) -> Dict[str, Dict[int, Dict[str, int]]]:
        """Return per-window MAC/IP counts captured during the latest processing step."""

        return {
            "macs": {int(idx): dict(counts) for idx, counts in self._last_host_maps.get("macs", {}).items()},
            "ips": {int(idx): dict(counts) for idx, counts in self._last_host_maps.get("ips", {}).items()},
        }

    def save_last_host_maps(self, pcap_path: Path, directory: Path | None = None) -> None:
        """Persist the most recent host maps to disk for reuse during inference."""

        cache_path = self._host_cache_path(pcap_path, directory)
        if cache_path is None:
            return
        ensure_dir(cache_path.parent)
        payload = {
            "macs": {str(idx): {mac: int(count) for mac, count in counts.items()} for idx, counts in self._last_host_maps.get("macs", {}).items()},
            "ips": {str(idx): {ip: int(count) for ip, count in counts.items()} for idx, counts in self._last_host_maps.get("ips", {}).items()},
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")

    def load_host_maps(self, pcap_path: Path, directory: Path | None = None) -> Dict[str, Dict[int, Dict[str, int]]] | None:
        """Load cached host maps if they exist."""

        cache_path = self._host_cache_path(pcap_path, directory)
        if cache_path is None or not cache_path.exists():
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return {
            "macs": {int(idx): {mac: int(count) for mac, count in counts.items()} for idx, counts in data.get("macs", {}).items()},
            "ips": {int(idx): {ip: int(count) for ip, count in counts.items()} for idx, counts in data.get("ips", {}).items()},
        }

    def process_files(self, pcaps: Iterable[Path], out_dir: Path) -> Dict[str, object]:
        ensure_dir(out_dir)
        paths = [Path(p) for p in pcaps]

        frames_meta: List[Dict[str, object]] = []
        feature_cols: List[str] = []

        for idx, p in enumerate(progress(paths, desc="Extracting", unit="pcap"), 1):
            try:
                print(f"--> ({idx}/{len(paths)}) {p.name}: start", flush=True)
                df, meta = self.process_single(p)

                if not feature_cols:
                    feature_cols = [c for c in df.columns if c not in {
                        "pcap","window_index","window_start","window_end","attack","family","family_index"
                    }]

                save_dataframe(out_dir / f"{p.stem}.parquet", df)
                self.save_last_host_maps(p, out_dir)
                frames_meta.append({
                    "pcap": p.name,
                    "rows": int(len(df)),
                    "packet_count": int(meta.packet_count),
                    "duration": float(meta.duration),
                })
                print(f"<-- ({idx}/{len(paths)}) {p.name}: saved ({len(df)} rows)", flush=True)
                gc.collect()

            except Exception as e:
                print(f"[SKIP] {p.name}: {e}", flush=True)
                continue

        save_json(self.config.paths.manifest_path, {"feature_columns": feature_cols, "frames": frames_meta})
        return {"feature_columns": feature_cols, "frames": frames_meta}

    def _build_host_maps(self, windows: Iterable[Window]) -> Dict[str, Dict[int, Dict[str, int]]]:
        macs: Dict[int, Dict[str, int]] = {}
        ips: Dict[int, Dict[str, int]] = {}
        for window in windows:
            mac_counts = Counter(pkt.src_mac for pkt in window.packets if pkt.src_mac)
            ip_counts = Counter(pkt.src_ip for pkt in window.packets if pkt.src_ip)
            if mac_counts:
                macs[int(window.index)] = {mac: int(count) for mac, count in mac_counts.items()}
            if ip_counts:
                ips[int(window.index)] = {ip: int(count) for ip, count in ip_counts.items()}
        return {"macs": macs, "ips": ips}

    def _host_cache_path(self, pcap_path: Path, directory: Path | None = None) -> Path | None:
        base = directory if directory is not None else getattr(self.config.paths, "processed_dir", None)
        if base is None:
            return None
        return Path(base) / f"{pcap_path.stem}_hosts.json"
