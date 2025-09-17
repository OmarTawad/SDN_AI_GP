from __future__ import annotations
import gc, os, time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from ..config.types import Config
from ..features.feature_engineering import FeatureExtractor
from ..utils.io import ensure_dir, save_dataframe, save_json
from .labels import load_attack_intervals, label_windows
from .pcap_reader import read_pcap, summarize_packets
from .windowing import WindowBuilder, WindowingParams
from ..utils.progress import progress

class FeaturePipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.extractor = FeatureExtractor(config.feature, config.windowing.window_size)

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
