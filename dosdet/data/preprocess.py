from __future__ import annotations
import argparse, glob, json, os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("PYARROW_NUM_THREADS", "2")

from typing import Dict, List
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _pick_parquet_engine() -> str:
    override = os.environ.get("DOSDET_PARQUET_ENGINE", "").strip().lower()
    candidates = [override] if override else []
    candidates.extend(["fastparquet", "pyarrow"])
    seen = set()
    for engine in candidates:
        if not engine or engine in seen:
            continue
        seen.add(engine)
        if engine == "fastparquet":
            try:
                import fastparquet  # noqa: F401
                return engine
            except Exception:
                continue
        if engine == "pyarrow":
            try:
                import pyarrow  # noqa: F401
                return engine
            except Exception:
                continue
    tried = ", ".join(seen) or "fastparquet, pyarrow"
    raise RuntimeError(f"No usable parquet engine found. Tried: {tried}. Install fastparquet or pyarrow.")

def _labels_map(labels_csv: str):
    labs = pd.read_csv(labels_csv).fillna("")
    labs["base"] = labs["file"].apply(lambda p: os.path.basename(p))
    labs["attack"] = labs["attack_label"].astype(int)
    # fix FutureWarning: use astype to get int64 epoch seconds
    ts = pd.to_datetime(labs["attack_start_iso"], utc=True, errors="coerce").astype("int64") / 1e9
    labs["start"] = ts
    by_base = {}
    for _, r in labs.iterrows():
        by_base[r["base"]] = (int(r["attack"]), float(r["start"]) if pd.notna(r["start"]) else np.nan)
    return by_base

def _label_for(meta_map: Dict[str, tuple], base: str, t0: float):
    if base not in meta_map: return 0, -1
    is_atk, start = meta_map[base]
    if is_atk == 0: return 0, -1
    if np.isnan(start) or t0 >= start: return 1, 0
    return 0, -1

def preprocess(cfg: dict, pcaps_glob: str, labels_csv: str):
    cache_dir = cfg["paths"]["cache_dir"]
    ensure_dir(cache_dir)
    shard_max_mb = int(cfg["preprocess"]["shard_max_mb"])
    W = float(cfg["windowing"]["window_sec"])
    S = float(cfg["windowing"]["stride_sec"])
    M = int(cfg["windowing"]["micro_bins"])
    top_ports = list(cfg["data"]["top_k_udp_ports"])
    ssdp_v4 = cfg["features"]["ssdp_multicast_ipv4"]
    ssdp_v6 = cfg["features"]["ssdp_multicast_ipv6"]

    label_map = _labels_map(labels_csv)

    files = sorted(glob.glob(pcaps_glob))
    assert files, f"No pcaps matched: {pcaps_glob}"

    engine = _pick_parquet_engine()
    if engine == "pyarrow":
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = pa.schema([
            ("file", pa.string()),
            ("t0", pa.float64()),
            ("t1", pa.float64()),
            ("y", pa.int32()),
            ("fam", pa.int32()),
            ("M", pa.int32()),
            ("K_seq", pa.int32()),
            ("K_static", pa.int32()),
            ("seq", pa.list_(pa.float32())),
            ("static", pa.list_(pa.float32())),
        ])
    else:
        import fastparquet
        schema = None

    # Streaming state
    shard_id = 0
    shard_writer = None
    shard_path = None
    bytes_written_in_shard = 0
    rows_written_in_shard = 0

    # Batch buffers (to keep memory bounded)
    BATCH_ROWS = 5000
    buf = {k: [] for k in ["file","t0","t1","y","fam","M","K_seq","K_static","seq","static"]}

    def _open_new_shard():
        nonlocal shard_id, shard_writer, shard_path, bytes_written_in_shard, rows_written_in_shard
        shard_path = os.path.join(cache_dir, f"shard_{shard_id:05d}.parquet")
        if engine == "pyarrow":
            shard_writer = pq.ParquetWriter(shard_path, schema, compression="zstd")
        else:
            shard_writer = None
            if os.path.exists(shard_path):
                os.remove(shard_path)
        bytes_written_in_shard = 0
        rows_written_in_shard = 0
        shard_id += 1

    def _flush_batch():
        nonlocal shard_writer, bytes_written_in_shard, rows_written_in_shard
        if not buf["file"]:
            return 0
        if engine == "pyarrow":
            table = pa.table({
                "file": pa.array(buf["file"], type=pa.string()),
                "t0": pa.array(buf["t0"], type=pa.float64()),
                "t1": pa.array(buf["t1"], type=pa.float64()),
                "y": pa.array(buf["y"], type=pa.int32()),
                "fam": pa.array(buf["fam"], type=pa.int32()),
                "M": pa.array(buf["M"], type=pa.int32()),
                "K_seq": pa.array(buf["K_seq"], type=pa.int32()),
                "K_static": pa.array(buf["K_static"], type=pa.int32()),
                "seq": pa.array(buf["seq"], type=pa.list_(pa.float32())),
                "static": pa.array(buf["static"], type=pa.list_(pa.float32())),
            })
            shard_writer.write_table(table)
            est = table.nbytes
            bytes_written_in_shard += est
            rows_written_in_shard += table.num_rows
        else:
            seq_vals = [np.asarray(v, dtype=np.float32).tolist() for v in buf["seq"]]
            static_vals = [np.asarray(v, dtype=np.float32).tolist() for v in buf["static"]]
            df = pd.DataFrame({
                "file": buf["file"],
                "t0": buf["t0"],
                "t1": buf["t1"],
                "y": buf["y"],
                "fam": buf["fam"],
                "M": buf["M"],
                "K_seq": buf["K_seq"],
                "K_static": buf["K_static"],
                "seq": seq_vals,
                "static": static_vals,
            })
            fastparquet.write(
                shard_path,
                df,
                append=os.path.exists(shard_path),
                compression=None,
                object_encoding="json",
            )
            bytes_written_in_shard = os.path.getsize(shard_path)
            rows_written_in_shard += len(df)
            est = bytes_written_in_shard
        # clear buffers
        for k in buf: buf[k].clear()
        return est

    # Open first shard
    _open_new_shard()

    # Build manifest incrementally (robust to kill -9)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = {"schema": {"M": M, "seq_feature_count": None}, "files": []}

    # Per-file loop
    for p in tqdm(files, desc="PCAP files", unit="file"):
        base = os.path.basename(p)
        windows = iter_windows(iter_rows_from_pcap(p, ssdp_v4, ssdp_v6), W, S, M)
        for (t0, t1, win_rows, bins) in tqdm(windows, desc=f"Windows: {base}", unit="win", leave=False):
            if not win_rows:
                continue
            seq_np, extras = compute_sequence_features(win_rows, bins, M, top_ports)
            static_vec, static_names, snaps = compute_static_features(
                win_rows, M, extras["per_bin_total_pkts"], top_ports, W
            )
            y, fam = _label_for(label_map, base, t0)

            # First time we know K_seq: stash it in manifest.schema
            if manifest["schema"]["seq_feature_count"] is None:
                manifest["schema"]["seq_feature_count"] = int(seq_np.shape[1])

            # Append to batch
            buf["file"].append(base)
            buf["t0"].append(float(t0))
            buf["t1"].append(float(t1))
            buf["y"].append(int(y))
            buf["fam"].append(int(fam))
            buf["M"].append(int(M))
            buf["K_seq"].append(int(seq_np.shape[1]))
            buf["K_static"].append(int(static_vec.size))
            buf["seq"].append(seq_np.astype(np.float32).reshape(-1))
            buf["static"].append(static_vec.astype(np.float32))

            # Flush by batch size
            if len(buf["file"]) >= BATCH_ROWS:
                _flush_batch()

            # Rotate shard if size exceeds limit
            if bytes_written_in_shard >= shard_max_mb * 1024 * 1024:
                # finalize current shard
                if engine == "pyarrow" and shard_writer is not None:
                    shard_writer.close()
                if rows_written_in_shard > 0:
                    manifest["files"].append({"path": shard_path})
                    # persist manifest incrementally (crash-safe)
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                # open next shard
                _open_new_shard()

        # end file loop

    # Flush any trailing rows
    if buf["file"]:
        _flush_batch()
    # Close last shard
    if engine == "pyarrow" and shard_writer is not None:
        shard_writer.close()
    if rows_written_in_shard > 0:
        manifest["files"].append({"path": shard_path})

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest['files'])} shard(s) to {cache_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--pcaps", default=None, help="Optional glob override; defaults to config preprocess.pcaps_glob")
    ap.add_argument("--labels", default=None, help="Optional labels.csv override; defaults to config preprocess.labels_csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pcaps = args.pcaps or cfg["preprocess"]["pcaps_glob"]
    labels = args.labels or cfg["preprocess"]["labels_csv"]
    preprocess(cfg, pcaps, labels)

if __name__ == "__main__":
    main()
