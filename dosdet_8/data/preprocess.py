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

import fastparquet

from data.pcap_reader import iter_rows_from_pcap
from data.windowizer import iter_windows
from features.seq_features import compute_sequence_features
from features.static_features import compute_static_features

def ensure_dir(p): os.makedirs(p, exist_ok=True)

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

def preprocess(cfg: dict, pcaps_glob, labels_csv: str):
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

    if isinstance(pcaps_glob, str):
        patterns = [pcaps_glob]
    else:
        patterns = list(pcaps_glob)
    files_set = set()
    for pat in patterns:
        for f in glob.glob(pat):
            files_set.add(f)
    files = sorted(files_set)
    assert files, f"No pcaps matched patterns: {patterns}"

    # Streaming state
    shard_id = 0
    shard_path = None
    bytes_in_buffer = 0

    # Batch buffers (to keep memory bounded)
    BATCH_ROWS = 5000
    buf = {k: [] for k in ["file","t0","t1","y","fam","M","K_seq","K_static","seq","static"]}

    def _estimate_row_bytes(seq_list, static_list) -> int:
        return (len(seq_list) + len(static_list)) * 4 + 256  # float32 bytes rough estimate

    def _flush_shard():
        nonlocal shard_id, shard_path, bytes_in_buffer
        if not buf["file"]:
            return
        shard_path = os.path.join(cache_dir, f"shard_{shard_id:05d}.parquet")
        shard_id += 1
        df = pd.DataFrame(buf)
        df.to_parquet(
            shard_path,
            engine="fastparquet",
            compression="zstd",
            index=False,
        )
        manifest["files"].append({"path": shard_path})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        for k in buf:
            buf[k].clear()
        bytes_in_buffer = 0

    # Build manifest incrementally (robust to kill -9)
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = {"schema": {"M": M, "seq_feature_count": None}, "files": []}

    # Per-file loop
    for p in tqdm(files, desc="PCAP files", unit="file"):
        base = os.path.basename(p)
        byte_limit = cfg.get("preprocess", {}).get("byte_limit", None)
        try:
            windows = iter_windows(iter_rows_from_pcap(p, ssdp_v4, ssdp_v6, byte_limit=byte_limit), W, S, M)
        except (FileNotFoundError, PermissionError) as e:
            print(f"[WARN] Skipping unreadable file {p}: {e}")
            continue
            
        limit = int(cfg.get("preprocess", {}).get("limit", 0))
        total_windows = 0
        
        pbar = tqdm(windows, desc=f"Windows: {base}", unit="win", leave=False)
        for (t0, t1, win_rows, bins) in pbar:
            if limit > 0:
                pbar.set_postfix(valid=f"{total_windows}/{limit}")
            if limit > 0 and total_windows >= limit:
                 break
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

            total_windows += 1

            # Flush by batch size
            if len(buf["file"]) >= BATCH_ROWS:
                _flush_batch()

            # Rotate shard if size exceeds limit
            if bytes_written_in_shard >= shard_max_mb * 1024 * 1024:
                # finalize current shard
                if shard_writer and hasattr(shard_writer, "close"):
                    shard_writer.close()
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
    if shard_writer and hasattr(shard_writer, "close"):
        shard_writer.close()
        manifest["files"].append({"path": shard_path})
    elif not use_pyarrow and bytes_written_in_shard > 0:
         # fastparquet/csv case: file is already written, just need to record it
         manifest["files"].append({"path": shard_path})
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest['files'])} shard(s) to {cache_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument(
        "--pcaps",
        nargs="+",
        default=None,
        help="Optional glob override(s); pass one or more patterns. Defaults to config preprocess.pcaps_glob",
    )
    ap.add_argument("--labels", default=None, help="Optional labels.csv override; defaults to config preprocess.labels_csv")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of windows to process per file")
    ap.add_argument("--limit-mb", type=float, default=0, help="Limit processing to N megabytes of data per file")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.limit > 0: 
        print(f"[INFO] Limiting to {args.limit} windows per file.")
        cfg["preprocess"]["limit"] = args.limit

    if args.limit_mb > 0:
        print(f"[INFO] Limiting to {args.limit_mb} MB per file.")
        cfg["preprocess"]["byte_limit"] = int(args.limit_mb * 1024 * 1024)
    
    pcaps = args.pcaps or cfg["preprocess"]["pcaps_glob"]
    labels = args.labels or cfg["preprocess"]["labels_csv"]
    preprocess(cfg, pcaps, labels)

if __name__ == "__main__":
    main()
