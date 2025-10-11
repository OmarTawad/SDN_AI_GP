#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib

from arp_detector.config import load_config
from arp_detector.utils.io import load_dataframe, load_json, ensure_dir
from arp_detector.models.supervised import SequenceClassifier

torch.set_num_threads(2)

def per_file_scores(df: pd.DataFrame, feat_cols, scaler, model, seq_len: int):
    """Return (window_index[], window_prob[]). Works even if len(df) < seq_len."""
    if df.empty or not feat_cols:
        return np.array([], dtype=int), np.array([], dtype=float)

    X = df[feat_cols].to_numpy(dtype=np.float32)
    X = scaler.transform(X)

    # Stable mapping index->time
    df = df.drop_duplicates(subset=["window_index"])
    widx = df["window_index"].astype(int).to_numpy()

    with torch.no_grad():
        if len(X) < seq_len:
            # pad by repeating last frame so short pcaps still score
            pad = np.repeat(X[-1:], seq_len - len(X), axis=0)
            Xb = np.concatenate([X, pad], axis=0)[None, :, :]
            logits = model(torch.from_numpy(Xb)).window_logits.numpy()[0, :len(X)]
            probs = 1/(1+np.exp(-logits))
            return widx, probs.astype(float)

        # non-overlapping chunks of length seq_len (fast & simple)
        chunks, flat_idx = [], []
        for s in range(0, len(X) - seq_len + 1, seq_len):
            chunks.append(X[s:s+seq_len])
            flat_idx.extend(widx[s:s+seq_len])
        Xb = np.stack(chunks)  # [B,seq_len,F]
        logits = model(torch.from_numpy(Xb)).window_logits.numpy().reshape(-1)
        probs = 1/(1+np.exp(-logits))
        return np.array(flat_idx[:len(probs)], dtype=int), probs.astype(float)

def main():
    ap = argparse.ArgumentParser(description="Score a folder of pcaps using the trained model.")
    ap.add_argument("--src", required=True, help="Folder containing *.pcap / *.pcapng")
    ap.add_argument("--suffix", default="samples", help="Suffix for report filenames")
    ap.add_argument("--override-tau-file", type=float, default=None, help="Override tau_file")
    ap.add_argument("--override-tau-window", type=float, default=None, help="Override tau_window")
    args = ap.parse_args()

    cfg = load_config("configs/config.yaml")
    man = load_json(cfg.paths.manifest_path)
    feat_cols = man.get("feature_columns", [])
    seq_len   = int(cfg.windowing.sequence_length)
    tau_file  = float(args.override_tau_file if args.override_tau_file is not None else cfg.postprocessing.tau_file)
    tau_win   = float(args.override_tau_window if args.override_tau_window is not None else getattr(cfg.postprocessing, "tau_window", tau_file))
    min_run   = int(getattr(cfg.postprocessing, "min_attack_windows", 2))

    ensure_dir(cfg.paths.reports_dir)
    src = Path(args.src)
    pcaps = sorted(list(src.glob("*.pcap")) + list(src.glob("*.pcapng")))
    if not pcaps:
        print(f"[err] no pcaps in {src}", file=sys.stderr)
        sys.exit(2)

    # Load scaler + model
    scaler = joblib.load(cfg.paths.scaler_path)
    model  = SequenceClassifier(len(feat_cols), len(cfg.labels.family_mapping), cfg.model.supervised)
    state  = torch.load(cfg.paths.supervised_model_path, map_location="cpu")
    model.load_state_dict(state); model.eval()

    file_rows, interval_rows = [], []
    for pcap in pcaps:
        pq = cfg.paths.processed_dir / f"{pcap.stem}.parquet"
        if not pq.exists():
            print(f"[skip] features missing: {pq}")
            continue
        df = load_dataframe(pq)
        wi, pr = per_file_scores(df, feat_cols, scaler, model, seq_len)
        if pr.size == 0:
            file_rows.append({"pcap": pcap.name, "file_score": 0.0, "pred": 0})
            continue
        file_score = float(pr.max())
        file_rows.append({"pcap": pcap.name, "file_score": file_score, "pred": int(file_score >= tau_file)})

        # intervals above tau_win
        idx2start = dict(zip(df["window_index"].astype(int), df["window_start"].astype(float)))
        idx2end   = dict(zip(df["window_index"].astype(int), df["window_end"].astype(float)))
        start = None
        for i, p in enumerate(pr):
            if p >= tau_win and start is None:
                start = i
            end_cond = (p < tau_win) or (i == len(pr)-1)
            if start is not None and end_cond:
                end = i if p < tau_win else i
                if end - start + 1 >= min_run:
                    w_start, w_end = int(wi[start]), int(wi[end])
                    interval_rows.append({
                        "pcap": pcap.name,
                        "start_time": float(idx2start.get(w_start, df["window_start"].min())),
                        "end_time":   float(idx2end.get(w_end,   df["window_end"].max())),
                        "max_window_score": float(pr[start:end+1].max()),
                    })
                start = None if p < tau_win else start

    # write reports
    summary = pd.DataFrame(file_rows).sort_values("pcap")
    intervals = pd.DataFrame(interval_rows).sort_values(["pcap","start_time"]) if interval_rows else pd.DataFrame(columns=["pcap","start_time","end_time","max_window_score"])

    out1 = cfg.paths.reports_dir / f"inference_summary_{args.suffix}.csv"
    out2 = cfg.paths.reports_dir / f"predicted_intervals_{args.suffix}.csv"
    summary.to_csv(out1, index=False)
    intervals.to_csv(out2, index=False)
    print("[ok] wrote:", out1)
    print(summary.to_string(index=False))
    print("[ok] wrote:", out2)
    if intervals.empty:
        print("[info] no intervals over tau_window")

if __name__ == "__main__":
    main()
