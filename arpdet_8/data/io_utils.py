from __future__ import annotations
import os
import pandas as pd
from typing import Iterable, Dict, List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rows_to_parquet(rows: Iterable[Dict], out_path: str, chunk_size: int = 100_000) -> None:
    """
    Stream rows (dicts) into a Parquet file without holding all in RAM.
    """
    ensure_dir(os.path.dirname(out_path) or ".")

    # Check preferred engine
    engine_pref = os.environ.get("ARPDET_PARQUET_ENGINE", "fastparquet").lower()
    use_pyarrow = engine_pref == "pyarrow"
    
    writer = None
    batch: List[Dict] = []

    # helper to flush batch
    def _flush(b: List[Dict], w):
        if not b: return w
        df = pd.DataFrame(b)
        if use_pyarrow:
             # Lazy import
             import pyarrow as pa
             import pyarrow.parquet as pq
             table = pa.Table.from_pandas(df, preserve_index=False)
             if w is None:
                 w = pq.ParquetWriter(out_path, table.schema)
             w.write_table(table)
             return w
        else:
            # fastparquet / auto
            append = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            df.to_parquet(
                out_path, 
                index=False, 
                engine="fastparquet", 
                append=append
            )
            return None

    try:
        for r in rows:
            batch.append(r)
            if len(batch) >= chunk_size:
                writer = _flush(batch, writer)
                batch.clear()
        if batch:
            writer = _flush(batch, writer)
    finally:
        if writer is not None and hasattr(writer, "close"):
            writer.close()
