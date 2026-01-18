from __future__ import annotations
import os
<<<<<<< HEAD
import pyarrow as pa
import pyarrow.parquet as pq
=======
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
import pandas as pd
from typing import Iterable, Dict, List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rows_to_parquet(rows: Iterable[Dict], out_path: str, chunk_size: int = 100_000) -> None:
    """
    Stream rows (dicts) into a Parquet file without holding all in RAM.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
<<<<<<< HEAD
    writer = None
    batch: List[Dict] = []
=======

    # Check preferred engine
    engine_pref = os.environ.get("DOSDET_PARQUET_ENGINE", "fastparquet").lower()
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

>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
    try:
        for r in rows:
            batch.append(r)
            if len(batch) >= chunk_size:
<<<<<<< HEAD
                table = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                batch.clear()
        if batch:
            table = pa.Table.from_pandas(pd.DataFrame(batch), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
=======
                writer = _flush(batch, writer)
                batch.clear()
        if batch:
            writer = _flush(batch, writer)
    finally:
        if writer is not None and hasattr(writer, "close"):
>>>>>>> b68ee83a7fee0eedac05e6edce1d1c740b008aa7
            writer.close()
