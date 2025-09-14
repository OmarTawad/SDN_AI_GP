from __future__ import annotations
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import Iterable, Dict, List

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def rows_to_parquet(rows: Iterable[Dict], out_path: str, chunk_size: int = 100_000) -> None:
    """
    Stream rows (dicts) into a Parquet file without holding all in RAM.
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    writer = None
    batch: List[Dict] = []
    try:
        for r in rows:
            batch.append(r)
            if len(batch) >= chunk_size:
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
            writer.close()
