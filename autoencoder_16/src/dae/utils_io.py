from __future__ import annotations

import glob
from pathlib import Path
from typing import Generator, Iterable, Iterator, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def iter_pcap_files(patterns: Iterable[str]) -> Iterator[Path]:
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            yield Path(path)


class ParquetBatchWriter:
    """Append batches to a Parquet file without loading everything into memory."""

    def __init__(self, output_path: Path, schema: pa.schema | None = None) -> None:
        self.output_path = Path(output_path)
        self.writer: pq.ParquetWriter | None = None
        self.schema = schema

    def write(self, rows: List[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.output_path, table.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_parquet_batches(path: Path | str, batch_size: int) -> Generator[pd.DataFrame, None, None]:
    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()
