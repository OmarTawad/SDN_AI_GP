from __future__ import annotations
from typing import Iterable, Optional
from tqdm import tqdm

def progress(it: Iterable,
             desc: str = "",
             unit: str = "it",
             total: Optional[int] = None,
             leave: bool = True):
    return tqdm(
        it, desc=desc, unit=unit, total=total, leave=leave,
        dynamic_ncols=True, smoothing=0.1, miniters=1, mininterval=0.1,
        bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} • {elapsed} < {remaining} • {rate_fmt}"
    )
