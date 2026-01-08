from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def replace_invalid(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def clip_quantiles(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if not 0.0 <= lower < upper <= 1.0:
        return df
    lower_bounds = df.quantile(lower)
    upper_bounds = df.quantile(upper)
    return df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)


def select_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected features: {missing}")
    return df[list(features)]


def apply_log_transform(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return df
    df = df.copy()
    for column in columns:
        if column in df.columns:
            values = df[column].clip(lower=0.0)
            df[column] = np.log1p(values)
    return df


def scaler_from_name(name: str):
    name_lower = name.lower()
    if name_lower == "robustscaler" or name_lower == "robust":
        return RobustScaler()
    if name_lower == "standardscaler" or name_lower == "standard":
        return StandardScaler()
    if name_lower == "minmaxscaler" or name_lower == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def fit_scaler(data: pd.DataFrame, scaler_name: str) -> Tuple[object, List[str]]:
    feats = list(data.columns)
    scaler = scaler_from_name(scaler_name)
    scaler.fit(data.values)
    return scaler, feats


def transform_with_scaler(data: pd.DataFrame, scaler) -> np.ndarray:
    return scaler.transform(data.values)


def save_scaler(scaler, path: Path | str) -> None:
    joblib.dump(scaler, Path(path))


def load_scaler(path: Path | str):
    return joblib.load(Path(path))


def save_feature_list(features: Sequence[str], path: Path | str) -> None:
    Path(path).write_text(json.dumps(list(features), indent=2), encoding="utf-8")


def load_feature_list(path: Path | str) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
