from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "PhV.phsA", "PhV.phsB", "PhV.phsC", "Vllavg", "Vlnavg",
    "A.phsA", "A.phsB", "A.phsC", "A.neut", "A.avg",
    "TotW", "TotWh.import", "TotWh.export", "ActiveEnergyTot", "TotVAr",
    "ReactiveEnergyTot", "TotVA", "Err", "frequency", "TotPF", "THD_I", "THD_V",
]

@dataclass(frozen=True)
class RealSeriesConfig:
    csv_path: str
    value_col: str = "TotW"
    time_col: str = "datetime"
    meter_id: str | None = None
    model: str | None = None
    building: str | None = None
    resample: str | None = "15min"
    agg: Literal["mean", "sum", "median", "max", "min"] = "mean"
    interpolate: bool = True
    drop_nonpositive: bool = False


def list_available_series(data_dir: str = "data/raw") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path, usecols=lambda c: c in {"building", "id", "model", "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        group_cols = [c for c in ["building", "id", "model"] if c in df.columns]
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"file": path.name, "n_rows": int(len(g)), "start": g["datetime"].min(), "end": g["datetime"].max()}
            row.update(dict(zip(group_cols, keys)))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["file", "id"]).reset_index(drop=True)


def load_real_series(config: RealSeriesConfig) -> pd.DataFrame:
    path = Path(config.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {config.time_col, config.value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)}. Available: {list(df.columns)}")
    if config.meter_id is not None:
        df = df[df["id"] == config.meter_id]
    if config.model is not None:
        df = df[df["model"] == config.model]
    if config.building is not None:
        df = df[df["building"] == config.building]
    if df.empty:
        raise ValueError("No rows after filtering. Check meter_id/model/building.")

    out = df[[config.time_col, config.value_col]].copy()
    out[config.time_col] = pd.to_datetime(out[config.time_col], errors="coerce")
    out[config.value_col] = pd.to_numeric(out[config.value_col], errors="coerce")
    out = out.dropna(subset=[config.time_col]).sort_values(config.time_col)
    out = out.drop_duplicates(subset=[config.time_col], keep="last")
    out = out.set_index(config.time_col).rename(columns={config.value_col: "value"})
    if config.drop_nonpositive:
        out = out[out["value"] > 0]
    if config.resample:
        resampler = out.resample(config.resample)
        out = getattr(resampler, config.agg)()
    if config.interpolate:
        out["value"] = out["value"].interpolate(limit_direction="both")
    out = out.dropna(subset=["value"])
    out.index.name = "timestamp"
    out["sample_index"] = np.arange(len(out), dtype=int)
    return out.reset_index()[["timestamp", "sample_index", "value"]]


def infer_sampling_rate_hz(frame: pd.DataFrame, time_col: str = "timestamp") -> float:
    t = pd.to_datetime(frame[time_col], errors="coerce").dropna().sort_values()
    if len(t) < 2:
        return 1.0
    dt = t.diff().dropna().dt.total_seconds().median()
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    return float(1.0 / dt)
