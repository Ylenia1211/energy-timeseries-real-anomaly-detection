from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatrixProfileConfig:
    subseq_len: int
    top_k: int = 10
    min_gap: int | None = None
    zscore_threshold: float = 2.5


def compute_matrix_profile(signal: np.ndarray, subseq_len: int) -> pd.DataFrame:
    """Compute the Matrix Profile using STUMPY.

    STUMPY is an optional dependency because some environments cannot install it quickly.
    Install the complete project with ``pip install -e '.[all]'`` before using this function.
    """
    try:
        import stumpy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Matrix Profile support requires stumpy. Install with: pip install stumpy") from exc

    x = np.asarray(signal, dtype=float).reshape(-1)
    if subseq_len < 4:
        raise ValueError("subseq_len should be >= 4")
    if len(x) <= subseq_len:
        raise ValueError("signal length must be > subseq_len")
    mp = stumpy.stump(x, m=subseq_len)
    return pd.DataFrame(
        {
            "start_index": np.arange(len(mp), dtype=int),
            "matrix_profile": mp[:, 0].astype(float),
            "nearest_neighbor_index": mp[:, 1].astype(int),
        }
    )


def find_discords(mp_df: pd.DataFrame, config: MatrixProfileConfig) -> pd.DataFrame:
    df = mp_df.copy()
    profile = df["matrix_profile"].to_numpy(dtype=float)
    median = np.nanmedian(profile)
    mad = np.nanmedian(np.abs(profile - median)) + 1e-12
    df["robust_zscore"] = 0.6745 * (profile - median) / mad
    candidates = df[df["robust_zscore"] >= config.zscore_threshold].sort_values(
        "matrix_profile", ascending=False
    )

    min_gap = config.min_gap or config.subseq_len
    selected: list[int] = []
    rows = []
    for _, row in candidates.iterrows():
        idx = int(row["start_index"])
        if all(abs(idx - s) >= min_gap for s in selected):
            selected.append(idx)
            rows.append(row)
        if len(rows) >= config.top_k:
            break
    if not rows:
        return pd.DataFrame(columns=list(df.columns))
    return pd.DataFrame(rows).sort_values("start_index").reset_index(drop=True)


def group_discords(discords: pd.DataFrame, gap: int) -> pd.DataFrame:
    if discords.empty:
        return pd.DataFrame(columns=["onset_index", "end_index", "n_windows", "max_profile"])
    starts = discords["start_index"].astype(int).to_numpy()
    profiles = discords["matrix_profile"].astype(float).to_numpy()
    groups: list[dict[str, float | int]] = []
    current = [starts[0]]
    current_profiles = [profiles[0]]
    for s, p in zip(starts[1:], profiles[1:]):
        if s - current[-1] <= gap:
            current.append(int(s))
            current_profiles.append(float(p))
        else:
            groups.append(
                {
                    "onset_index": int(current[0]),
                    "end_index": int(current[-1] + gap),
                    "n_windows": len(current),
                    "max_profile": float(np.max(current_profiles)),
                }
            )
            current = [int(s)]
            current_profiles = [float(p)]
    groups.append(
        {
            "onset_index": int(current[0]),
            "end_index": int(current[-1] + gap),
            "n_windows": len(current),
            "max_profile": float(np.max(current_profiles)),
        }
    )
    return pd.DataFrame(groups)
