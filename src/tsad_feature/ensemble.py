from __future__ import annotations

import numpy as np
import pandas as pd


def robust_minmax(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    lo = values.quantile(0.01)
    hi = values.quantile(0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return ((values.clip(lo, hi) - lo) / (hi - lo)).fillna(0.0)


def window_scores_to_samples(scores: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    out = pd.DataFrame({"sample_index": np.arange(n_samples), "feature_score": 0.0, "feature_votes": 0})
    for _, row in scores.iterrows():
        a = int(row["window_start"])
        b = min(int(row["window_end"]), n_samples)
        score = float(row["anomaly_score"])
        out.loc[a:b - 1, "feature_score"] = np.maximum(out.loc[a:b - 1, "feature_score"], score)
        if bool(row["is_anomaly"]):
            out.loc[a:b - 1, "feature_votes"] += 1
    out["feature_is_anomaly"] = out["feature_votes"] > 0
    return out


def matrix_profile_to_samples(discords: pd.DataFrame, n_samples: int, subseq_len: int) -> pd.DataFrame:
    out = pd.DataFrame({"sample_index": np.arange(n_samples), "mp_score": 0.0, "mp_is_anomaly": False})
    if discords.empty:
        return out
    for _, row in discords.iterrows():
        a = int(row["start_index"])
        b = min(a + subseq_len, n_samples)
        score = float(row.get("robust_zscore", row.get("matrix_profile", 0.0)))
        out.loc[a:b - 1, "mp_score"] = np.maximum(out.loc[a:b - 1, "mp_score"], score)
        out.loc[a:b - 1, "mp_is_anomaly"] = True
    return out


def combine_anomaly_outputs(
    n_samples: int,
    feature_scores: pd.DataFrame | None = None,
    matrix_discords: pd.DataFrame | None = None,
    matrix_subseq_len: int | None = None,
    arima_scores: pd.DataFrame | None = None,
    prophet_scores: pd.DataFrame | None = None,
    min_votes: int = 2,
) -> pd.DataFrame:
    combined = pd.DataFrame({"sample_index": np.arange(n_samples)})
    vote_cols: list[str] = []
    score_cols: list[str] = []

    if feature_scores is not None:
        f = window_scores_to_samples(feature_scores, n_samples)
        combined = combined.merge(f, on="sample_index", how="left")
        combined["feature_score_norm"] = robust_minmax(combined["feature_score"])
        vote_cols.append("feature_is_anomaly")
        score_cols.append("feature_score_norm")

    if matrix_discords is not None and matrix_subseq_len is not None:
        m = matrix_profile_to_samples(matrix_discords, n_samples, matrix_subseq_len)
        combined = combined.merge(m, on="sample_index", how="left")
        combined["mp_score_norm"] = robust_minmax(combined["mp_score"])
        vote_cols.append("mp_is_anomaly")
        score_cols.append("mp_score_norm")

    for name, scores in [("arima", arima_scores), ("prophet", prophet_scores)]:
        if scores is None:
            continue
        s = scores[["sample_index", "anomaly_score", "is_anomaly"]].copy()
        s = s.rename(columns={"anomaly_score": f"{name}_score", "is_anomaly": f"{name}_is_anomaly"})
        combined = combined.merge(s, on="sample_index", how="left")
        combined[f"{name}_score_norm"] = robust_minmax(combined[f"{name}_score"].fillna(0.0))
        vote_cols.append(f"{name}_is_anomaly")
        score_cols.append(f"{name}_score_norm")

    for col in vote_cols:
        combined[col] = combined[col].fillna(False).astype(bool)
    combined["vote_count"] = combined[vote_cols].sum(axis=1) if vote_cols else 0
    combined["ensemble_score"] = combined[score_cols].mean(axis=1) if score_cols else 0.0
    combined["ensemble_is_anomaly"] = combined["vote_count"] >= min_votes
    return combined


def group_sample_anomalies(sample_scores: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    anomalous = sample_scores[sample_scores["ensemble_is_anomaly"]].copy()
    if anomalous.empty:
        return pd.DataFrame(columns=["onset_index", "end_index", "duration_samples", "max_score", "max_votes"])
    idx = anomalous["sample_index"].astype(int).to_numpy()
    groups = []
    start = prev = int(idx[0])
    rows = [anomalous.iloc[0]]
    for i, row in anomalous.iloc[1:].iterrows():
        cur = int(row["sample_index"])
        if cur - prev <= max_gap + 1:
            rows.append(row)
        else:
            g = pd.DataFrame(rows)
            groups.append({
                "onset_index": start,
                "end_index": prev,
                "duration_samples": prev - start + 1,
                "max_score": float(g["ensemble_score"].max()),
                "max_votes": int(g["vote_count"].max()),
            })
            start = cur
            rows = [row]
        prev = cur
    g = pd.DataFrame(rows)
    groups.append({
        "onset_index": start,
        "end_index": prev,
        "duration_samples": prev - start + 1,
        "max_score": float(g["ensemble_score"].max()),
        "max_votes": int(g["vote_count"].max()),
    })
    return pd.DataFrame(groups)
