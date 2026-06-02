from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .detectors import DetectorConfig, fit_detector, score_detector
from .features import extract_features
from .matrix_profile import MatrixProfileConfig, compute_matrix_profile, find_discords, group_discords
from .windowing import WindowConfig, sliding_windows


@dataclass(frozen=True)
class FeaturePipelineConfig:
    sampling_rate: float
    window_size: int
    overlap: float = 0.5
    bands: tuple[tuple[float, float], ...] = ((0, 10), (10, 50), (50, 200))
    fundamental_freq: float | None = None


def build_feature_table(signal: np.ndarray, config: FeaturePipelineConfig) -> pd.DataFrame:
    windows, starts = sliding_windows(
        signal, WindowConfig(window_size=config.window_size, overlap=config.overlap)
    )
    features = extract_features(
        windows,
        sampling_rate=config.sampling_rate,
        bands=list(config.bands),
        fundamental_freq=config.fundamental_freq,
    )
    features.insert(0, "window_start", starts)
    features.insert(1, "window_end", starts + config.window_size)
    return features


def train_and_score(signal: np.ndarray, cfg: FeaturePipelineConfig, detector_cfg: DetectorConfig):
    feature_table = build_feature_table(signal, cfg)
    feature_cols = [c for c in feature_table.columns if not c.startswith("window_")]
    model = fit_detector(feature_table[feature_cols], detector_cfg)
    scores = score_detector(model, feature_table[feature_cols])
    return model, pd.concat([feature_table[["window_start", "window_end"]], scores, feature_table[feature_cols]], axis=1)


def matrix_profile_onsets(signal: np.ndarray, subseq_len: int, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mp_cfg = MatrixProfileConfig(subseq_len=subseq_len, top_k=top_k)
    mp = compute_matrix_profile(signal, subseq_len)
    discords = find_discords(mp, mp_cfg)
    onset_groups = group_discords(discords, gap=subseq_len)
    return mp, discords, onset_groups
