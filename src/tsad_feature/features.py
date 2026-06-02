from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

_EPS = 1e-12


def _safe_entropy(probabilities: np.ndarray) -> float:
    p = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def time_domain_features(window: np.ndarray) -> dict[str, float]:
    x = np.asarray(window, dtype=float)
    centered = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak_abs = float(np.max(np.abs(x)))
    signs = np.signbit(centered)
    zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1])) if len(x) > 1 else 0.0
    mad = float(np.median(np.abs(x - np.median(x))))
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "var": float(np.var(x, ddof=1)) if len(x) > 1 else 0.0,
        "rms": rms,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "median": float(np.median(x)),
        "mad": mad,
        "skew": float(skew(x, bias=False, nan_policy="omit")) if len(x) > 2 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=True, bias=False, nan_policy="omit")) if len(x) > 3 else 0.0,
        "zero_crossing_rate": zero_crossing_rate,
        "crest_factor": peak_abs / (rms + _EPS),
    }


def frequency_domain_features(
    window: np.ndarray,
    sampling_rate: float,
    bands: list[tuple[float, float]] | None = None,
    fundamental_freq: float | None = None,
    n_harmonics: int = 5,
) -> dict[str, float]:
    x = np.asarray(window, dtype=float)
    x = x - np.mean(x)
    freqs, psd = scipy_signal.welch(x, fs=sampling_rate, nperseg=min(len(x), 256))
    total_power = float(np.sum(psd) + _EPS)
    peak_idx = int(np.argmax(psd)) if len(psd) else 0
    dominant_freq = float(freqs[peak_idx]) if len(freqs) else 0.0
    peak_amp = float(psd[peak_idx]) if len(psd) else 0.0
    spectral_centroid = float(np.sum(freqs * psd) / total_power) if len(freqs) else 0.0
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
    ) if len(freqs) else 0.0
    cumulative = np.cumsum(psd)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1])) if len(cumulative) else 0
    spectral_rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)]) if len(freqs) else 0.0
    spectral_entropy = _safe_entropy(psd / total_power) / np.log2(len(psd) + _EPS) if len(psd) > 1 else 0.0

    out = {
        "dominant_freq": dominant_freq,
        "main_peak_power": peak_amp,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff_85": spectral_rolloff,
        "spectral_entropy": float(spectral_entropy),
        "peak_ratio": peak_amp / total_power,
    }

    if bands is not None:
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            out[f"band_energy_{low:g}_{high:g}hz"] = float(np.sum(psd[mask]) / total_power)

    if fundamental_freq and fundamental_freq > 0:
        out["thd"] = total_harmonic_distortion(freqs, psd, fundamental_freq, n_harmonics)
    else:
        out["thd"] = np.nan
    return out


def total_harmonic_distortion(
    freqs: np.ndarray, psd: np.ndarray, fundamental_freq: float, n_harmonics: int = 5
) -> float:
    def nearest_power(target: float) -> float:
        idx = int(np.argmin(np.abs(freqs - target)))
        return float(psd[idx])

    fundamental_power = nearest_power(fundamental_freq)
    harmonic_power = sum(nearest_power(fundamental_freq * h) for h in range(2, n_harmonics + 1))
    return float(np.sqrt(harmonic_power) / (np.sqrt(fundamental_power) + _EPS))


def dynamic_features(window: np.ndarray, sampling_rate: float) -> dict[str, float]:
    x = np.asarray(window, dtype=float)
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered) + _EPS)
    autocorr_lag1 = float(np.dot(centered[:-1], centered[1:]) / denom) if len(x) > 1 else 0.0
    t = np.arange(len(x), dtype=float) / sampling_rate
    slope = float(np.polyfit(t, x, deg=1)[0]) if len(x) > 1 else 0.0
    return {"autocorr_lag1": autocorr_lag1, "trend_slope": slope}


def extract_features(
    windows: np.ndarray,
    sampling_rate: float,
    bands: list[tuple[float, float]] | None = None,
    fundamental_freq: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for w in windows:
        row = {}
        row.update(time_domain_features(w))
        row.update(frequency_domain_features(w, sampling_rate, bands, fundamental_freq))
        row.update(dynamic_features(w, sampling_rate))
        rows.append(row)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    return df.fillna(df.median(numeric_only=True)).fillna(0.0)
