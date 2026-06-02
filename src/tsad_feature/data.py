from __future__ import annotations

import numpy as np
import pandas as pd


def load_signal_csv(path: str, value_col: str = "value", time_col: str | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(path)
    if value_col not in df.columns:
        raise ValueError(f"Column {value_col!r} not found. Available columns: {list(df.columns)}")
    signal = df[value_col].to_numpy(dtype=float)
    if time_col is not None and time_col not in df.columns:
        raise ValueError(f"Column {time_col!r} not found. Available columns: {list(df.columns)}")
    return signal, df


def synthetic_electrical_signal(
    n_samples: int = 20_000,
    sampling_rate: float = 1000.0,
    fundamental_freq: float = 50.0,
    noise_std: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate a demo signal with local anomalies.

    Anomalies include amplitude swell, sag, harmonic distortion and transient spikes.
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples) / sampling_rate
    x = np.sin(2 * np.pi * fundamental_freq * t)
    x += 0.08 * np.sin(2 * np.pi * 3 * fundamental_freq * t)
    x += rng.normal(0, noise_std, size=n_samples)

    labels = np.zeros(n_samples, dtype=int)

    def apply_interval(start_frac: float, end_frac: float, fn) -> None:
        a = int(start_frac * n_samples)
        b = max(a + 1, int(end_frac * n_samples))
        b = min(b, n_samples)
        fn(a, b)

    # Swell
    apply_interval(0.20, 0.25, lambda a, b: (x.__setitem__(slice(a, b), x[a:b] * 1.8), labels.__setitem__(slice(a, b), 1)))

    # Sag
    apply_interval(0.45, 0.49, lambda a, b: (x.__setitem__(slice(a, b), x[a:b] * 0.35), labels.__setitem__(slice(a, b), 1)))

    # Harmonic distortion
    def add_harmonic(a: int, b: int) -> None:
        x[a:b] += 0.45 * np.sin(2 * np.pi * 5 * fundamental_freq * t[a:b])
        labels[a:b] = 1
    apply_interval(0.65, 0.70, add_harmonic)

    # Transients
    spike_start = int(0.80 * n_samples)
    spike_end = max(spike_start + 1, int(0.90 * n_samples))
    candidates = np.arange(spike_start, min(spike_end, n_samples))
    n_spikes = min(20, len(candidates))
    if n_spikes > 0:
        spike_idx = rng.choice(candidates, size=n_spikes, replace=False)
        x[spike_idx] += rng.normal(3.0, 0.5, size=len(spike_idx))
        labels[spike_idx] = 1

    df = pd.DataFrame({"time": t, "value": x, "label": labels})
    return x, df
