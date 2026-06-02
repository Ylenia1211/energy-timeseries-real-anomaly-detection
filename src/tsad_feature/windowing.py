from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for sliding-window segmentation."""

    window_size: int
    step_size: int | None = None
    overlap: float = 0.5

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise ValueError("window_size must be > 1")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size must be positive")

    @property
    def effective_step(self) -> int:
        if self.step_size is not None:
            return self.step_size
        return max(1, int(round(self.window_size * (1 - self.overlap))))


def sliding_windows(signal: np.ndarray, config: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return windows and their start indices.

    Parameters
    ----------
    signal:
        One-dimensional time series.
    config:
        Window configuration.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = len(x)
    if n < config.window_size:
        raise ValueError("signal length must be >= window_size")

    starts = np.arange(0, n - config.window_size + 1, config.effective_step, dtype=int)
    windows = np.vstack([x[s : s + config.window_size] for s in starts])
    return windows, starts
