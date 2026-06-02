from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import logging

import numpy as np
import pandas as pd

# Keep optional Stan/Prophet logs readable when the CLI runs many experiments.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ResidualThresholdConfig:
    """Robust residual threshold based on median absolute deviation.

    A point is anomalous when its absolute robust z-score is above ``z_threshold``.
    The MAD scale is calibrated on the training section only.
    """

    z_threshold: float = 3.5
    min_scale: float = 1e-8


@dataclass(frozen=True)
class ARIMAConfig:
    order: tuple[int, int, int] = (2, 1, 2)
    # For 15 minute electrical data, 96 samples = one day.
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 96)
    train_ratio: float = 0.7
    threshold: ResidualThresholdConfig = ResidualThresholdConfig(z_threshold=4.5)


@dataclass(frozen=True)
class ProphetConfig:
    train_ratio: float = 0.7
    freq: str = "15min"
    interval_width: float = 0.99
    yearly_seasonality: bool | str = False
    weekly_seasonality: bool | str = True
    daily_seasonality: bool | str = "auto"
    threshold: ResidualThresholdConfig = ResidualThresholdConfig(z_threshold=3.5)


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("Forecast-based detectors require a 1D signal.")
    if len(values) < 20:
        raise ValueError("Signal is too short for forecast-based anomaly detection.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signal contains NaN or infinite values. Clean/impute it first.")
    return values


def _train_size(n_samples: int, train_ratio: float) -> int:
    if not 0.1 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in the open interval (0.1, 1.0).")
    size = int(n_samples * train_ratio)
    return max(10, min(size, n_samples - 1))


def robust_residual_scores(
    residuals: np.ndarray,
    train_end: int,
    config: ResidualThresholdConfig = ResidualThresholdConfig(),
) -> pd.DataFrame:
    """Convert residuals into robust z-scores and anomaly flags."""
    residuals = np.asarray(residuals, dtype=float)
    calibration = residuals[:train_end]
    median = float(np.median(calibration))
    mad = float(np.median(np.abs(calibration - median)))
    scale = max(1.4826 * mad, config.min_scale)
    robust_z = np.abs((residuals - median) / scale)
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(residuals)),
            "residual": residuals,
            "anomaly_score": robust_z,
            "is_anomaly": robust_z >= config.z_threshold,
            "threshold_z": config.z_threshold,
            "residual_median_train": median,
            "residual_mad_scale_train": scale,
        }
    )


def detect_arima_anomalies(signal: np.ndarray, config: ARIMAConfig = ARIMAConfig()) -> pd.DataFrame:
    """Detect anomalies through SARIMAX one-step/recursive residuals.

    The default seasonal order is tuned for 15-minute electrical data. If your
    series is not sampled every 15 minutes, pass a different seasonal period or
    ``(0, 0, 0, 0)`` to disable seasonality.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "ARIMA support requires statsmodels. Install with: pip install -e '.[forecast]'"
        ) from exc

    values = _validate_signal(signal)
    train_end = _train_size(len(values), config.train_ratio)
    train = values[:train_end]

    fitted = SARIMAX(
        train,
        order=config.order,
        seasonal_order=config.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    pred = fitted.get_prediction(start=0, end=len(values) - 1, dynamic=False)
    yhat = np.asarray(pred.predicted_mean, dtype=float)
    residuals = values - yhat
    scores = robust_residual_scores(residuals, train_end, config.threshold)
    scores.insert(1, "y", values)
    scores.insert(2, "yhat", yhat)
    scores.insert(3, "model", "arima")
    scores["arima_order"] = str(config.order)
    scores["seasonal_order"] = str(config.seasonal_order)
    return scores


def _make_prophet_frame(signal: np.ndarray, freq: str, timestamps: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    if timestamps is None:
        dates = pd.date_range("2000-01-01", periods=len(signal), freq=freq)
    else:
        dates = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        if dates.isna().any():
            raise ValueError("timestamps contains invalid datetime values for Prophet.")
        if len(dates) != len(signal):
            raise ValueError("timestamps length must match signal length.")
    return pd.DataFrame({"ds": dates.to_numpy(), "y": signal})


def detect_prophet_anomalies(
    signal: np.ndarray,
    config: ProphetConfig = ProphetConfig(),
    timestamps: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Detect anomalies using Prophet forecast intervals and residual scores.

    When real timestamps are provided, they are propagated to the output. This
    makes the Prophet CSV directly interpretable and avoids synthetic year-2000
    timestamps in real-data runs.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Prophet support requires prophet. Install with: pip install -e '.[forecast]'"
        ) from exc

    values = _validate_signal(signal)
    train_end = _train_size(len(values), config.train_ratio)
    full = _make_prophet_frame(values, config.freq, timestamps=timestamps)
    train = full.iloc[:train_end].copy()

    model = Prophet(
        interval_width=config.interval_width,
        yearly_seasonality=config.yearly_seasonality,
        weekly_seasonality=config.weekly_seasonality,
        daily_seasonality=config.daily_seasonality,
    )
    model.fit(train)
    forecast = model.predict(full[["ds"]])

    yhat = forecast["yhat"].to_numpy(dtype=float)
    lower = forecast["yhat_lower"].to_numpy(dtype=float)
    upper = forecast["yhat_upper"].to_numpy(dtype=float)
    residuals = values - yhat
    scores = robust_residual_scores(residuals, train_end, config.threshold)
    interval_flag = (values < lower) | (values > upper)
    scores.insert(1, "ds", full["ds"].to_numpy())
    scores.insert(2, "y", values)
    scores.insert(3, "yhat", yhat)
    scores.insert(4, "yhat_lower", lower)
    scores.insert(5, "yhat_upper", upper)
    scores.insert(6, "model", "prophet")
    scores["outside_interval"] = interval_flag
    scores["is_anomaly"] = scores["is_anomaly"] | interval_flag
    return scores


def detect_forecast_anomalies(
    signal: np.ndarray,
    model: Literal["arima", "prophet"],
    arima_config: ARIMAConfig = ARIMAConfig(),
    prophet_config: ProphetConfig = ProphetConfig(),
    timestamps: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    if model == "arima":
        return detect_arima_anomalies(signal, arima_config)
    if model == "prophet":
        return detect_prophet_anomalies(signal, prophet_config, timestamps=timestamps)
    raise ValueError("model must be either 'arima' or 'prophet'.")
