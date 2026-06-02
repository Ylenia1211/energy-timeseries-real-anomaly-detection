from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_signal_with_scores(signal: np.ndarray, scores: pd.DataFrame, path: str, sampling_rate: float | None = None) -> None:
    x_axis = np.arange(len(signal)) / sampling_rate if sampling_rate else np.arange(len(signal))
    start_axis = scores["window_start"].to_numpy()
    if sampling_rate:
        start_axis = start_axis / sampling_rate

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(x_axis, signal, linewidth=0.8, label="signal")
    ax1.set_xlabel("time [s]" if sampling_rate else "sample")
    ax1.set_ylabel("signal")

    ax2 = ax1.twinx()
    ax2.plot(start_axis, scores["anomaly_score"], linewidth=1.2, label="anomaly score")
    ax2.scatter(
        start_axis[scores["is_anomaly"].to_numpy()],
        scores.loc[scores["is_anomaly"], "anomaly_score"],
        marker="x",
        label="feature anomalies",
    )
    ax2.set_ylabel("anomaly score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_matrix_profile(mp: pd.DataFrame, discords: pd.DataFrame, path: str) -> None:
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.plot(mp["start_index"], mp["matrix_profile"], linewidth=1.0, label="matrix profile")
    if not discords.empty:
        ax.scatter(discords["start_index"], discords["matrix_profile"], marker="x", label="discords")
    ax.set_xlabel("subsequence start index")
    ax.set_ylabel("z-normalized distance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_forecast_anomalies(scores: pd.DataFrame, path: str, sampling_rate: float | None = None) -> None:
    x_axis = scores["sample_index"].to_numpy()
    if sampling_rate:
        x_axis = x_axis / sampling_rate
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(x_axis, scores["y"], linewidth=0.8, label="signal")
    ax.plot(x_axis, scores["yhat"], linewidth=1.0, label="forecast")
    if {"yhat_lower", "yhat_upper"}.issubset(scores.columns):
        ax.fill_between(x_axis, scores["yhat_lower"], scores["yhat_upper"], alpha=0.2, label="forecast interval")
    anomalies = scores["is_anomaly"].to_numpy(dtype=bool)
    if anomalies.any():
        ax.scatter(x_axis[anomalies], scores.loc[anomalies, "y"], marker="x", label="forecast anomalies")
    ax.set_xlabel("time [s]" if sampling_rate else "sample")
    ax.set_ylabel("signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
