from collections import namedtuple
from datetime import datetime, timedelta
from pandas import DataFrame
import matplotlib.pyplot as plt

from util import plot_dataframe


def apply_zscore_threshold(
    df: DataFrame,
    threshold_val: int,
    threshold_col: str = "Threshold",
) -> None:
    df.loc[:, threshold_col] = threshold_val


def apply_mean_threshold(
    df: DataFrame,
    col: str,
    threshold_col: str = "Threshold",
) -> None:
    df.loc[:, threshold_col] = df[col].mean()


def apply_rolling_std_threshold(
    df: DataFrame,
    col: str,
    window: int = 10,
    multiplier: float = 1.2,
    threshold_col: str = "Threshold",
) -> None:
    df.loc[:, threshold_col] = df[col].rolling(window=window).std() * multiplier


def plot_anomalies(df: DataFrame) -> None:
    # highlight all anomalous data points
    anomalous = df[df["anomalous"] == 1]

    anomaly_meta = namedtuple("anomaly_meta", ["column", "color", "label"])

    anomaly_columns = [
        anomaly_meta("high_flag", "black", "High values"),
    ]

    for col in anomaly_columns:
        plt.scatter(
            anomalous.index, anomalous["TotW"], color=col.color, s=50, label=col.label
        )

    plt.legend()


def plot_anomalous_windows(df: DataFrame) -> None:
    anomalous = df[df["anomalous"] == 1].reset_index()

    for i, anomaly in anomalous.iterrows():
        dt = datetime.strptime(anomaly.datetime, "%Y-%m-%d %H:%M:%S")

        if i == 0:
            last_dt = dt

        if dt < last_dt:
            continue

        bound = timedelta(minutes=60)
        last_dt = dt + bound
        x = df.loc[str(dt - bound) : str(dt + bound)]

        plot_dataframe(
            x,
            ["TotW", "TotW_zscore", "Threshold"],
            {"TotW": "#1F77B4", "TotW_zscore": "#000000", "Threshold": "red"},
            figsize=(10, 5),
        )
        plot_anomalies(x)

        plt.title(f"Anomalie tra {dt - bound} e {dt + bound}")
