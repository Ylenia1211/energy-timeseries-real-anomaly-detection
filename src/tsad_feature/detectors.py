from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class DetectorConfig:
    n_estimators: int = 300
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    random_state: int = 42
    scaler: str = "robust"


def build_isolation_forest(config: DetectorConfig) -> Pipeline:
    scaler = RobustScaler() if config.scaler == "robust" else StandardScaler()
    model = IsolationForest(
        n_estimators=config.n_estimators,
        max_samples=config.max_samples,
        contamination=config.contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    return Pipeline([("scaler", scaler), ("model", model)])


def fit_detector(features: pd.DataFrame, config: DetectorConfig) -> Pipeline:
    pipe = build_isolation_forest(config)
    pipe.fit(features)
    return pipe


def score_detector(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    # sklearn: predict returns -1 for anomaly, 1 for normal.
    prediction = model.predict(features)
    # decision_function: lower values are more anomalous. We invert for readability.
    anomaly_score = -model.decision_function(features)
    return pd.DataFrame(
        {
            "anomaly_score": np.asarray(anomaly_score, dtype=float),
            "is_anomaly": prediction == -1,
        }
    )


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)
