import numpy as np

from tsad_feature.forecast import ResidualThresholdConfig, robust_residual_scores


def test_robust_residual_scores_flags_large_residual():
    residuals = np.zeros(40)
    residuals[30] = 10.0
    scores = robust_residual_scores(residuals, train_end=20, config=ResidualThresholdConfig(z_threshold=3.5))
    assert {"sample_index", "residual", "anomaly_score", "is_anomaly"}.issubset(scores.columns)
    assert bool(scores.loc[30, "is_anomaly"])
    assert scores.loc[30, "anomaly_score"] > scores.loc[0, "anomaly_score"]
