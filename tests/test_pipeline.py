from tsad_feature.data import synthetic_electrical_signal
from tsad_feature.detectors import DetectorConfig
from tsad_feature.pipeline import FeaturePipelineConfig, train_and_score


def test_train_and_score_smoke():
    signal, _ = synthetic_electrical_signal(n_samples=3000, sampling_rate=500, fundamental_freq=50)
    cfg = FeaturePipelineConfig(sampling_rate=500, window_size=256, overlap=0.5, fundamental_freq=50)
    model, scored = train_and_score(signal, cfg, DetectorConfig(n_estimators=20, contamination="auto"))
    assert model is not None
    assert {"anomaly_score", "is_anomaly", "window_start", "window_end"}.issubset(scored.columns)
    assert len(scored) > 0
