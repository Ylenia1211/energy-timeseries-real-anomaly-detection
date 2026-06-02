import numpy as np

from tsad_feature.features import extract_features


def test_extract_features_has_expected_columns():
    t = np.linspace(0, 1, 256, endpoint=False)
    windows = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])
    df = extract_features(windows, sampling_rate=256, bands=[(0, 30)], fundamental_freq=10)
    expected = {"mean", "std", "rms", "kurtosis", "dominant_freq", "spectral_entropy", "band_energy_0_30hz", "thd", "autocorr_lag1", "trend_slope"}
    assert expected.issubset(df.columns)
    assert len(df) == 2
