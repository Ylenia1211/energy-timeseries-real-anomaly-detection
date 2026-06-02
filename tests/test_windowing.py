import numpy as np

from tsad_feature.windowing import WindowConfig, sliding_windows


def test_sliding_windows_overlap():
    x = np.arange(10)
    windows, starts = sliding_windows(x, WindowConfig(window_size=4, overlap=0.5))
    assert starts.tolist() == [0, 2, 4, 6]
    assert windows.shape == (4, 4)
    assert windows[0].tolist() == [0, 1, 2, 3]
