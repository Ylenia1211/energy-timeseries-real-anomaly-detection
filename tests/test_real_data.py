from pathlib import Path

from tsad_feature.real_data import RealSeriesConfig, infer_sampling_rate_hz, load_real_series


def test_load_real_series_small_fixture(tmp_path):
    p = tmp_path / "meter.csv"
    p.write_text(
        "datetime,id,building,model,TotW\n"
        "2024-01-01 00:00:00,A,B,M,1.0\n"
        "2024-01-01 00:05:00,A,B,M,2.0\n"
        "2024-01-01 00:10:00,A,B,M,3.0\n",
        encoding="utf-8",
    )
    df = load_real_series(RealSeriesConfig(csv_path=str(p), meter_id="A", value_col="TotW", resample="5min"))
    assert list(df["value"]) == [1.0, 2.0, 3.0]
    assert infer_sampling_rate_hz(df) == 1 / 300
