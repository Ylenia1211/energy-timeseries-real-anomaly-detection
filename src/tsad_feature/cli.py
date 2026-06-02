from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_signal_csv, synthetic_electrical_signal
from .detectors import DetectorConfig, save_model
from .ensemble import combine_anomaly_outputs, group_sample_anomalies
from .forecast import ARIMAConfig, ProphetConfig, ResidualThresholdConfig, detect_forecast_anomalies
from .pipeline import FeaturePipelineConfig, matrix_profile_onsets, train_and_score
from .matrix_profile import MatrixProfileConfig, compute_matrix_profile, find_discords, group_discords
from .report import enrich_events, generate_html_report
from .plotting import plot_forecast_anomalies, plot_matrix_profile, plot_signal_with_scores
from .real_data import RealSeriesConfig, infer_sampling_rate_hz, list_available_series, load_real_series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anomaly detection su time series: feature, Matrix Profile, ARIMA e Prophet.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run completo su segnale elettrico sintetico.")
    demo.add_argument("--outdir", default="outputs", help="Output directory.")

    run = sub.add_parser("run", help="Run feature-based + Matrix Profile su un CSV generico.")
    run.add_argument("--csv", required=True, help="Input CSV path.")
    run.add_argument("--value-col", default="value", help="Colonna segnale.")
    run.add_argument("--sampling-rate", type=float, required=True, help="Sampling rate in Hz.")
    run.add_argument("--window-size", type=int, default=1024, help="Dimensione finestra in campioni.")
    run.add_argument("--overlap", type=float, default=0.5, help="Overlap finestre in [0, 1).")
    run.add_argument("--fundamental-freq", type=float, default=None, help="Frequenza fondamentale per THD.")
    run.add_argument("--contamination", default="auto", help="IsolationForest contamination: 'auto' o float.")
    run.add_argument("--outdir", default="outputs", help="Output directory.")

    forecast = sub.add_parser("forecast", help="Run ARIMA o Prophet forecast-based su CSV generico.")
    forecast.add_argument("--csv", required=True, help="Input CSV path.")
    forecast.add_argument("--value-col", default="value", help="Colonna segnale.")
    forecast.add_argument("--sampling-rate", type=float, default=None, help="Sampling rate opzionale per plotting.")
    forecast.add_argument("--model", choices=("arima", "prophet"), default="arima", help="Forecast model.")
    forecast.add_argument("--train-ratio", type=float, default=0.7, help="Frazione iniziale per fit/calibrazione.")
    forecast.add_argument("--threshold-z", type=float, default=3.5, help="Soglia robust z-score residui.")
    forecast.add_argument("--arima-order", default="2,0,2", help="ARIMA order p,d,q.")
    forecast.add_argument("--seasonal-order", default="1,0,1,96", help="Seasonal order P,D,Q,s. Per dati a 15min, s=96 equivale a 24h.")
    forecast.add_argument("--prophet-freq", default="s", help="Frequenza pandas per Prophet: s, min, 15min, h.")
    forecast.add_argument("--outdir", default="outputs", help="Output directory.")

    real_summary = sub.add_parser("real-summary", help="Lista sensori/serie presenti in data/raw.")
    real_summary.add_argument("--data-dir", default="data/raw", help="Directory CSV reali.")
    real_summary.add_argument("--out", default=None, help="CSV opzionale per salvare il riepilogo.")

    real = sub.add_parser("real-run", help="Pipeline completa su dati reali del progetto caricato.")
    real.add_argument("--csv", required=True, help="CSV reale, es. data/raw/ED14_20240426.csv.")
    real.add_argument("--meter-id", default=None, help="Filtro id sensore, es. ARCH_FM.")
    real.add_argument("--building", default=None, help="Filtro building, es. ED14.")
    real.add_argument("--model-filter", default=None, help="Filtro model del meter.")
    real.add_argument("--value-col", default="TotW", help="Colonna numerica da analizzare.")
    real.add_argument("--resample", default="15min", help="Resampling pandas, es. 5min, 15min, 1h. Usa 'none' per disabilitare.")
    real.add_argument("--agg", choices=("mean", "sum", "median", "max", "min"), default="mean", help="Aggregazione sul resampling.")
    real.add_argument("--window-size", type=int, default=96, help="Finestra in campioni. Con 15min, 96 = 1 giorno.")
    real.add_argument("--overlap", type=float, default=0.5, help="Overlap finestre.")
    real.add_argument("--contamination", default="0.03", help="Quota attesa anomalie per IsolationForest.")
    real.add_argument("--min-votes", type=int, default=2, help="Voti minimi nell'ensemble.")
    real.add_argument("--mp-top-k", type=int, default=30, help="Numero massimo di discords Matrix Profile da selezionare.")
    real.add_argument("--mp-zscore-threshold", type=float, default=3.0, help="Soglia robust z-score per discords Matrix Profile.")
    real.add_argument("--arima-threshold-z", type=float, default=4.5, help="Soglia robust z-score specifica per ARIMA/SARIMAX.")
    real.add_argument("--prophet-threshold-z", type=float, default=3.5, help="Soglia robust z-score specifica per Prophet.")
    real.add_argument("--no-report", action="store_true", help="Non generare report.html automatico.")
    real.add_argument("--skip-matrix-profile", action="store_true", help="Salta Matrix Profile.")
    real.add_argument("--skip-arima", action="store_true", help="Salta ARIMA/SARIMAX.")
    real.add_argument("--skip-prophet", action="store_true", help="Salta Prophet.")
    real.add_argument("--train-ratio", type=float, default=0.7, help="Frazione train per forecast detectors.")
    real.add_argument("--threshold-z", type=float, default=3.5, help="Soglia robust z-score residui.")
    real.add_argument("--arima-order", default="2,1,2", help="ARIMA order p,d,q.")
    real.add_argument("--seasonal-order", default="1,0,1,96", help="Seasonal order P,D,Q,s. Per dati a 15min, s=96 equivale a 24h.")
    real.add_argument("--outdir", default="outputs/real_run", help="Output directory.")
    return parser


def _parse_contamination(value: str):
    return "auto" if value == "auto" else float(value)


def _parse_tuple(value: str, expected_len: int) -> tuple[int, ...]:
    parts = tuple(int(v.strip()) for v in value.split(","))
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated integers, got {value!r}")
    return parts


def _default_bands(sampling_rate: float) -> tuple[tuple[float, float], ...]:
    nyquist = max(sampling_rate / 2.0, 1e-12)
    return ((0.0, nyquist * 0.1), (nyquist * 0.1, nyquist * 0.3), (nyquist * 0.3, nyquist))


def run_demo(outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    signal, df = synthetic_electrical_signal()
    df.to_csv(out / "synthetic_signal.csv", index=False)

    feature_cfg = FeaturePipelineConfig(
        sampling_rate=1000.0,
        window_size=1024,
        overlap=0.5,
        bands=((0, 10), (10, 50), (50, 100), (100, 250), (250, 500)),
        fundamental_freq=50.0,
    )
    detector_cfg = DetectorConfig(contamination=0.08)
    model, scored = train_and_score(signal, feature_cfg, detector_cfg)
    scored.to_csv(out / "feature_anomaly_scores.csv", index=False)
    save_model(model, str(out / "isolation_forest.joblib"))
    plot_signal_with_scores(signal, scored, str(out / "signal_scores.png"), sampling_rate=1000.0)

    try:
        mp, discords, onsets = matrix_profile_onsets(signal, subseq_len=1024, top_k=20)
        mp.to_csv(out / "matrix_profile.csv", index=False)
        discords.to_csv(out / "matrix_profile_discords.csv", index=False)
        onsets.to_csv(out / "estimated_onsets.csv", index=False)
        plot_matrix_profile(mp, discords, str(out / "matrix_profile.png"))
    except ImportError as exc:
        (out / "matrix_profile_skipped.txt").write_text(str(exc), encoding="utf-8")
    print(f"Demo completed. Files written to: {out.resolve()}")


def run_csv(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    signal, _ = load_signal_csv(args.csv, value_col=args.value_col)
    contamination = _parse_contamination(args.contamination)
    feature_cfg = FeaturePipelineConfig(
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        overlap=args.overlap,
        fundamental_freq=args.fundamental_freq,
    )
    detector_cfg = DetectorConfig(contamination=contamination)
    model, scored = train_and_score(signal, feature_cfg, detector_cfg)
    scored.to_csv(out / "feature_anomaly_scores.csv", index=False)
    save_model(model, str(out / "isolation_forest.joblib"))
    plot_signal_with_scores(signal, scored, str(out / "signal_scores.png"), sampling_rate=args.sampling_rate)

    try:
        mp, discords, onsets = matrix_profile_onsets(signal, subseq_len=args.window_size, top_k=20)
        mp.to_csv(out / "matrix_profile.csv", index=False)
        discords.to_csv(out / "matrix_profile_discords.csv", index=False)
        onsets.to_csv(out / "estimated_onsets.csv", index=False)
        plot_matrix_profile(mp, discords, str(out / "matrix_profile.png"))
    except ImportError as exc:
        (out / "matrix_profile_skipped.txt").write_text(str(exc), encoding="utf-8")
    print(f"Run completed. Files written to: {out.resolve()}")


def run_forecast(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    signal, _ = load_signal_csv(args.csv, value_col=args.value_col)
    threshold = ResidualThresholdConfig(z_threshold=args.threshold_z)
    arima_cfg = ARIMAConfig(
        order=_parse_tuple(args.arima_order, 3),
        seasonal_order=_parse_tuple(args.seasonal_order, 4),
        train_ratio=args.train_ratio,
        threshold=threshold,
    )
    prophet_cfg = ProphetConfig(
        train_ratio=args.train_ratio,
        freq=args.prophet_freq,
        threshold=threshold,
    )
    scores = detect_forecast_anomalies(
        signal,
        model=args.model,
        arima_config=arima_cfg,
        prophet_config=prophet_cfg,
    )
    scores.to_csv(out / f"{args.model}_forecast_anomaly_scores.csv", index=False)
    plot_forecast_anomalies(scores, str(out / f"{args.model}_forecast_anomalies.png"), sampling_rate=args.sampling_rate)
    print(f"Forecast run completed. Files written to: {out.resolve()}")


def run_real_summary(args: argparse.Namespace) -> None:
    summary = list_available_series(args.data_dir)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False, max_rows=100))


def run_real(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    resample = None if str(args.resample).lower() == "none" else args.resample
    series_cfg = RealSeriesConfig(
        csv_path=args.csv,
        value_col=args.value_col,
        meter_id=args.meter_id,
        model=args.model_filter,
        building=args.building,
        resample=resample,
        agg=args.agg,
    )
    frame = load_real_series(series_cfg)
    frame.to_csv(out / "clean_series.csv", index=False)
    signal = frame["value"].to_numpy(dtype=float)
    sampling_rate = infer_sampling_rate_hz(frame)
    metadata = {
        "csv": args.csv,
        "meter_id": args.meter_id,
        "building": args.building,
        "value_col": args.value_col,
        "resample": resample,
        "agg": args.agg,
        "n_samples": int(len(signal)),
        "sampling_rate_hz_inferred": sampling_rate,
        "window_size": args.window_size,
    }

    feature_scores = None
    matrix_discords = None
    arima_scores = None
    prophet_scores = None

    if len(signal) <= args.window_size:
        raise ValueError("La serie pulita è più corta della finestra. Riduci --window-size o cambia resample.")

    feature_cfg = FeaturePipelineConfig(
        sampling_rate=sampling_rate,
        window_size=args.window_size,
        overlap=args.overlap,
        bands=_default_bands(sampling_rate),
        fundamental_freq=None,
    )
    detector_cfg = DetectorConfig(contamination=_parse_contamination(args.contamination))
    model, feature_scores = train_and_score(signal, feature_cfg, detector_cfg)
    feature_scores.to_csv(out / "feature_isolation_forest_scores.csv", index=False)
    save_model(model, str(out / "isolation_forest.joblib"))
    plot_signal_with_scores(signal, feature_scores, str(out / "feature_isolation_forest.png"), sampling_rate=None)

    if not args.skip_matrix_profile:
        try:
            mp_cfg = MatrixProfileConfig(
                subseq_len=args.window_size,
                top_k=args.mp_top_k,
                zscore_threshold=args.mp_zscore_threshold,
            )
            mp = compute_matrix_profile(signal, subseq_len=args.window_size)
            matrix_discords = find_discords(mp, mp_cfg)
            onsets = group_discords(matrix_discords, gap=args.window_size)
            mp.to_csv(out / "matrix_profile.csv", index=False)
            matrix_discords.to_csv(out / "matrix_profile_discords.csv", index=False)
            onsets.to_csv(out / "matrix_profile_onsets.csv", index=False)
            plot_matrix_profile(mp, matrix_discords, str(out / "matrix_profile.png"))
            metadata["matrix_profile_top_k"] = args.mp_top_k
            metadata["matrix_profile_zscore_threshold"] = args.mp_zscore_threshold
        except Exception as exc:  # keep end-to-end pipeline usable when optional deps fail
            metadata["matrix_profile_error"] = repr(exc)
            (out / "matrix_profile_skipped.txt").write_text(repr(exc), encoding="utf-8")
            matrix_discords = None

    threshold = ResidualThresholdConfig(z_threshold=args.threshold_z)
    arima_threshold = ResidualThresholdConfig(z_threshold=args.arima_threshold_z)
    prophet_threshold = ResidualThresholdConfig(z_threshold=args.prophet_threshold_z)
    metadata["arima_threshold_z"] = args.arima_threshold_z
    metadata["prophet_threshold_z"] = args.prophet_threshold_z
    if not args.skip_arima:
        try:
            arima_cfg = ARIMAConfig(
                order=_parse_tuple(args.arima_order, 3),
                seasonal_order=_parse_tuple(args.seasonal_order, 4),
                train_ratio=args.train_ratio,
                threshold=arima_threshold,
            )
            arima_scores = detect_forecast_anomalies(signal, model="arima", arima_config=arima_cfg)
            arima_scores.to_csv(out / "arima_scores.csv", index=False)
            plot_forecast_anomalies(arima_scores, str(out / "arima_anomalies.png"), sampling_rate=None)
        except Exception as exc:
            metadata["arima_error"] = repr(exc)
            (out / "arima_skipped.txt").write_text(repr(exc), encoding="utf-8")
            arima_scores = None

    if not args.skip_prophet:
        try:
            prophet_cfg = ProphetConfig(train_ratio=args.train_ratio, freq=args.resample, threshold=prophet_threshold)
            prophet_scores = detect_forecast_anomalies(
                signal,
                model="prophet",
                prophet_config=prophet_cfg,
                timestamps=frame["timestamp"],
            )
            prophet_scores.to_csv(out / "prophet_scores.csv", index=False)
            plot_forecast_anomalies(prophet_scores, str(out / "prophet_anomalies.png"), sampling_rate=None)
        except Exception as exc:
            metadata["prophet_error"] = repr(exc)
            (out / "prophet_skipped.txt").write_text(repr(exc), encoding="utf-8")
            prophet_scores = None

    ensemble = combine_anomaly_outputs(
        n_samples=len(signal),
        feature_scores=feature_scores,
        matrix_discords=matrix_discords,
        matrix_subseq_len=args.window_size if matrix_discords is not None else None,
        arima_scores=arima_scores,
        prophet_scores=prophet_scores,
        min_votes=args.min_votes,
    )
    ensemble = frame[["timestamp", "sample_index", "value"]].merge(ensemble, on="sample_index", how="left")
    ensemble.to_csv(out / "ensemble_sample_scores.csv", index=False)
    events = group_sample_anomalies(ensemble, max_gap=max(1, args.window_size // 4))
    events = enrich_events(events, ensemble)
    events.to_csv(out / "ensemble_anomaly_events.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if not args.no_report:
        generate_html_report(out, title=f"Anomaly report - {args.meter_id or args.building or Path(args.csv).stem} / {args.value_col}")
    print(f"Real-data pipeline completed. Files written to: {out.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.outdir)
    elif args.command == "run":
        run_csv(args)
    elif args.command == "forecast":
        run_forecast(args)
    elif args.command == "real-summary":
        run_real_summary(args)
    elif args.command == "real-run":
        run_real(args)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
