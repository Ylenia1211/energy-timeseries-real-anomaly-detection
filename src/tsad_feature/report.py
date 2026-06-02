from __future__ import annotations

from pathlib import Path
import html
import json
import numpy as np
import pandas as pd

DETECTOR_COLS = {
    "feature": "feature_is_anomaly",
    "matrix_profile": "mp_is_anomaly",
    "arima": "arima_is_anomaly",
    "prophet": "prophet_is_anomaly",
}


def _safe_bool_sum(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())


def classify_event(row: pd.Series) -> str:
    """Rule-based, interpretable event class for meter anomalies."""
    value_mean = float(row.get("value_mean", np.nan))
    global_median = float(row.get("global_value_median", np.nan))
    global_q75 = float(row.get("global_value_q75", np.nan))
    feature = int(row.get("feature_samples", 0)) > 0
    mp = int(row.get("matrix_profile_samples", 0)) > 0
    arima = int(row.get("arima_samples", 0)) > 0
    prophet = int(row.get("prophet_samples", 0)) > 0

    high_load = np.isfinite(value_mean) and np.isfinite(global_q75) and value_mean >= global_q75
    low_or_baseline = np.isfinite(value_mean) and np.isfinite(global_median) and value_mean <= global_median * 1.15

    if high_load and (prophet or arima):
        return "HIGH_LOAD_FORECAST_RESIDUAL"
    if mp and feature and low_or_baseline:
        return "LOW_LOAD_SHAPE_PROFILE_SHIFT"
    if mp and not prophet:
        return "SHAPE_BASED_DISCORD"
    if arima and prophet and not mp:
        return "FORECAST_RESIDUAL"
    if feature and not (arima or prophet or mp):
        return "FEATURE_WINDOW_OUTLIER"
    return "MIXED_OR_UNCERTAIN"


def enrich_events(events: pd.DataFrame, sample_scores: pd.DataFrame) -> pd.DataFrame:
    """Add timestamps, detector contributions, value stats and class labels."""
    if events.empty:
        return events.copy()

    out = events.copy()
    if "timestamp" in sample_scores.columns:
        idx_to_ts = sample_scores.set_index("sample_index")["timestamp"]
        if "onset_timestamp" not in out.columns:
            out.insert(1, "onset_timestamp", out["onset_index"].map(idx_to_ts))
        if "end_timestamp" not in out.columns:
            out.insert(3, "end_timestamp", out["end_index"].map(idx_to_ts))

    if "timestamp" in sample_scores.columns:
        dt = pd.to_datetime(sample_scores["timestamp"], errors="coerce")
        step_hours = float(dt.diff().dropna().dt.total_seconds().median() / 3600.0) if dt.notna().sum() > 1 else np.nan
    else:
        step_hours = np.nan

    value_median = float(sample_scores["value"].median()) if "value" in sample_scores.columns else np.nan
    value_q75 = float(sample_scores["value"].quantile(0.75)) if "value" in sample_scores.columns else np.nan

    rows = []
    for _, event in out.iterrows():
        start = int(event["onset_index"])
        end = int(event["end_index"])
        chunk = sample_scores[(sample_scores["sample_index"] >= start) & (sample_scores["sample_index"] <= end)]
        row = event.to_dict()
        if "value" in chunk.columns and not chunk.empty:
            row.update(
                {
                    "value_mean": float(chunk["value"].mean()),
                    "value_min": float(chunk["value"].min()),
                    "value_max": float(chunk["value"].max()),
                    "global_value_median": value_median,
                    "global_value_q75": value_q75,
                }
            )
        if np.isfinite(step_hours):
            row["duration_hours"] = float(row.get("duration_samples", 0)) * step_hours
        for name, col in DETECTOR_COLS.items():
            row[f"{name}_samples"] = _safe_bool_sum(chunk, col)
        rows.append(row)

    enriched = pd.DataFrame(rows)
    enriched["event_class"] = enriched.apply(classify_event, axis=1)
    preferred = [
        "onset_index", "onset_timestamp", "end_index", "end_timestamp", "duration_samples", "duration_hours",
        "max_score", "max_votes", "event_class", "value_mean", "value_min", "value_max",
        "feature_samples", "matrix_profile_samples", "arima_samples", "prophet_samples",
    ]
    cols = [c for c in preferred if c in enriched.columns] + [c for c in enriched.columns if c not in preferred]
    return enriched[cols]


def _table_html(df: pd.DataFrame, max_rows: int = 20, float_format: str = "{:.3f}") -> str:
    if df.empty:
        return "<p><em>Nessun dato disponibile.</em></p>"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=["float"]).columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else float_format.format(x))
    return view.to_html(index=False, escape=True, classes="table")


def generate_html_report(outdir: str | Path, title: str = "Time Series Anomaly Detection Report") -> Path:
    """Generate a self-contained HTML summary that links to generated PNG/CSV files."""
    out = Path(outdir)
    sample_path = out / "ensemble_sample_scores.csv"
    events_path = out / "ensemble_anomaly_events.csv"
    metadata_path = out / "run_metadata.json"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    sample = pd.read_csv(sample_path)
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    n = len(sample)
    anomalous = int(sample.get("ensemble_is_anomaly", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    pct = 100.0 * anomalous / n if n else 0.0
    vote_counts = sample.get("vote_count", pd.Series(dtype=int)).value_counts().sort_index().rename_axis("votes").reset_index(name="samples")

    detector_summary = []
    for name, col in DETECTOR_COLS.items():
        count = _safe_bool_sum(sample, col)
        detector_summary.append({"detector": name, "anomaly_samples": count, "percentage": 100 * count / n if n else 0.0})
    detector_summary = pd.DataFrame(detector_summary)

    top_events = events.sort_values(["max_votes", "max_score", "duration_samples"], ascending=False) if not events.empty else events
    if "timestamp" in sample.columns:
        ts = pd.to_datetime(sample["timestamp"], errors="coerce")
        tmp = sample.copy()
        tmp["month"] = ts.dt.to_period("M").astype(str)
        tmp["hour"] = ts.dt.hour
        tmp["weekday"] = ts.dt.day_name()
        monthly = tmp.groupby("month").agg(samples=("sample_index", "size"), anomalies=("ensemble_is_anomaly", "sum")).reset_index()
        monthly["anomaly_pct"] = 100 * monthly["anomalies"] / monthly["samples"]
        hourly = tmp.groupby("hour").agg(samples=("sample_index", "size"), anomalies=("ensemble_is_anomaly", "sum")).reset_index()
        hourly["anomaly_pct"] = 100 * hourly["anomalies"] / hourly["samples"]
    else:
        monthly = pd.DataFrame()
        hourly = pd.DataFrame()

    imgs = [p.name for p in out.glob("*.png")]
    img_html = "".join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in imgs)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #17202a; }
    h1, h2 { color: #102a43; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
    .card { border: 1px solid #d9e2ec; border-radius: 12px; padding: 14px; background: #f8fbff; }
    .metric { font-size: 1.6rem; font-weight: 700; }
    .table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.92rem; }
    .table th, .table td { border: 1px solid #d9e2ec; padding: 8px; text-align: left; }
    .table th { background: #eef4fb; }
    img { max-width: 100%; border: 1px solid #d9e2ec; border-radius: 10px; }
    figure { margin: 18px 0; }
    code { background: #eef4fb; padding: 2px 4px; border-radius: 4px; }
    """
    html_doc = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p>Report automatico generato dalla pipeline ensemble: feature-based Isolation Forest, Matrix Profile, ARIMA/SARIMAX e Prophet.</p>
<h2>Configurazione run</h2>
<pre>{html.escape(json.dumps(metadata, indent=2, default=str))}</pre>
<div class="cards">
  <div class="card"><div>Campioni</div><div class="metric">{n}</div></div>
  <div class="card"><div>Campioni anomali ensemble</div><div class="metric">{anomalous}</div></div>
  <div class="card"><div>Percentuale anomalie</div><div class="metric">{pct:.2f}%</div></div>
  <div class="card"><div>Eventi aggregati</div><div class="metric">{len(events)}</div></div>
</div>
<h2>Contributo detector</h2>
{_table_html(detector_summary)}
<h2>Distribuzione voti</h2>
{_table_html(vote_counts)}
<h2>Top eventi</h2>
{_table_html(top_events, max_rows=25)}
<h2>Anomalie per mese</h2>
{_table_html(monthly, max_rows=36)}
<h2>Anomalie per ora</h2>
{_table_html(hourly, max_rows=24)}
<h2>Grafici</h2>
{img_html if img_html else '<p><em>Nessun grafico PNG trovato nella directory.</em></p>'}
</body></html>"""
    report_path = out / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
