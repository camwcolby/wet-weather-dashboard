from __future__ import annotations
import numpy as np
import pandas as pd


def _scaled(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.quantile(0.10), s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def rank_events(history: pd.DataFrame, runtimes: pd.DataFrame, _rain_unused=None):
    """Rank 2026 wet-weather events from local rainfall and delayed response.

    Each trigger date uses local measured rainfall and searches the following
    48 hours for the plant peak. Pump runtime and station flow are aggregated
    across the trigger date plus the next two calendar days.
    """
    hist = history.copy()
    hist["event_date"] = pd.to_datetime(hist["date"], errors="coerce").dt.normalize()
    hist = hist.dropna(subset=["event_date"]).sort_values("event_date")
    for col in ["rain_in", "plant_flow_mgd", "plant_peak_mgd"]:
        hist[col] = pd.to_numeric(hist.get(col), errors="coerce")

    rt = runtimes.copy()
    if rt is None or rt.empty:
        rt_daily = pd.DataFrame(columns=["date", "total_runtime_hr", "station_flow_kgal"])
    else:
        rt["date"] = pd.to_datetime(rt["date"], errors="coerce").dt.normalize()
        rt_daily = rt.groupby("date", as_index=False).agg(
            total_runtime_hr=("total_runtime_hr", "sum"),
            station_flow_kgal=("flow_kgal", "sum"),
        )
    runtime_idx = rt_daily.set_index("date")["total_runtime_hr"] if not rt_daily.empty else pd.Series(dtype=float)
    flow_idx = rt_daily.set_index("date")["station_flow_kgal"] if not rt_daily.empty else pd.Series(dtype=float)
    peak_idx = hist.set_index("event_date")["plant_peak_mgd"]

    rows = []
    for _, base in hist.iterrows():
        d = base["event_date"]
        response_dates = pd.date_range(d, d + pd.Timedelta(days=2), freq="D")
        peaks = peak_idx.reindex(response_dates)
        response_date = peaks.idxmax() if peaks.notna().any() else d
        rows.append({
            "event_date": d,
            "response_date": response_date,
            "response_lag_hr": (response_date - d).total_seconds() / 3600,
            "rain_in": base.get("rain_in", np.nan),
            "plant_flow_mgd": base.get("plant_flow_mgd", np.nan),
            "plant_peak_mgd": peaks.max(),
            "total_runtime_72h": runtime_idx.reindex(response_dates).sum(min_count=1),
            "station_flow_72h_kgal": flow_idx.reindex(response_dates).sum(min_count=1),
        })
    daily = pd.DataFrame(rows)
    daily["total_runtime_48h"] = daily["total_runtime_72h"]
    daily["rain_score"] = _scaled(daily["rain_in"])
    daily["plant_score"] = _scaled(daily["plant_peak_mgd"])
    daily["runtime_score"] = _scaled(daily["total_runtime_72h"])
    daily["station_flow_score"] = _scaled(daily["station_flow_72h_kgal"])
    daily["storm_score"] = (
        0.38 * daily["rain_score"].fillna(0)
        + 0.34 * daily["plant_score"].fillna(0)
        + 0.18 * daily["runtime_score"].fillna(0)
        + 0.10 * daily["station_flow_score"].fillna(0)
    )
    daily["rainfall_available"] = True

    candidates = daily[daily["rain_in"].fillna(0) >= 0.10].copy()
    # Prevent adjacent rainy dates from appearing as separate storms. Keep the
    # highest-scoring trigger within a rolling 72-hour cluster.
    selected = []
    for _, row in candidates.sort_values("storm_score", ascending=False).iterrows():
        if all(abs((row.event_date - existing).days) > 2 for existing in selected):
            selected.append(row.event_date)
    significant = candidates[candidates.event_date.isin(selected)].sort_values("event_date", ascending=False)
    return daily.sort_values("storm_score", ascending=False), significant
