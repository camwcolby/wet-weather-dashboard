from __future__ import annotations
import pandas as pd
import numpy as np


def _scaled(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.quantile(.10), s.quantile(.95)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def rank_events(process, runtimes, rain):
    """Rank trigger dates using rainfall plus the following 48-hour response.

    The rainfall day is retained as the event date. Plant and station metrics
    are searched through the next two days so Hull's delayed peak is captured.
    When rainfall is unavailable, the engine still ranks hydraulic response and
    labels rainfall as unavailable rather than inventing it.
    """
    proc = process.copy()
    proc["date"] = pd.to_datetime(proc["date"]).dt.normalize()
    inflow_col = next((c for c in proc.columns if c.startswith("Influent All Flows")), None)
    if inflow_col is None:
        candidates = [c for c in proc.columns if "Influent" in c and "Total" in c]
        inflow_col = candidates[0] if candidates else None
    plant = pd.DataFrame({"date": proc["date"]})
    plant["plant_flow_mgd"] = pd.to_numeric(proc[inflow_col], errors="coerce") if inflow_col else np.nan
    plant = plant.groupby("date", as_index=False)["plant_flow_mgd"].max()

    rt = runtimes.copy()
    rt["date"] = pd.to_datetime(rt["date"]).dt.normalize()
    rt = rt.groupby("date", as_index=False).agg(
        total_runtime_hr=("total_runtime_hr", "sum"), station_flow_kgal=("flow_kgal", "sum")
    )
    start, end = min(plant.date.min(), rt.date.min()), max(plant.date.max(), rt.date.max())
    daily = pd.DataFrame({"event_date": pd.date_range(start, end, freq="D")})

    if rain is not None and not rain.empty:
        rd = rain.copy(); rd["event_date"] = pd.to_datetime(rd["date"]).dt.normalize()
        rd = rd.groupby("event_date", as_index=False)["rain_in"].sum()
        daily = daily.merge(rd, on="event_date", how="left")
    else:
        daily["rain_in"] = np.nan

    plant_idx = plant.set_index("date")["plant_flow_mgd"]
    runtime_idx = rt.set_index("date")["total_runtime_hr"]
    flow_idx = rt.set_index("date")["station_flow_kgal"]
    rows = []
    for d in daily.event_date:
        response_dates = pd.date_range(d, d + pd.Timedelta(days=2), freq="D")
        flows = plant_idx.reindex(response_dates)
        response_date = flows.idxmax() if flows.notna().any() else d
        rows.append({
            "event_date": d, "response_date": response_date,
            "response_lag_hr": (response_date - d).total_seconds() / 3600,
            "plant_peak_mgd": flows.max(),
            "total_runtime_72h": runtime_idx.reindex(response_dates).sum(min_count=1),
            "station_flow_72h_kgal": flow_idx.reindex(response_dates).sum(min_count=1),
        })
    daily = daily.merge(pd.DataFrame(rows), on="event_date", how="left")
    # Backward-compatible name used on the current page.
    daily["total_runtime_48h"] = daily["total_runtime_72h"]

    daily["rain_score"] = _scaled(daily.rain_in)
    daily["plant_score"] = _scaled(daily.plant_peak_mgd)
    daily["runtime_score"] = _scaled(daily.total_runtime_72h)
    daily["station_flow_score"] = _scaled(daily.station_flow_72h_kgal)
    rainfall_available = daily.rain_in.notna().any()
    weights = (0.35, 0.35, 0.18, 0.12) if rainfall_available else (0.0, 0.50, 0.30, 0.20)
    daily["storm_score"] = (
        weights[0] * daily.rain_score.fillna(0) + weights[1] * daily.plant_score.fillna(0)
        + weights[2] * daily.runtime_score.fillna(0) + weights[3] * daily.station_flow_score.fillna(0)
    )
    daily["rainfall_available"] = rainfall_available

    candidates = daily[daily.rain_in.fillna(0) >= 0.10] if rainfall_available else daily
    if candidates.empty: candidates = daily
    significant = candidates[candidates.storm_score >= candidates.storm_score.quantile(.65)]
    return daily.sort_values("storm_score", ascending=False), significant.sort_values("event_date", ascending=False)
