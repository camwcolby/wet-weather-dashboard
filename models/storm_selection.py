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
    """Rank rainfall-triggered events using the following 24-hour response.

    Each candidate date is the rainfall trigger day. Hydraulic metrics use the
    trigger day plus the next calendar day so delayed I/I response at the WWTP
    is not missed. The selected response date is whichever of those two days
    has the higher plant influent flow.
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
        total_runtime_hr=("total_runtime_hr", "sum"),
        station_flow_kgal=("flow_kgal", "sum"),
    )

    start = min(plant.date.min(), rt.date.min())
    end = max(plant.date.max(), rt.date.max())
    daily = pd.DataFrame({"event_date": pd.date_range(start, end, freq="D")})

    if rain is not None and not rain.empty:
        rain_daily = rain.copy()
        rain_daily["event_date"] = pd.to_datetime(rain_daily["date"]).dt.normalize()
        rain_daily = rain_daily.groupby("event_date", as_index=False)["rain_in"].sum()
        daily = daily.merge(rain_daily, on="event_date", how="left")
    else:
        daily["rain_in"] = np.nan

    plant_idx = plant.set_index("date")["plant_flow_mgd"]
    runtime_idx = rt.set_index("date")["total_runtime_hr"]
    flow_idx = rt.set_index("date")["station_flow_kgal"]

    response_rows = []
    for event_date in daily["event_date"]:
        dates = [event_date, event_date + pd.Timedelta(days=1)]
        flows = plant_idx.reindex(dates)
        peak_flow = flows.max()
        response_date = flows.idxmax() if flows.notna().any() else event_date
        response_rows.append({
            "event_date": event_date,
            "response_date": response_date,
            "response_lag_hr": int((response_date - event_date).total_seconds() / 3600),
            "plant_peak_mgd": peak_flow,
            "total_runtime_48h": runtime_idx.reindex(dates).sum(min_count=1),
            "station_flow_48h_kgal": flow_idx.reindex(dates).sum(min_count=1),
        })
    response = pd.DataFrame(response_rows)
    daily = daily.merge(response, on="event_date", how="left")

    daily["rain_score"] = _scaled(daily["rain_in"])
    daily["plant_score"] = _scaled(daily["plant_peak_mgd"])
    daily["runtime_score"] = _scaled(daily["total_runtime_48h"])
    daily["station_flow_score"] = _scaled(daily["station_flow_48h_kgal"])
    daily["storm_score"] = (
        0.35 * daily["rain_score"].fillna(0)
        + 0.35 * daily["plant_score"].fillna(0)
        + 0.18 * daily["runtime_score"].fillna(0)
        + 0.12 * daily["station_flow_score"].fillna(0)
    )

    rainfall_available = daily["rain_in"].notna().any()
    if rainfall_available:
        candidates = daily[daily["rain_in"].fillna(0) >= 0.10]
    else:
        candidates = daily
    if candidates.empty:
        candidates = daily
    cutoff = candidates["storm_score"].quantile(.65)
    significant = candidates[candidates["storm_score"] >= cutoff]

    ranked = daily.sort_values("storm_score", ascending=False)
    significant = significant.sort_values("event_date", ascending=False)
    return ranked, significant
