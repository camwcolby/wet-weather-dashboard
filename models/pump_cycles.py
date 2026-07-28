from __future__ import annotations

import numpy as np
import pandas as pd


def _status_to_binary(series: pd.Series) -> pd.Series:
    """Convert numeric/boolean pump status to a clean nullable 0/1 series."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isna(), (s > 0).astype(int))


def count_pump_cycles(
    collection: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    short_cycle_minutes: float = 10.0,
) -> pd.DataFrame:
    """Count starts, stops, completed cycles and runtimes for each pump.

    A start is a 0→1 transition and a stop is a 1→0 transition. A completed
    cycle is a start followed by the next stop within the selected window.
    Boundary states are retained as unmatched starts/stops rather than being
    forced into a cycle.
    """
    if collection is None or collection.empty:
        return pd.DataFrame()

    df = collection.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    df = df.dropna(subset=["timestamp", "asset_id"]).sort_values(["asset_id", "timestamp"])

    rows: list[dict] = []
    for asset_id, group in df.groupby("asset_id", sort=True):
        for pump_number, col in [(1, "pump1_status"), (2, "pump2_status")]:
            if col not in group.columns:
                continue
            p = group[["timestamp", col]].dropna().copy()
            if p.empty:
                continue
            p["state"] = _status_to_binary(p[col])
            p = p.dropna(subset=["state"])
            if p.empty:
                continue
            # Collapse repeated telemetry values before finding transitions.
            p = p.loc[p["state"].ne(p["state"].shift())].copy()
            p["previous"] = p["state"].shift()
            starts = p[(p["previous"] == 0) & (p["state"] == 1)]["timestamp"].tolist()
            stops = p[(p["previous"] == 1) & (p["state"] == 0)]["timestamp"].tolist()

            durations: list[float] = []
            stop_index = 0
            for start_time in starts:
                while stop_index < len(stops) and stops[stop_index] <= start_time:
                    stop_index += 1
                if stop_index < len(stops):
                    durations.append((stops[stop_index] - start_time).total_seconds() / 60.0)
                    stop_index += 1

            runtime_minutes = float(np.nansum(durations))
            period_hours = (
                (p["timestamp"].max() - p["timestamp"].min()).total_seconds() / 3600.0
                if len(p) > 1 else np.nan
            )
            rows.append({
                "asset_id": asset_id,
                "pump": f"Pump {pump_number}",
                "starts": len(starts),
                "stops": len(stops),
                "complete_cycles": len(durations),
                "unmatched_starts": max(len(starts) - len(durations), 0),
                "runtime_hr_from_cycles": runtime_minutes / 60.0,
                "median_cycle_min": float(np.nanmedian(durations)) if durations else np.nan,
                "mean_cycle_min": float(np.nanmean(durations)) if durations else np.nan,
                "short_cycles": int(sum(d < short_cycle_minutes for d in durations)),
                "starts_per_24h": (len(starts) / period_hours * 24.0) if period_hours and period_hours > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def station_cycle_summary(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles is None or cycles.empty:
        return pd.DataFrame()
    return (
        cycles.groupby("asset_id", as_index=False)
        .agg(
            starts=("starts", "sum"),
            stops=("stops", "sum"),
            complete_cycles=("complete_cycles", "sum"),
            short_cycles=("short_cycles", "sum"),
            runtime_hr_from_cycles=("runtime_hr_from_cycles", "sum"),
            median_cycle_min=("median_cycle_min", "median"),
        )
        .sort_values("starts", ascending=False)
    )


def daily_pump_starts(collection: pd.DataFrame) -> pd.DataFrame:
    """Count 0→1 transitions by station, pump, and calendar day.

    The prior status is computed before daily grouping, so a pump that turns on
    just after midnight is counted correctly when the previous sample was off.
    """
    if collection is None or collection.empty:
        return pd.DataFrame(columns=["date", "asset_id", "pump", "starts"])
    df = collection.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "asset_id"]).sort_values(["asset_id", "timestamp"])
    frames = []
    for pump_number, col in [(1, "pump1_status"), (2, "pump2_status")]:
        if col not in df.columns:
            continue
        part = df[["timestamp", "asset_id", col]].copy()
        part["state"] = _status_to_binary(part[col])
        part = part.dropna(subset=["state"])
        part["previous"] = part.groupby("asset_id")["state"].shift()
        part["start"] = ((part["previous"] == 0) & (part["state"] == 1)).astype(int)
        part["date"] = part["timestamp"].dt.normalize()
        summary = part.groupby(["date", "asset_id"], as_index=False)["start"].sum()
        summary["pump"] = f"Pump {pump_number}"
        summary = summary.rename(columns={"start": "starts"})
        frames.append(summary)
    if not frames:
        return pd.DataFrame(columns=["date", "asset_id", "pump", "starts"])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "asset_id", "pump"])
