from __future__ import annotations

import numpy as np
import pandas as pd


def build_operations_summary(
    event_row: pd.Series,
    station_cycles: pd.DataFrame,
    ii_summary: dict,
    station_snapshot: pd.DataFrame | None = None,
) -> dict:
    """Create deterministic operator-facing findings from calculated metrics."""
    rain = pd.to_numeric(event_row.get("rain_in"), errors="coerce")
    peak = pd.to_numeric(event_row.get("plant_peak_mgd"), errors="coerce")
    lag = pd.to_numeric(ii_summary.get("lag_hr"), errors="coerce")
    excess = pd.to_numeric(ii_summary.get("excess_volume_mg"), errors="coerce")

    top_cycle = None
    cycle_count = np.nan
    if station_cycles is not None and not station_cycles.empty:
        r = station_cycles.sort_values("starts", ascending=False).iloc[0]
        top_cycle, cycle_count = r["asset_id"], r["starts"]

    top_wetwell = None
    top_level = np.nan
    if station_snapshot is not None and not station_snapshot.empty and "level_in" in station_snapshot:
        valid = station_snapshot.dropna(subset=["level_in"])
        if not valid.empty:
            r = valid.sort_values("level_in", ascending=False).iloc[0]
            top_wetwell, top_level = r["asset_id"], r["level_in"]

    findings: list[str] = []
    if pd.notna(rain):
        findings.append(f"The event delivered {rain:.2f} inches of rainfall.")
    else:
        findings.append("Measured rainfall is not available for this event; hydraulic response metrics remain valid.")
    if pd.notna(peak):
        findings.append(f"Plant influent peaked at {peak:.2f} MGD during the response window.")
    if pd.notna(lag):
        findings.append(f"The estimated wet-weather response peaked approximately {lag:.0f} hours after the event began.")
    if pd.notna(excess):
        findings.append(f"Estimated flow above the dry-weather baseline totaled {excess:.2f} MG.")
    if top_cycle:
        findings.append(f"{top_cycle} recorded the most pump starts ({int(cycle_count)}).")
    if top_wetwell:
        findings.append(f"{top_wetwell} had the highest wet-well reading ({top_level:.1f} in).")

    score = pd.to_numeric(event_row.get("storm_score"), errors="coerce")
    if pd.isna(score):
        condition = "Insufficient data"
    elif score >= 0.75:
        condition = "Major system response"
    elif score >= 0.50:
        condition = "Elevated system response"
    elif score >= 0.30:
        condition = "Moderate system response"
    else:
        condition = "Limited system response"

    return {"condition": condition, "findings": findings}
