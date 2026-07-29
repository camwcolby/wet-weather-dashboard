from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlantResponseEstimate:
    baseline_mgd: float
    response_threshold_mgd: float
    response_detected: bool
    response_onset: pd.Timestamp | pd.NaT
    response_lag_hr: float
    estimated_peak_time: pd.Timestamp | pd.NaT
    estimated_hours_to_peak: float
    estimated_peak_flow_mgd: float
    observed_peak_flow_mgd: float
    observed_peak_time: pd.Timestamp | pd.NaT
    analog_count: int
    confidence: str


def _clean_influent(influent: pd.DataFrame) -> pd.DataFrame:
    if (
        influent is None
        or influent.empty
        or "timestamp" not in influent.columns
        or "influent_total_mgd" not in influent.columns
    ):
        return pd.DataFrame(columns=["timestamp", "influent_total_mgd"])

    data = influent[["timestamp", "influent_total_mgd"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["influent_total_mgd"] = pd.to_numeric(
        data["influent_total_mgd"],
        errors="coerce",
    )
    return (
        data.dropna(subset=["timestamp", "influent_total_mgd"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _event_window(
    data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return data.loc[
        data["timestamp"].between(start, end, inclusive="both")
    ].copy()


def _baseline_and_threshold(
    event: pd.DataFrame,
    event_start: pd.Timestamp,
    baseline_hours: float,
    response_percent: float,
    response_minimum_mgd: float,
) -> tuple[float, float]:
    baseline_end = event_start + pd.Timedelta(hours=baseline_hours)
    baseline_values = event.loc[
        event["timestamp"] < baseline_end,
        "influent_total_mgd",
    ]

    if baseline_values.empty:
        baseline_values = event["influent_total_mgd"].head(60)

    baseline = float(baseline_values.median()) if not baseline_values.empty else np.nan
    threshold = (
        baseline + max(response_percent * baseline, response_minimum_mgd)
        if pd.notna(baseline)
        else np.nan
    )
    return baseline, threshold


def _detect_response_onset(
    event_to_date: pd.DataFrame,
    event_start: pd.Timestamp,
    threshold: float,
    baseline_hours: float,
    smoothing_minutes: int,
    sustained_minutes: int,
) -> pd.Timestamp | pd.NaT:
    if event_to_date.empty or pd.isna(threshold):
        return pd.NaT

    event = event_to_date.copy()
    smoothed = event["influent_total_mgd"].rolling(
        smoothing_minutes,
        min_periods=max(3, smoothing_minutes // 2),
    ).median()
    above = smoothed.ge(threshold)
    sustained = above.rolling(
        sustained_minutes,
        min_periods=sustained_minutes,
    ).sum().ge(sustained_minutes)

    eligible = sustained & (
        event["timestamp"]
        >= event_start + pd.Timedelta(hours=baseline_hours)
    )
    if not eligible.any():
        return pd.NaT

    first_sustained_index = eligible[eligible].index[0]
    onset_index_position = max(
        0,
        event.index.get_loc(first_sustained_index) - sustained_minutes + 1,
    )
    return pd.Timestamp(event.iloc[onset_index_position]["timestamp"])


def _contiguous_periods(data: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if data.empty:
        return []

    dates = pd.Series(data["timestamp"].dt.normalize().unique()).sort_values()
    if dates.empty:
        return []

    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = pd.Timestamp(dates.iloc[0])
    previous = start

    for value in dates.iloc[1:]:
        current = pd.Timestamp(value)
        if current - previous > pd.Timedelta(days=1):
            periods.append((start, previous + pd.Timedelta(days=1)))
            start = current
        previous = current

    periods.append((start, previous + pd.Timedelta(days=1)))
    return periods


def _build_analogs(
    data: pd.DataFrame,
    history: pd.DataFrame,
    selected_start: pd.Timestamp,
    baseline_hours: float,
    response_percent: float,
    response_minimum_mgd: float,
    smoothing_minutes: int,
    sustained_minutes: int,
) -> pd.DataFrame:
    history_data = history.copy() if history is not None else pd.DataFrame()
    if not history_data.empty and "date" in history_data.columns:
        history_data["date"] = pd.to_datetime(
            history_data["date"],
            errors="coerce",
        ).dt.normalize()
        history_data["rain_in"] = pd.to_numeric(
            history_data.get("rain_in"),
            errors="coerce",
        ).fillna(0)

    analog_rows: list[dict] = []
    for period_start, period_end in _contiguous_periods(data):
        if period_start >= selected_start:
            continue

        period = _event_window(data, period_start, period_end)
        if period.empty:
            continue

        baseline, threshold = _baseline_and_threshold(
            period,
            period_start,
            baseline_hours,
            response_percent,
            response_minimum_mgd,
        )
        onset = _detect_response_onset(
            period,
            period_start,
            threshold,
            baseline_hours,
            smoothing_minutes,
            sustained_minutes,
        )
        peak_idx = period["influent_total_mgd"].idxmax()
        peak_time = pd.Timestamp(period.loc[peak_idx, "timestamp"])
        peak_flow = float(period.loc[peak_idx, "influent_total_mgd"])

        if not history_data.empty:
            rain_end = period_end.normalize()
            rain_total = float(
                history_data.loc[
                    history_data["date"].between(
                        period_start.normalize(),
                        rain_end,
                    ),
                    "rain_in",
                ].sum()
            )
        else:
            rain_total = np.nan

        analog_rows.append(
            {
                "event_start": period_start,
                "baseline_mgd": baseline,
                "onset_lag_hr": (
                    (onset - period_start).total_seconds() / 3600
                    if pd.notna(onset)
                    else np.nan
                ),
                "onset_to_peak_hr": (
                    (peak_time - onset).total_seconds() / 3600
                    if pd.notna(onset) and peak_time >= onset
                    else np.nan
                ),
                "peak_lag_hr": (
                    peak_time - period_start
                ).total_seconds() / 3600,
                "peak_flow_mgd": peak_flow,
                "peak_multiple": (
                    peak_flow / baseline
                    if pd.notna(baseline) and baseline > 0
                    else np.nan
                ),
                "rain_total_in": rain_total,
            }
        )

    return pd.DataFrame(analog_rows)


def estimate_plant_response(
    influent: pd.DataFrame,
    history: pd.DataFrame,
    event_start: pd.Timestamp,
    as_of: pd.Timestamp,
    event_hours: int = 72,
    baseline_hours: float = 1.0,
    response_percent: float = 0.15,
    response_minimum_mgd: float = 0.25,
    smoothing_minutes: int = 10,
    sustained_minutes: int = 15,
) -> PlantResponseEstimate:
    """Estimate plant response using only information available at ``as_of``.

    Response onset is the first sustained increase above a pre-response
    baseline. Peak time and peak flow are estimated from earlier contiguous
    high-resolution telemetry periods. Historical periods after the selected
    event are never used, which prevents playback from looking into the future.
    """

    event_start = pd.Timestamp(event_start)
    as_of = pd.Timestamp(as_of)
    event_end = event_start + pd.Timedelta(hours=event_hours)
    effective_as_of = min(as_of, event_end)

    data = _clean_influent(influent)
    full_event = _event_window(data, event_start, event_end)
    event_to_date = _event_window(data, event_start, effective_as_of)

    baseline, threshold = _baseline_and_threshold(
        full_event if not full_event.empty else event_to_date,
        event_start,
        baseline_hours,
        response_percent,
        response_minimum_mgd,
    )
    onset = _detect_response_onset(
        event_to_date,
        event_start,
        threshold,
        baseline_hours,
        smoothing_minutes,
        sustained_minutes,
    )

    if event_to_date.empty:
        observed_peak_flow = np.nan
        observed_peak_time = pd.NaT
    else:
        observed_peak_idx = event_to_date["influent_total_mgd"].idxmax()
        observed_peak_flow = float(
            event_to_date.loc[observed_peak_idx, "influent_total_mgd"]
        )
        observed_peak_time = pd.Timestamp(
            event_to_date.loc[observed_peak_idx, "timestamp"]
        )

    analogs = _build_analogs(
        data,
        history,
        event_start,
        baseline_hours,
        response_percent,
        response_minimum_mgd,
        smoothing_minutes,
        sustained_minutes,
    )

    if not analogs.empty:
        current_rain = np.nan
        if history is not None and not history.empty and "date" in history.columns:
            history_current = history.copy()
            history_current["date"] = pd.to_datetime(
                history_current["date"],
                errors="coerce",
            ).dt.normalize()
            history_current["rain_in"] = pd.to_numeric(
                history_current.get("rain_in"),
                errors="coerce",
            ).fillna(0)
            current_rain = float(
                history_current.loc[
                    history_current["date"].between(
                        event_start.normalize(),
                        min(event_end, effective_as_of).normalize(),
                    ),
                    "rain_in",
                ].sum()
            )

        ranked = analogs.copy()
        if pd.notna(current_rain) and ranked["rain_total_in"].notna().any():
            ranked["distance"] = (
                ranked["rain_total_in"] - current_rain
            ).abs()
        else:
            ranked["distance"] = (
                ranked["baseline_mgd"] - baseline
            ).abs()
        ranked = ranked.sort_values("distance").head(5)
    else:
        ranked = analogs

    analog_count = len(ranked)
    if analog_count:
        if pd.notna(onset) and ranked["onset_to_peak_hr"].notna().any():
            estimated_peak_time = onset + pd.Timedelta(
                hours=float(ranked["onset_to_peak_hr"].median())
            )
        else:
            estimated_peak_time = event_start + pd.Timedelta(
                hours=float(ranked["peak_lag_hr"].median())
            )

        peak_multiple = float(ranked["peak_multiple"].median())
        if pd.notna(baseline) and pd.notna(peak_multiple):
            estimated_peak_flow = baseline * peak_multiple
        else:
            estimated_peak_flow = float(ranked["peak_flow_mgd"].median())

        if pd.notna(observed_peak_flow):
            estimated_peak_flow = max(
                estimated_peak_flow,
                observed_peak_flow,
            )

        if estimated_peak_time < effective_as_of:
            estimated_peak_time = max(
                effective_as_of,
                observed_peak_time,
            )
    else:
        estimated_peak_time = pd.NaT
        estimated_peak_flow = np.nan

    response_lag = (
        (onset - event_start).total_seconds() / 3600
        if pd.notna(onset)
        else np.nan
    )
    hours_to_peak = (
        (estimated_peak_time - effective_as_of).total_seconds() / 3600
        if pd.notna(estimated_peak_time)
        else np.nan
    )

    if analog_count >= 4:
        confidence = "Moderate"
    elif analog_count >= 2:
        confidence = "Preliminary"
    else:
        confidence = "Limited"

    return PlantResponseEstimate(
        baseline_mgd=baseline,
        response_threshold_mgd=threshold,
        response_detected=pd.notna(onset),
        response_onset=onset,
        response_lag_hr=response_lag,
        estimated_peak_time=estimated_peak_time,
        estimated_hours_to_peak=max(0.0, hours_to_peak) if pd.notna(hours_to_peak) else np.nan,
        estimated_peak_flow_mgd=estimated_peak_flow,
        observed_peak_flow_mgd=observed_peak_flow,
        observed_peak_time=observed_peak_time,
        analog_count=analog_count,
        confidence=confidence,
    )
