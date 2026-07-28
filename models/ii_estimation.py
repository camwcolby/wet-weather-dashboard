from __future__ import annotations

import numpy as np
import pandas as pd


def build_dry_weather_baseline(
    influent: pd.DataFrame,
    rain: pd.DataFrame,
    dry_threshold_in: float = 0.05,
    antecedent_days: int = 2,
) -> pd.DataFrame:
    """Estimate expected sanitary flow by hour-of-week using dry periods.

    A timestamp is dry when rainfall on that day and the preceding antecedent
    days remains below the threshold. Median flow by hour-of-week is used to
    reduce sensitivity to outliers and brief process anomalies.
    """
    if influent is None or influent.empty:
        return pd.DataFrame()
    df = influent[["timestamp", "influent_total_mgd"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "influent_total_mgd"])
    df["date"] = df["timestamp"].dt.normalize()

    if rain is not None and not rain.empty:
        r = rain.copy()
        r["date"] = pd.to_datetime(r["date"], errors="coerce").dt.normalize()
        r = r.groupby("date", as_index=False)["rain_in"].sum().sort_values("date")
        full = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
        full = full.merge(r, on="date", how="left").fillna({"rain_in": 0.0})
        rolling_rain = full["rain_in"].rolling(antecedent_days + 1, min_periods=1).sum()
        full["is_dry"] = rolling_rain <= dry_threshold_in
        df = df.merge(full[["date", "is_dry"]], on="date", how="left")
    else:
        df["is_dry"] = True

    dry = df[df["is_dry"].fillna(False)].copy()
    if dry.empty:
        dry = df.copy()
    dry["hour_of_week"] = dry["timestamp"].dt.dayofweek * 24 + dry["timestamp"].dt.hour
    baseline = (
        dry.groupby("hour_of_week", as_index=False)["influent_total_mgd"]
        .median()
        .rename(columns={"influent_total_mgd": "baseline_mgd"})
    )
    # Fill missing hours with the overall dry-weather median.
    complete = pd.DataFrame({"hour_of_week": np.arange(168)})
    complete = complete.merge(baseline, on="hour_of_week", how="left")
    complete["baseline_mgd"] = complete["baseline_mgd"].fillna(dry["influent_total_mgd"].median())
    return complete


def estimate_ii_timeseries(
    influent: pd.DataFrame,
    baseline: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Estimate wet-weather-derived flow above the dry-weather baseline."""
    if influent is None or influent.empty or baseline is None or baseline.empty:
        return pd.DataFrame()
    df = influent[["timestamp", "influent_total_mgd"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "influent_total_mgd"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if df.empty:
        return df
    df["hour_of_week"] = df["timestamp"].dt.dayofweek * 24 + df["timestamp"].dt.hour
    df = df.merge(baseline, on="hour_of_week", how="left")
    df["excess_mgd"] = (df["influent_total_mgd"] - df["baseline_mgd"]).clip(lower=0)
    df = df.sort_values("timestamp")
    delta_hr = df["timestamp"].diff().dt.total_seconds().div(3600)
    typical = delta_hr[(delta_hr > 0) & (delta_hr < 6)].median()
    df["interval_hr"] = delta_hr.fillna(typical if pd.notna(typical) else 1.0).clip(lower=0, upper=6)
    df["excess_volume_mg"] = df["excess_mgd"] * df["interval_hr"] / 24.0
    return df


def summarize_ii_event(
    ii: pd.DataFrame,
    rain_total_in: float | None = None,
    event_start: pd.Timestamp | None = None,
) -> dict:
    if ii is None or ii.empty:
        return {
            "excess_volume_mg": np.nan,
            "peak_excess_mgd": np.nan,
            "peak_observed_mgd": np.nan,
            "peak_time": pd.NaT,
            "lag_hr": np.nan,
            "mg_per_in": np.nan,
        }
    peak_idx = ii["excess_mgd"].idxmax()
    peak_time = ii.loc[peak_idx, "timestamp"]
    lag_hr = (
        (peak_time - pd.Timestamp(event_start)).total_seconds() / 3600.0
        if event_start is not None else np.nan
    )
    volume = ii["excess_volume_mg"].sum()
    return {
        "excess_volume_mg": float(volume),
        "peak_excess_mgd": float(ii["excess_mgd"].max()),
        "peak_observed_mgd": float(ii["influent_total_mgd"].max()),
        "peak_time": peak_time,
        "lag_hr": float(lag_hr) if pd.notna(lag_hr) else np.nan,
        "mg_per_in": float(volume / rain_total_in) if rain_total_in and rain_total_in > 0 else np.nan,
    }
