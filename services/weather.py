from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd
import streamlit as st
from config.assets import HULL_LAT, HULL_LON

HEADERS = {
    "User-Agent": "Inframark Hull Wet Weather Prototype technical-services@inframark.com",
    "Accept": "application/geo+json",
}
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "external_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _request_json(url: str, params: dict, timeout: int = 25) -> dict:
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload.get("reason") or payload.get("error"))
    return payload


@st.cache_data(ttl=1800, show_spinner=False)
def nws_bundle():
    try:
        p = requests.get(
            f"https://api.weather.gov/points/{HULL_LAT},{HULL_LON}", headers=HEADERS, timeout=12
        ).json()["properties"]
        hourly = requests.get(p["forecastHourly"], headers=HEADERS, timeout=12).json()["properties"]["periods"]
        alerts = requests.get(
            f"https://api.weather.gov/alerts/active?point={HULL_LAT},{HULL_LON}", headers=HEADERS, timeout=12
        ).json().get("features", [])
        return {"hourly": hourly, "alerts": alerts, "ok": True}
    except Exception as exc:
        return {"hourly": [], "alerts": [], "ok": False, "error": str(exc)}


def _fetch_daily_source(start_date, end_date, url: str) -> pd.DataFrame:
    params = {
        "latitude": HULL_LAT,
        "longitude": HULL_LON,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "precipitation_sum",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }
    j = _request_json(url, params)
    return pd.DataFrame({
        "date": pd.to_datetime(j["daily"]["time"]),
        "rain_in": pd.to_numeric(j["daily"]["precipitation_sum"], errors="coerce"),
    })


@st.cache_data(ttl=86400, show_spinner=False)
def historical_precip(start_date, end_date):
    """Daily rainfall with archive, historical-forecast, and local-cache fallbacks."""
    start_date, end_date = pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
    cache_path = CACHE_DIR / "hull_daily_rainfall.csv"
    sources = [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
    ]
    frames = []
    # Monthly chunks prevent one unavailable date from blanking the full year.
    chunk_starts = pd.date_range(start_date, end_date, freq="MS").tolist()
    if not chunk_starts or chunk_starts[0].date() != start_date:
        chunk_starts.insert(0, pd.Timestamp(start_date))
    for chunk_start in chunk_starts:
        chunk_end = min((chunk_start + pd.offsets.MonthEnd(0)).date(), end_date)
        chunk_start_date = max(chunk_start.date(), start_date)
        fetched = pd.DataFrame()
        for source in sources:
            try:
                fetched = _fetch_daily_source(chunk_start_date, chunk_end, source)
                if not fetched.empty:
                    break
            except Exception:
                continue
        if not fetched.empty:
            frames.append(fetched)
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
        try:
            existing = pd.read_csv(cache_path, parse_dates=["date"]) if cache_path.exists() else pd.DataFrame()
            combined = pd.concat([existing, out], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
            combined.to_csv(cache_path, index=False)
        except Exception:
            pass
        return out[(out.date.dt.date >= start_date) & (out.date.dt.date <= end_date)]
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached[(cached.date.dt.date >= start_date) & (cached.date.dt.date <= end_date)]
    return pd.DataFrame(columns=["date", "rain_in"])


@st.cache_data(ttl=86400, show_spinner=False)
def historical_hourly_precip(start_date, end_date):
    """Hourly event rainfall, with an on-disk cache for repeatable demos."""
    cache_path = CACHE_DIR / "hull_hourly_rainfall.csv"
    params = {
        "latitude": HULL_LAT,
        "longitude": HULL_LON,
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "hourly": "precipitation",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }
    for url in [
        "https://archive-api.open-meteo.com/v1/archive",
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
    ]:
        try:
            j = _request_json(url, params)
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(j["hourly"]["time"]),
                "actual_precip_in": pd.to_numeric(j["hourly"]["precipitation"], errors="coerce"),
            })
            existing = pd.read_csv(cache_path, parse_dates=["timestamp"]) if cache_path.exists() else pd.DataFrame()
            pd.concat([existing, out], ignore_index=True).drop_duplicates("timestamp", keep="last").sort_values("timestamp").to_csv(cache_path, index=False)
            return out
        except Exception:
            continue
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
        s, e = pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return cached[(cached.timestamp >= s) & (cached.timestamp < e)]
    return pd.DataFrame(columns=["timestamp", "actual_precip_in"])


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_precipitation(days: int = 7) -> pd.DataFrame:
    try:
        params = {
            "latitude": HULL_LAT, "longitude": HULL_LON,
            "hourly": "precipitation,precipitation_probability",
            "forecast_days": int(days), "precipitation_unit": "inch",
            "timezone": "America/New_York",
        }
        j = _request_json("https://api.open-meteo.com/v1/forecast", params)
        return pd.DataFrame({
            "timestamp": pd.to_datetime(j["hourly"]["time"]),
            "forecast_precip_in": pd.to_numeric(j["hourly"]["precipitation"], errors="coerce"),
            "precip_probability_pct": pd.to_numeric(j["hourly"]["precipitation_probability"], errors="coerce"),
        })
    except Exception:
        return pd.DataFrame(columns=["timestamp", "forecast_precip_in", "precip_probability_pct"])


@st.cache_data(ttl=3600, show_spinner=False)
def recent_actual_precipitation(days: int = 10) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="America/New_York").date()
    start = end - pd.Timedelta(days=int(days))
    return historical_hourly_precip(start, end)
