from __future__ import annotations
import requests
import pandas as pd
import streamlit as st
from config.assets import HULL_LAT,HULL_LON

HEADERS={"User-Agent":"Inframark Hull Wet Weather Prototype contact: technical-services@inframark.com","Accept":"application/geo+json"}

@st.cache_data(ttl=1800,show_spinner=False)
def nws_bundle():
    try:
        p=requests.get(f"https://api.weather.gov/points/{HULL_LAT},{HULL_LON}",headers=HEADERS,timeout=12).json()["properties"]
        hourly=requests.get(p["forecastHourly"],headers=HEADERS,timeout=12).json()["properties"]["periods"]
        alerts=requests.get(f"https://api.weather.gov/alerts/active?point={HULL_LAT},{HULL_LON}",headers=HEADERS,timeout=12).json().get("features",[])
        return {"hourly":hourly,"alerts":alerts,"ok":True}
    except Exception as exc:
        return {"hourly":[],"alerts":[],"ok":False,"error":str(exc)}

@st.cache_data(ttl=86400,show_spinner=False)
def historical_precip(start_date,end_date):
    """Gap-free archive rainfall used for prototype event screening."""
    try:
        params={"latitude":HULL_LAT,"longitude":HULL_LON,"start_date":str(start_date),"end_date":str(end_date),"daily":"precipitation_sum","temperature_unit":"fahrenheit","precipitation_unit":"inch","timezone":"America/New_York"}
        j=requests.get("https://archive-api.open-meteo.com/v1/archive",params=params,timeout=20).json()
        return pd.DataFrame({"date":pd.to_datetime(j["daily"]["time"]),"rain_in":j["daily"]["precipitation_sum"]})
    except Exception:
        return pd.DataFrame(columns=["date","rain_in"])

@st.cache_data(ttl=1800, show_spinner=False)
def forecast_precipitation(days: int = 7) -> pd.DataFrame:
    """Hourly forecast precipitation for Hull from Open-Meteo.

    NWS remains the official alert/forecast narrative source in the app. This
    endpoint supplies a convenient numeric precipitation series for charts.
    """
    try:
        params = {
            "latitude": HULL_LAT,
            "longitude": HULL_LON,
            "hourly": "precipitation,precipitation_probability,rain,showers,snowfall",
            "forecast_days": int(days),
            "precipitation_unit": "inch",
            "timezone": "America/New_York",
        }
        j = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20).json()
        h = j["hourly"]
        return pd.DataFrame({
            "timestamp": pd.to_datetime(h["time"]),
            "forecast_precip_in": pd.to_numeric(h["precipitation"], errors="coerce"),
            "precip_probability_pct": pd.to_numeric(h["precipitation_probability"], errors="coerce"),
        })
    except Exception:
        return pd.DataFrame(columns=["timestamp", "forecast_precip_in", "precip_probability_pct"])


@st.cache_data(ttl=3600, show_spinner=False)
def recent_actual_precipitation(days: int = 10) -> pd.DataFrame:
    """Recent observed/model-assimilated hourly precipitation for comparison."""
    try:
        end = pd.Timestamp.now(tz="America/New_York").date()
        start = end - pd.Timedelta(days=int(days))
        params = {
            "latitude": HULL_LAT,
            "longitude": HULL_LON,
            "start_date": str(start),
            "end_date": str(end),
            "hourly": "precipitation",
            "precipitation_unit": "inch",
            "timezone": "America/New_York",
        }
        j = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20).json()
        h = j["hourly"]
        return pd.DataFrame({
            "timestamp": pd.to_datetime(h["time"]),
            "actual_precip_in": pd.to_numeric(h["precipitation"], errors="coerce"),
        })
    except Exception:
        return pd.DataFrame(columns=["timestamp", "actual_precip_in"])
