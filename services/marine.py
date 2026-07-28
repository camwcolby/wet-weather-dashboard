from __future__ import annotations
import requests
import pandas as pd
import streamlit as st
from config.assets import HULL_LAT,HULL_LON

@st.cache_data(ttl=1800,show_spinner=False)
def marine_forecast():
    try:
        params={"latitude":HULL_LAT,"longitude":HULL_LON,"hourly":"wave_height,wave_direction,wave_period,wind_wave_height","length_unit":"imperial","timezone":"America/New_York","forecast_days":3}
        j=requests.get("https://marine-api.open-meteo.com/v1/marine",params=params,timeout=15).json()["hourly"]
        return pd.DataFrame(j).rename(columns={"time":"timestamp"}).assign(timestamp=lambda d:pd.to_datetime(d.timestamp))
    except Exception: return pd.DataFrame()
