from __future__ import annotations
import requests
import pandas as pd
import streamlit as st
from datetime import date,timedelta
from config.assets import NOAA_TIDE_STATION

BASE="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

def _call(params):
    params.update({"station":NOAA_TIDE_STATION,"units":"english","time_zone":"lst_ldt","format":"json","application":"InframarkHullPrototype"})
    return requests.get(BASE,params=params,timeout=15).json()

@st.cache_data(ttl=1800,show_spinner=False)
def tide_predictions(days=3):
    try:
        begin=date.today().strftime("%Y%m%d"); end=(date.today()+timedelta(days=days)).strftime("%Y%m%d")
        j=_call({"product":"predictions","datum":"MLLW","interval":"hilo","begin_date":begin,"end_date":end})
        df=pd.DataFrame(j.get("predictions",[])); df["t"]=pd.to_datetime(df["t"]); df["v"]=pd.to_numeric(df["v"])
        return df
    except Exception: return pd.DataFrame(columns=["t","v","type"])

@st.cache_data(ttl=86400,show_spinner=False)
def historical_tides(start_date,end_date):
    try:
        j=_call({"product":"predictions","datum":"MLLW","interval":"h","begin_date":pd.Timestamp(start_date).strftime("%Y%m%d"),"end_date":pd.Timestamp(end_date).strftime("%Y%m%d")})
        df=pd.DataFrame(j.get("predictions",[])); df["t"]=pd.to_datetime(df["t"]); df["v"]=pd.to_numeric(df["v"])
        return df
    except Exception: return pd.DataFrame(columns=["t","v"])
