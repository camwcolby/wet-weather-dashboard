from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def _excel_datetime(series):
    def convert(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (pd.Timestamp, datetime)):
            return pd.Timestamp(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and 1 <= float(value) <= 100000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
        return pd.to_datetime(value, errors="coerce")
    return series.apply(convert)


COLLECTION_GROUPS = {
    "PS 1": (1,5), "PS 3": (5,9), "PS 4": (9,13), "PS 5": (13,17),
    "PS 6": (17,22), "PS 9": (22,26), "LS A": (26,30)
}

@st.cache_data(show_spinner=False)
def load_collection():
    path = RAW / "Collection_System 2026.xlsx"
    xl = pd.ExcelFile(path)
    frames=[]
    for sheet in [s for s in xl.sheet_names if s != "Template"]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.empty: continue
        dates = _excel_datetime(raw.iloc[2:,0])
        for asset,(start,end) in COLLECTION_GROUPS.items():
            hdr = raw.iloc[1,start:end].astype(str).tolist()
            vals = raw.iloc[2:,start:end].copy()
            vals.columns = hdr
            vals.insert(0,"timestamp",dates.values)
            vals.insert(1,"asset_id",asset)
            rename={"Pmp1 Stat":"pump1_status","Pmp2 Stat":"pump2_status","Flow (gpm)":"flow_gpm","Level (in)":"level_in","Int. Level":"interceptor_level"}
            vals=vals.rename(columns=rename)
            frames.append(vals)
    out=pd.concat(frames,ignore_index=True)
    out=out.dropna(subset=["timestamp"])
    for c in ["pump1_status","pump2_status","flow_gpm","level_in","interceptor_level"]:
        if c in out: out[c]=pd.to_numeric(out[c],errors="coerce")
    return out.sort_values("timestamp")

@st.cache_data(show_spinner=False)
def load_influent():
    path=RAW/"Influent_Data_2026.xlsx"
    xl=pd.ExcelFile(path); frames=[]
    for sheet in [s for s in xl.sheet_names if s!="Template"]:
        raw=pd.read_excel(path,sheet_name=sheet,header=None)
        df=raw.iloc[2:].copy(); df.columns=[str(x) for x in raw.iloc[1].tolist()]
        df=df.rename(columns={"Date/Time":"timestamp","Wetwell Level(in)":"influent_wetwell_in","8\" Flow (mgd)":"influent_8_mgd","16\" Flow(mgd)":"influent_16_mgd","Effluent Flow(mgd)":"effluent_flow_mgd"})
        df["timestamp"]=_excel_datetime(df["timestamp"])
        frames.append(df)
    out=pd.concat(frames,ignore_index=True).dropna(subset=["timestamp"])
    for c in out.columns:
        if c!="timestamp": out[c]=pd.to_numeric(out[c],errors="coerce")
    out["influent_total_mgd"]=out.get("influent_8_mgd",0).fillna(0)+out.get("influent_16_mgd",0).fillna(0)
    return out.sort_values("timestamp")

@st.cache_data(show_spinner=False)
def load_process_summary():
    path=RAW/"Process Summary 2026.xlsx"; xl=pd.ExcelFile(path); frames=[]
    for sheet in [s for s in xl.sheet_names if s!="Template"]:
        raw=pd.read_excel(path,sheet_name=sheet,header=None)
        if len(raw)<7: continue
        labels1=raw.iloc[4].ffill().astype(str)
        labels2=raw.iloc[5].astype(str)
        cols=[]
        for a,b in zip(labels1,labels2):
            cols.append(
    f"{'' if pd.isna(a) else str(a)} {'' if pd.isna(b) else str(b)}".strip()
)
        df=raw.iloc[6:].copy(); df.columns=cols
        df=df.rename(columns={cols[0]:"date"})
        df["date"]=_excel_datetime(df["date"])
        frames.append(df)
    out=pd.concat(frames,ignore_index=True).dropna(subset=["date"])
    for c in out.columns:
        if c!="date": out[c]=pd.to_numeric(out[c],errors="coerce")
    return out.sort_values("date")

@st.cache_data(show_spinner=False)
def load_station_runtimes():
    path=RAW/"Station Runtimes 2026.xlsx"; xl=pd.ExcelFile(path); frames=[]
    station_starts={"PS 1":1,"PS 3":9,"PS 4":17,"PS 5":25,"PS 6":33,"PS 9":41,"LS A":49,"PS D":57}
    for sheet in [s for s in xl.sheet_names if s!="Template"]:
        raw=pd.read_excel(path,sheet_name=sheet,header=None)
        dates=_excel_datetime(raw.iloc[4:,0])
        for asset,start in station_starts.items():
            frame=pd.DataFrame({
                "date":dates.values,"asset_id":asset,
                "pump1_runtime_hr":pd.to_numeric(raw.iloc[4:,start+2],errors="coerce").values,
                "flow_kgal":pd.to_numeric(raw.iloc[4:,start+3],errors="coerce").values,
                "pump2_runtime_hr":pd.to_numeric(raw.iloc[4:,start+6],errors="coerce").values,
            })
            frame["total_runtime_hr"]=frame[["pump1_runtime_hr","pump2_runtime_hr"]].sum(axis=1,min_count=1)
            frames.append(frame)
    return pd.concat(frames,ignore_index=True).dropna(subset=["date"]).sort_values("date")

def latest_snapshot(collection, as_of):
    subset=collection[collection.timestamp<=as_of]
    if subset.empty: return pd.DataFrame()
    idx=subset.groupby("asset_id")["timestamp"].idxmax()
    return subset.loc[idx].copy()

REFERENCE = ROOT / "data" / "reference"

@st.cache_data(show_spinner=False)
def load_asset_locations():
    """Load authoritative asset coordinates and metadata from locations.xlsx.

    Falls back to config.assets.ASSETS if the workbook is missing or malformed.
    """
    from config.assets import ASSETS
    path = REFERENCE / "locations.xlsx"
    if not path.exists():
        return pd.DataFrame(ASSETS)
    try:
        raw = pd.read_excel(path)
        name_map = {
            "L.S.A.": "LS A", "P.S.#1": "PS 1", "P.S.#3": "PS 3",
            "P.S.#4": "PS 4", "P.S.#5": "PS 5", "P.S.#6": "PS 6",
            "P.S.#9": "PS 9", "P.S. D": "PS D", "WWTP": "WWTP",
        }
        fallback = pd.DataFrame(ASSETS).set_index("asset_id")
        coords = raw["Latitude / Longitude"].astype(str).str.split(",", n=1, expand=True)
        raw["lat"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
        raw["lon"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")
        raw["asset_id"] = raw["Name"].map(name_map)
        raw = raw.dropna(subset=["asset_id", "lat", "lon"]).set_index("asset_id")
        rows = []
        for asset_id, base in fallback.iterrows():
            item = base.to_dict()
            item["asset_id"] = asset_id
            if asset_id in raw.index:
                r = raw.loc[asset_id]
                item["lat"] = float(r["lat"])
                item["lon"] = float(r["lon"])
                if pd.notna(r.get("Address")):
                    item["address"] = f"{r['Address']}, Hull, MA"
                if pd.notna(r.get("Design capacity (gpm)")):
                    item["capacity_gpm"] = float(r["Design capacity (gpm)"])
            rows.append(item)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(ASSETS)
