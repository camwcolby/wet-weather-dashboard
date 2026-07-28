from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data
except ModuleNotFoundError:  # Allows loader unit tests outside Streamlit.
    def cache_data(**_kwargs):
        def decorator(func):
            return func
        return decorator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"


def _excel_datetime(series: pd.Series) -> pd.Series:
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
    "PS 1": (1, 5), "PS 3": (5, 9), "PS 4": (9, 13), "PS 5": (13, 17),
    "PS 6": (17, 22), "PS 9": (22, 26), "LS A": (26, 30),
}


@cache_data(show_spinner=False)
def load_collection() -> pd.DataFrame:
    path = RAW / "Collection_System 2026.xlsx"
    xl = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in [s for s in xl.sheet_names if s != "Template"]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if len(raw) < 3:
            continue
        dates = _excel_datetime(raw.iloc[2:, 0])
        for asset, (start, end) in COLLECTION_GROUPS.items():
            hdr = raw.iloc[1, start:end].astype(str).str.strip().tolist()
            vals = raw.iloc[2:, start:end].copy()
            vals.columns = hdr
            vals.insert(0, "timestamp", dates.values)
            vals.insert(1, "asset_id", asset)
            vals = vals.rename(columns={
                "Pmp1 Stat": "pump1_status", "Pmp2 Stat": "pump2_status",
                "Flow (gpm)": "flow_gpm", "Level (in)": "level_in",
                "Int. Level": "interceptor_level",
            })
            frames.append(vals)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    for c in ["pump1_status", "pump2_status", "flow_gpm", "level_in", "interceptor_level"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)


@cache_data(show_spinner=False)
def load_influent() -> pd.DataFrame:
    path = RAW / "Influent_Data_2026.xlsx"
    xl = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in [s for s in xl.sheet_names if s != "Template"]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if len(raw) < 3:
            continue
        df = raw.iloc[2:].copy()
        df.columns = [str(x).strip() for x in raw.iloc[1].tolist()]
        df = df.rename(columns={
            "Date/Time": "timestamp", "Wetwell Level(in)": "influent_wetwell_in",
            '8" Flow (mgd)': "influent_8_mgd", '16" Flow(mgd)': "influent_16_mgd",
            "Effluent Flow(mgd)": "effluent_flow_mgd",
        })
        df["timestamp"] = _excel_datetime(df["timestamp"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    for c in out.columns:
        if c != "timestamp":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["influent_total_mgd"] = out.get("influent_8_mgd", 0).fillna(0) + out.get("influent_16_mgd", 0).fillna(0)
    return out.sort_values("timestamp").reset_index(drop=True)


@cache_data(show_spinner=False)
def load_process_summary() -> pd.DataFrame:
    """Load the process workbook while safely combining mixed-type headers."""
    path = RAW / "Process Summary 2026.xlsx"
    xl = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet in [s for s in xl.sheet_names if s != "Template"]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if len(raw) < 7:
            continue
        labels1 = raw.iloc[4].ffill()
        labels2 = raw.iloc[5]
        cols: list[str] = []
        for index, (a, b) in enumerate(zip(labels1, labels2)):
            part_a = "" if pd.isna(a) else str(a).strip()
            part_b = "" if pd.isna(b) else str(b).strip()
            combined = " ".join(x for x in (part_a, part_b) if x).strip()
            cols.append(combined or f"column_{index}")
        # Make duplicate Excel headers deterministic.
        seen: dict[str, int] = {}
        unique_cols: list[str] = []
        for col in cols:
            seen[col] = seen.get(col, 0) + 1
            unique_cols.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
        df = raw.iloc[6:].copy()
        df.columns = unique_cols
        df = df.rename(columns={unique_cols[0]: "date"})
        df["date"] = _excel_datetime(df["date"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


@cache_data(show_spinner=False)
def load_historical_influent_rain() -> pd.DataFrame:
    """Load the local five-year daily rainfall and influent history.

    This is the authoritative source for historical event rainfall. External
    services remain useful for current forecasts, tides, and weather context.
    """
    path = RAW / "hull_5yr_influent_primary_wide.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "rain_in", "plant_flow_mgd", "plant_peak_mgd"])
    df = pd.read_csv(path)
    rename = {
        "DATESTAMP": "date",
        "rainfall (inches)": "rain_in",
        "influent flow (mgd)": "plant_flow_mgd",
        "max influent flow (mgd)": "plant_peak_mgd",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["rain_in", "plant_flow_mgd", "plant_peak_mgd"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["rain_in"] = df["rain_in"].fillna(0).clip(lower=0)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@cache_data(show_spinner=False)
def load_station_runtimes() -> pd.DataFrame:
    path = RAW / "Station Runtimes 2026.xlsx"
    xl = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    station_starts = {"PS 1": 1, "PS 3": 9, "PS 4": 17, "PS 5": 25, "PS 6": 33, "PS 9": 41, "LS A": 49, "PS D": 57}
    for sheet in [s for s in xl.sheet_names if s != "Template"]:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if len(raw) < 5:
            continue
        dates = _excel_datetime(raw.iloc[4:, 0])
        for asset, start in station_starts.items():
            frame = pd.DataFrame({
                "date": dates.values, "asset_id": asset,
                "pump1_runtime_hr": pd.to_numeric(raw.iloc[4:, start + 2], errors="coerce").values,
                "flow_kgal": pd.to_numeric(raw.iloc[4:, start + 3], errors="coerce").values,
                "pump2_runtime_hr": pd.to_numeric(raw.iloc[4:, start + 6], errors="coerce").values,
            })
            frame["total_runtime_hr"] = frame[["pump1_runtime_hr", "pump2_runtime_hr"]].sum(axis=1, min_count=1)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["date"]).sort_values(["date", "asset_id"]).reset_index(drop=True)


def latest_snapshot(collection: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    subset = collection[collection.timestamp <= as_of]
    if subset.empty:
        return pd.DataFrame()
    idx = subset.groupby("asset_id")["timestamp"].idxmax()
    return subset.loc[idx].copy()


@cache_data(show_spinner=False)
def load_asset_locations() -> pd.DataFrame:
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
            item = base.to_dict(); item["asset_id"] = asset_id
            if asset_id in raw.index:
                r = raw.loc[asset_id]
                item["lat"], item["lon"] = float(r["lat"]), float(r["lon"])
                if pd.notna(r.get("Address")):
                    item["address"] = f"{r['Address']}, Hull, MA"
                if pd.notna(r.get("Design capacity (gpm)")):
                    item["capacity_gpm"] = float(r["Design capacity (gpm)"])
            rows.append(item)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(ASSETS)
