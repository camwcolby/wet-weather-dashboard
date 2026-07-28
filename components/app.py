from __future__ import annotations

from datetime import timedelta
from html import escape

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Hull Wet Weather Operations",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from components.header import render_header
from components.style import inject_css
from config.theme import AMBER, BLUE, GREEN, LIME, MUTED_BLUE, NAVY, RED
from models.ii_estimation import (
    build_dry_weather_baseline,
    estimate_ii_timeseries,
    summarize_ii_event,
)
from models.operations_summary import build_operations_summary
from models.pump_cycles import count_pump_cycles, station_cycle_summary
from models.storm_selection import rank_events
from services.data_loader import (
    load_asset_locations,
    load_collection,
    load_historical_influent_rain,
    load_influent,
    load_process_summary,
    load_station_runtimes,
    station_flow_snapshot,
)
from services.marine import marine_forecast
from services.tides import historical_tides, tide_predictions
from services.weather import nws_bundle
from utils.formatting import fmt, status_from_utilization
from services.radar import nws_radar_layer

EVENT_WINDOW_HOURS = 72
WET_WELL_REFERENCE_DEPTH_IN = 84.0
SANITARY_TELEMETRY_STATIONS = 7


def exact_snapshot(collection: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return telemetry only for the exact playback minute.

    This function intentionally does not carry values forward, search backward,
    or substitute the nearest available timestamp. If no record exists at the
    selected playback minute, the result is empty and the dashboard displays a
    data-availability message.
    """
    if collection is None or collection.empty:
        return pd.DataFrame()

    target = pd.Timestamp(as_of).floor("min")
    data = collection.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.floor("min")
    snapshot = data.loc[data["timestamp"] == target].copy()

    if snapshot.empty:
        return pd.DataFrame()

    # Keep one record per station if a source workbook contains duplicate rows
    # at the same exact minute. This is deduplication, not date substitution.
    return (
        snapshot.sort_values(["asset_id", "timestamp"])
        .drop_duplicates(subset=["asset_id"], keep="last")
        .reset_index(drop=True)
    )


def normalized_dates(frame: pd.DataFrame, column: str) -> set[pd.Timestamp]:
    """Return exact calendar dates represented in a data source."""
    if frame is None or frame.empty or column not in frame.columns:
        return set()
    values = pd.to_datetime(frame[column], errors="coerce").dropna().dt.normalize()
    return set(values.tolist())


def score_is_complete(event_row: pd.Series) -> bool:
    """A composite storm score is displayed only when all four inputs exist."""
    required = [
        "rain_in",
        "plant_peak_mgd",
        "total_runtime_72h",
        "station_flow_72h_kgal",
    ]
    return all(pd.notna(pd.to_numeric(event_row.get(col), errors="coerce")) for col in required)


def score_severity(score: float | int | None, complete: bool) -> tuple[str, str]:
    """Use one consistent set of score bands across the operator page."""
    if not complete or pd.isna(score):
        return "DATA GAP", "#98A5B3"
    if score >= 0.75:
        return "ALARM", RED
    if score >= 0.50:
        return "ELEVATED", AMBER
    if score >= 0.30:
        return "WATCH", LIME
    return "NORMAL", GREEN


def display_value(value, digits: int = 1, suffix: str = "", unavailable: str = "Unavailable") -> str:
    """Format a value without converting missing data into zero."""
    numeric = pd.to_numeric(value, errors="coerce")
    return fmt(numeric, digits, suffix) if pd.notna(numeric) else unavailable


def selected_asset_from_map(
    map_event: dict | None,
    assets: pd.DataFrame,
) -> str | None:
    """Resolve a clicked Folium marker to an asset ID."""

    if not map_event or assets is None or assets.empty:
        return None

    tooltip = map_event.get("last_object_clicked_tooltip")
    if tooltip:
        tooltip_text = str(tooltip).strip()

        asset_match = assets.loc[
            assets["asset_id"].astype(str).eq(tooltip_text)
        ]
        if not asset_match.empty:
            return str(asset_match.iloc[0]["asset_id"])

        name_match = assets.loc[
            assets["display_name"].astype(str).eq(tooltip_text)
        ]
        if not name_match.empty:
            return str(name_match.iloc[0]["asset_id"])

    clicked = map_event.get("last_object_clicked")
    if not isinstance(clicked, dict):
        return None

    clicked_lat = pd.to_numeric(clicked.get("lat"), errors="coerce")
    clicked_lon = pd.to_numeric(clicked.get("lng"), errors="coerce")

    if pd.isna(clicked_lat) or pd.isna(clicked_lon):
        return None

    candidates = assets.copy()
    candidates["_lat"] = pd.to_numeric(candidates["lat"], errors="coerce")
    candidates["_lon"] = pd.to_numeric(candidates["lon"], errors="coerce")
    candidates = candidates.dropna(subset=["_lat", "_lon"])

    if candidates.empty:
        return None

    candidates["_distance_sq"] = (
        (candidates["_lat"] - clicked_lat) ** 2
        + (candidates["_lon"] - clicked_lon) ** 2
    )

    closest = candidates.sort_values("_distance_sq").iloc[0]

    # About 55 metres at Hull's latitude. This prevents ordinary map clicks
    # from being mistaken for station-marker clicks.
    if float(closest["_distance_sq"]) > 0.0005**2:
        return None

    return str(closest["asset_id"])


inject_css()

with st.spinner("Loading 2026 operating data..."):
    collection = load_collection()
    influent = load_influent()
    process = load_process_summary()
    runtimes = load_station_runtimes()
    history = load_historical_influent_rain()

    if collection.empty or history.empty:
        st.error("Required collection-system or rainfall/influent history data could not be loaded.")
        st.stop()

    min_date = max(collection["timestamp"].min().date(), history["date"].min().date())
    max_date = min(collection["timestamp"].max().date(), history["date"].max().date())

    history_2026 = history.loc[
        (history["date"].dt.date >= min_date)
        & (history["date"].dt.date <= max_date)
    ].copy()

    rain = history_2026[["date", "rain_in"]].copy()
    ranked, significant = rank_events(history_2026, runtimes)

    collection_dates = normalized_dates(collection, "timestamp")
    influent_dates = normalized_dates(influent, "timestamp")
    history_dates = normalized_dates(history_2026, "date")
    runtime_dates = normalized_dates(runtimes, "date")

    # Dates used for automatic defaults must exist in every core data source.
    # Custom dates remain available, but missing sources display as unavailable.
    common_dates = collection_dates & influent_dates & history_dates & runtime_dates

    significant_common = significant.loc[
        significant["event_date"].dt.normalize().isin(common_dates)
    ].copy()

    ranked_common = ranked.loc[
        ranked["event_date"].dt.normalize().isin(common_dates)
    ].copy()

if not significant_common.empty:
    latest_storm = pd.Timestamp(significant_common.iloc[0]["event_date"])
elif not ranked_common.empty:
    latest_storm = pd.Timestamp(ranked_common.iloc[0]["event_date"])
elif common_dates:
    latest_storm = max(common_dates)
else:
    latest_storm = pd.Timestamp(max_date)

mode = st.sidebar.radio(
    "Operating view",
    ["Latest significant storm", "Latest common-data date", "Custom date"],
    index=0,
)

if mode == "Latest significant storm":
    selected_day = latest_storm
elif mode == "Latest common-data date":
    selected_day = max(common_dates) if common_dates else pd.Timestamp(max_date)
else:
    selected_day = pd.Timestamp(
        st.sidebar.date_input(
            "Date",
            value=latest_storm.date(),
            min_value=min_date,
            max_value=max_date,
        )
    )

selected_day = pd.Timestamp(selected_day).normalize()
row_match = ranked.loc[ranked["event_date"].dt.normalize() == selected_day]
row = row_match.iloc[0] if not row_match.empty else pd.Series(dtype="object")

response_value = row.get("response_date", pd.NaT)
response_day = (
    pd.Timestamp(response_value).normalize()
    if pd.notna(response_value)
    else pd.NaT
)

event_start = selected_day
event_end = event_start + pd.Timedelta(hours=EVENT_WINDOW_HOURS)

playback_hour = st.sidebar.slider(
    "Storm playback hour",
    min_value=0,
    max_value=EVENT_WINDOW_HOURS,
    value=48,
    step=1,
)
as_of = event_start + pd.Timedelta(hours=playback_hour)

response_label = response_day.strftime("%b %d, %Y") if pd.notna(response_day) else "Unavailable"
render_header(
    f"Historical Playback | Trigger {selected_day:%b %d, %Y} | Response {response_label}"
)

control_cols = st.columns([1.4, 3.6, 1.0])
with control_cols[0]:
    st.page_link(
        "pages/2_Wet_Weather_Analytics.py",
        label="Open Wet Weather Analytics",
        icon="🌧️",
    )
with control_cols[1]:
    st.caption(
        f"Playback time: **{as_of:%a %b %d, %Y at %I:%M %p}** | "
        f"No nearest-date or carry-forward substitution is permitted."
    )
with control_cols[2]:
    st.caption("Use the sidebar slider to replay the event")

source_availability = {
    "Rain/plant daily history": selected_day in history_dates,
    "Collection telemetry": selected_day in collection_dates,
    "Influent telemetry": selected_day in influent_dates,
    "Station runtime": selected_day in runtime_dates,
}
missing_sources = [name for name, available in source_availability.items() if not available]
if missing_sources:
    st.warning(
        "Selected date is missing: " + ", ".join(missing_sources) + ". "
        "Those widgets remain unavailable; no other date is substituted."
    )

rain_val = pd.to_numeric(row.get("rain_in"), errors="coerce")
plant_flow = pd.to_numeric(row.get("plant_peak_mgd"), errors="coerce")
raw_storm_score = pd.to_numeric(row.get("storm_score"), errors="coerce")
response_lag = pd.to_numeric(row.get("response_lag_hr"), errors="coerce")
score_complete = score_is_complete(row)
storm_score = raw_storm_score if score_complete else np.nan

# Exact-minute collection snapshot. Empty means genuinely unavailable.
snap = exact_snapshot(collection, as_of)
telemetry_available = not snap.empty

max_level = (
    pd.to_numeric(snap.get("level_in"), errors="coerce").max()
    if telemetry_available and "level_in" in snap.columns
    else np.nan
)

if telemetry_available:
    pump1 = pd.to_numeric(snap.get("pump1_status"), errors="coerce").fillna(0)
    pump2 = pd.to_numeric(snap.get("pump2_status"), errors="coerce").fillna(0)
    running = int(((pump1 + pump2) > 0).sum())
    reporting_stations = int(snap["asset_id"].nunique())
    running_text = f"{running} / {reporting_stations}"
else:
    running = np.nan
    reporting_stations = 0
    running_text = "Unavailable"

severity, sev_color = score_severity(storm_score, score_complete)
rain_text = display_value(rain_val, 2, " in", "Unavailable")
score_text = display_value(storm_score * 100 if pd.notna(storm_score) else np.nan, 0, "%", "Incomplete")
lag_text = display_value(response_lag, 0, " hr", "Unavailable")

st.markdown(
    f'<div class="status-strip" style="border-left-color:{sev_color}">'
    f'<b style="color:{NAVY}">{severity} WET WEATHER STATUS</b>'
    f' &nbsp; Storm score {score_text} | Rainfall {rain_text} | '
    f'plant response lag {lag_text} | Stations running {running_text}</div>',
    unsafe_allow_html=True,
)

kpis = st.columns(6)
items = [
    ("Rainfall trigger", rain_text, "Selected trigger date"),
    ("Peak plant influent", display_value(plant_flow, 2, " MGD"), "Trigger through +48 hr"),
    ("Highest wet well", display_value(max_level, 1, " in"), "Exact playback minute"),
    ("Stations running", running_text, "Exact playback minute"),
    (
        "System runtime",
        display_value(row.get("total_runtime_72h"), 1, " hr"),
        "Trigger plus two calendar days",
    ),
    ("Storm score", score_text, "Relative composite; complete inputs only"),
]

for column, (label, value, subtitle) in zip(kpis, items):
    column.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("How the dashboard scores are calculated"):
    st.markdown(
        """
**Component scores are relative, not regulatory thresholds.** For rainfall, plant peak flow,
station runtime, and station flow, the event engine identifies the 10th and 95th percentiles
across the evaluated daily dataset. Each value is converted to a 0–1 score using:

`component score = (event value − 10th percentile) / (95th percentile − 10th percentile)`

Values below the 10th percentile are clipped to 0; values above the 95th percentile are clipped
to 1.

**Storm score** is the weighted composite:

- 38% rainfall score
- 34% plant peak-influent score, using the trigger date and following two days
- 18% combined station-runtime score over the same three calendar dates
- 10% combined station-flow score over the same three calendar dates

The displayed storm score is suppressed as **Incomplete** unless all four raw inputs are present.
The score is intended to rank events within this dataset. It is not a probability of failure,
percent capacity, compliance risk, or calibrated hydraulic risk.

**Wet-weather status bands** use the composite score consistently:

- Normal: below 30%
- Watch: 30% to less than 50%
- Elevated: 50% to less than 75%
- Alarm: 75% or greater

**Pump-station map status** is separate from the storm score. It currently divides wet-well level
by a provisional 84-inch reference depth: Normal below 65%, Watch at 65–80%, Warning at 80–95%,
and Alarm at 95% or greater. This should be replaced with station-specific operating and alarm
levels when those thresholds are available.

**I/I response factor** is not a score. It is estimated excess influent volume divided by event
rainfall, expressed as MG per inch. Estimated excess flow is observed influent above a dry-weather
hour-of-week median baseline, clipped at zero.
        """
    )

assets = load_asset_locations()
measurement_columns = [
    "flow_gpm",
    "level_in",
    "pump1_status",
    "pump2_status",
    "interceptor_level",
]

if telemetry_available:
    available_columns = [
        col for col in ["asset_id", *measurement_columns] if col in snap.columns
    ]
    assets = assets.merge(snap[available_columns], on="asset_id", how="left")
else:
    for column in measurement_columns:
        assets[column] = np.nan

assets["utilization"] = (
    pd.to_numeric(assets["level_in"], errors="coerce") / WET_WELL_REFERENCE_DEPTH_IN
).clip(0, 1.2)
assets["status"] = assets["utilization"].apply(status_from_utilization)
assets.loc[assets["asset_type"] == "Treatment Plant", "status"] = "Plant"

colors = {
    "Normal": GREEN,
    "Watch": LIME,
    "Warning": AMBER,
    "Alarm": RED,
    "No Data": "#98A5B3",
    "Plant": NAVY,
}

radar_layer = nws_radar_layer()

radar_toggle_col, opacity_col, radar_status_col = st.columns(
    [1.2, 1.5, 3.3]
)

with radar_toggle_col:
    show_radar = st.toggle(
        "NWS radar",
        value=True,
        help=(
            "Overlay the current NOAA/NWS MRMS quality-controlled "
            "base-reflectivity mosaic."
        ),
    )

with opacity_col:
    radar_opacity = st.slider(
        "Radar opacity",
        min_value=0.10,
        max_value=0.80,
        value=0.40,
        step=0.05,
        disabled=not show_radar,
    )

with radar_status_col:
    st.caption(
        "Radar is live/current NOAA/NWS MRMS data. Pump-station values remain "
        f"historical at **{as_of:%b %d, %Y %I:%M %p}**. "
        "A clear radar layer may simply mean no precipitation is detected."
    )

left, right = st.columns([2.45, 1], gap="medium")

selected_asset = str(
    st.session_state.get("selected_asset", "PS 3")
)

with left:
    hull_map = folium.Map(
        location=[42.286, -70.882],
        zoom_start=12,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street map",
        control=True,
        show=True,
    ).add_to(hull_map)

    folium.TileLayer(
        tiles=(
            "https://{s}.basemaps.cartocdn.com/light_all/"
            "{z}/{x}/{y}{r}.png"
        ),
        attr="© OpenStreetMap contributors © CARTO",
        name="Light map",
        control=True,
        show=False,
    ).add_to(hull_map)

    if show_radar:
        folium.raster_layers.WmsTileLayer(
            url=radar_layer["wms_url"],
            layers=radar_layer["layers"],
            styles=radar_layer.get("styles", ""),
            fmt=radar_layer.get("format", "image/png"),
            transparent=True,
            version=radar_layer.get("version", "1.3.0"),
            name="NOAA/NWS MRMS radar",
            attr=radar_layer["attribution"],
            opacity=radar_opacity,
            overlay=True,
            control=True,
            show=True,
        ).add_to(hull_map)

    for _, asset_row in assets.iterrows():
        latitude = pd.to_numeric(asset_row.get("lat"), errors="coerce")
        longitude = pd.to_numeric(asset_row.get("lon"), errors="coerce")

        if pd.isna(latitude) or pd.isna(longitude):
            continue

        asset_id = str(asset_row.get("asset_id", ""))
        display_name = str(asset_row.get("display_name", asset_id))
        asset_type = str(asset_row.get("asset_type", "Pump Station"))
        status = str(asset_row.get("status", "No Data"))
        address = str(asset_row.get("address", "Address unavailable"))

        flow_text = display_value(
            asset_row.get("flow_gpm"),
            0,
            " gpm",
        )
        level_text = display_value(
            asset_row.get("level_in"),
            1,
            " in",
        )

        marker_color = colors.get(status, "#98A5B3")
        marker_radius = 10 if asset_type == "Treatment Plant" else 7

        if asset_id == selected_asset:
            marker_radius += 2
            marker_weight = 4
        else:
            marker_weight = 2

        popup_html = (
            '<div style="min-width:220px">'
            f"<b>{escape(display_name)}</b><br>"
            f"<span>{escape(address)}</span><br><br>"
            f"Status: <b>{escape(status)}</b><br>"
            f"Wet well: <b>{escape(level_text)}</b><br>"
            f"Reported flow: <b>{escape(flow_text)}</b><br>"
            f'<span style="font-size:11px;color:#667085">'
            f"Asset ID: {escape(asset_id)}</span>"
            "</div>"
        )

        folium.CircleMarker(
            location=[float(latitude), float(longitude)],
            radius=marker_radius,
            tooltip=asset_id,
            popup=folium.Popup(popup_html, max_width=320),
            color="white",
            weight=marker_weight,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.95,
            pane="markerPane",
        ).add_to(hull_map)

    Fullscreen(
        position="topright",
        title="Expand map",
        title_cancel="Exit full screen",
        force_separate_button=True,
    ).add_to(hull_map)

    folium.LayerControl(
        collapsed=True,
        position="topright",
    ).add_to(hull_map)

    map_event = st_folium(
        hull_map,
        use_container_width=True,
        height=650,
        key="system_map",
    )

    clicked_asset = selected_asset_from_map(
        map_event,
        assets,
    )

    if clicked_asset and clicked_asset != selected_asset:
        selected_asset = clicked_asset
        st.session_state.selected_asset = selected_asset
        st.rerun()

with right:
    choices = assets["asset_id"].astype(str).tolist()
    selected_asset = st.selectbox(
        "Selected asset",
        choices,
        index=choices.index(selected_asset) if selected_asset in choices else 0,
        label_visibility="collapsed",
    )
    st.session_state.selected_asset = selected_asset
    asset = assets.loc[assets["asset_id"] == selected_asset].iloc[0]

    p1_value = pd.to_numeric(asset.get("pump1_status"), errors="coerce")
    p2_value = pd.to_numeric(asset.get("pump2_status"), errors="coerce")
    pumps_available = pd.notna(p1_value) or pd.notna(p2_value)
    pumps_running = (
        int(
            (0 if pd.isna(p1_value) else p1_value)
            + (0 if pd.isna(p2_value) else p2_value)
        )
        if pumps_available
        else "Unavailable"
    )

    capacity = (
        fmt(asset["capacity_gpm"], 0, " gpm")
        if pd.notna(asset["capacity_gpm"])
        else "Not available"
    )

    wet_well_text = display_value(asset.get("level_in"), 1, " in")

    flow_snapshot = station_flow_snapshot(
        collection=collection,
        asset_id=selected_asset,
        as_of=as_of,
        window_minutes=5,
    )

    reported_flow_text = display_value(
        flow_snapshot.get("raw_flow_gpm"),
        0,
        " gpm",
    )
    recent_flow_text = display_value(
        flow_snapshot.get("smoothed_flow_gpm"),
        0,
        " gpm",
    )
    flow_quality = flow_snapshot.get(
        "flow_quality",
        "Flow telemetry unavailable",
    )

    st.markdown(
        f'<div class="panel"><div class="station-title">{asset["display_name"]}</div>'
        f'<div style="color:#73808c;font-size:.82rem;margin:3px 0 10px">{asset["address"]}</div>'
        f'<span class="pill">{asset["status"]}</span>'
        f'<hr style="border:none;border-top:1px solid #E8EDF2;margin:12px 0">'
        f'<b>Playback snapshot</b><br><br>'
        f'Wet well <b style="float:right">{wet_well_text}</b><br>'
        f'Reported flow <b style="float:right">{reported_flow_text}</b><br>'
        f'Recent operating flow <b style="float:right">{recent_flow_text}</b><br>'
        f'Pumps running <b style="float:right">{pumps_running}</b><br>'
        f'Design capacity <b style="float:right">{capacity}</b>'
        f'<hr style="border:none;border-top:1px solid #E8EDF2;margin:12px 0 8px">'
        f'<span style="color:#73808c;font-size:.78rem">{flow_quality}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not telemetry_available:
        st.caption(
            f"No collection-system record exists at the exact playback minute "
            f"({as_of:%b %d, %Y %I:%M %p})."
        )

    st.page_link(
        "pages/1_Pump_Station_Detail.py",
        label="Open dedicated asset page",
        icon="🔎",
        use_container_width=True,
    )

    st.markdown("#### What needs attention")
    valid_alert_assets = assets.loc[
        (assets["asset_type"] == "Pump Station")
        & assets["level_in"].notna()
    ].copy()

    alerts = [
        f"**{item.asset_id}** | {item.status} | {fmt(item.level_in, 1, ' in')} wet well"
        for _, item in valid_alert_assets.sort_values(
            "utilization", ascending=False
        ).head(4).iterrows()
    ]

    st.info(
        "\n\n".join(alerts)
        if alerts
        else "No exact-minute station telemetry is available for this playback time."
    )

# Every downstream calculation uses only records inside the selected event
# window. No nearest date, previous observation, or future observation is used.
cycles = station_cycle_summary(
    count_pump_cycles(collection, event_start, event_end)
)
baseline = build_dry_weather_baseline(influent, rain)
ii_ts = estimate_ii_timeseries(influent, baseline, event_start, event_end)
ii_summary = summarize_ii_event(ii_ts, rain_val, event_start)
summary = build_operations_summary(row, cycles, ii_summary, snap)

st.markdown("### Operations summary")
st.markdown(
    f'<div class="panel"><b style="color:{NAVY}">{summary["condition"]}</b><br><br>'
    + "<br>".join(f"• {finding}" for finding in summary["findings"])
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown("### Coordinated storm response")

collection_event = collection.loc[
    (collection["timestamp"] >= event_start)
    & (collection["timestamp"] <= event_end)
].copy()

if collection_event.empty:
    collection_chart = pd.DataFrame(
        columns=["timestamp", "collection_flow_gpm", "max_wetwell_in"]
    )
else:
    collection_chart = (
        collection_event.groupby("timestamp", as_index=False)
        .agg(
            collection_flow_gpm=("flow_gpm", "sum"),
            max_wetwell_in=("level_in", "max"),
        )
    )

influent_chart = influent.loc[
    (influent["timestamp"] >= event_start)
    & (influent["timestamp"] <= event_end),
    ["timestamp", "influent_total_mgd"],
].copy()

rain_chart = history_2026.loc[
    (history_2026["date"] >= event_start.normalize())
    & (history_2026["date"] <= event_end.normalize()),
    ["date", "rain_in"],
].copy()

tides_chart = historical_tides(
    event_start.date(),
    event_end.date() + timedelta(days=1),
)

fig = go.Figure()
fig.add_vline(
    x=as_of,
    line_width=2,
    line_dash="dash",
    line_color=RED,
    annotation_text="Playback",
)

if not rain_chart.empty:
    fig.add_trace(
        go.Bar(
            x=rain_chart["date"] + pd.Timedelta(hours=12),
            y=rain_chart["rain_in"] * 1200,
            name="Daily rainfall (scaled)",
            marker_color=MUTED_BLUE,
            opacity=0.35,
        )
    )

if not collection_chart.empty:
    fig.add_trace(
        go.Scatter(
            x=collection_chart["timestamp"],
            y=collection_chart["collection_flow_gpm"],
            name="Collection flow",
            line=dict(color=BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=collection_chart["timestamp"],
            y=collection_chart["max_wetwell_in"] * 35,
            name="Max wet well (scaled)",
            line=dict(color=AMBER, width=1.5, dash="dot"),
        )
    )

if not influent_chart.empty:
    fig.add_trace(
        go.Scatter(
            x=influent_chart["timestamp"],
            y=influent_chart["influent_total_mgd"] * 694.444,
            name="Plant influent (gpm equivalent)",
            line=dict(color=NAVY, width=2.5),
        )
    )

if not tides_chart.empty:
    fig.add_trace(
        go.Scatter(
            x=tides_chart["t"],
            y=tides_chart["v"] * 250,
            name="Tide (scaled)",
            line=dict(color=GREEN, width=1, dash="dash"),
        )
    )

fig.update_layout(
    height=410,
    margin=dict(l=15, r=15, t=10, b=20),
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.13),
    hovermode="x unified",
    xaxis=dict(showgrid=False, range=[event_start, event_end]),
    yaxis=dict(title="Operational response index", gridcolor="#EDF1F5"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("External API status and live context"):
    weather = nws_bundle()
    tides = tide_predictions()
    marine = marine_forecast()

    weather_col, tide_col, marine_col = st.columns(3)

    weather_col.write("**National Weather Service**")
    if weather.get("ok"):
        weather_col.success(
            f"Connected | {len(weather.get('hourly', []))} hourly periods"
        )
    else:
        weather_col.warning(
            "Unavailable; historical dashboard remains operational"
        )

    tide_col.write("**NOAA Tides & Currents**")
    if not tides.empty:
        tide_col.success(f"Connected | {len(tides)} tide events")
    else:
        tide_col.warning("Unavailable; historical tide layer is omitted")

    marine_col.write("**Marine forecast**")
    if not marine.empty:
        marine_col.success(f"Connected | {len(marine)} hourly periods")
    else:
        marine_col.warning("Unavailable; marine forecast layer is omitted")
        marine_error = marine.attrs.get("error")
        if marine_error:
            marine_col.caption(marine_error)