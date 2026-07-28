from __future__ import annotations

from datetime import timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hull Wet Weather Operations", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

from components.style import inject_css
from components.header import render_header
from config.theme import NAVY, BLUE, GREEN, LIME, AMBER, RED, MUTED_BLUE
from models.ii_estimation import build_dry_weather_baseline, estimate_ii_timeseries, summarize_ii_event
from models.operations_summary import build_operations_summary
from models.pump_cycles import count_pump_cycles, station_cycle_summary
from models.storm_selection import rank_events
from services.data_loader import (
    load_collection, load_influent, load_process_summary, load_station_runtimes,
    latest_snapshot, load_asset_locations,
)
from services.weather import historical_precip, historical_hourly_precip, nws_bundle
from services.tides import tide_predictions, historical_tides
from services.marine import marine_forecast
from utils.formatting import fmt, status_from_utilization

inject_css()
with st.spinner("Loading 2026 operating data..."):
    collection = load_collection()
    influent = load_influent()
    process = load_process_summary()
    runtimes = load_station_runtimes()
    min_date = min(collection.timestamp.min().date(), process.date.min().date())
    max_date = max(collection.timestamp.max().date(), process.date.max().date(), runtimes.date.max().date())
    rain = historical_precip(min_date, max_date)
    ranked, significant = rank_events(process, runtimes, rain)

latest_storm = significant.iloc[0].event_date if not significant.empty else ranked.iloc[0].event_date
mode = st.sidebar.radio("Operating view", ["Latest significant storm", "Latest available data", "Custom date"], index=0)
if mode == "Latest significant storm":
    selected_day = pd.Timestamp(latest_storm)
elif mode == "Latest available data":
    selected_day = pd.Timestamp(max_date)
else:
    selected_day = pd.Timestamp(st.sidebar.date_input("Date", value=latest_storm, min_value=min_date, max_value=max_date))

row_match = ranked[ranked.event_date.dt.normalize() == selected_day.normalize()]
row = row_match.iloc[0] if not row_match.empty else pd.Series(dtype=float)
response_day = pd.Timestamp(row.get("response_date", selected_day))
event_start = selected_day.normalize()
event_end = event_start + pd.Timedelta(hours=72)

# Historical playback is the control spine of the operator page.
playback_hour = st.sidebar.slider("Storm playback hour", min_value=0, max_value=72, value=min(48, 72), step=1)
as_of = event_start + pd.Timedelta(hours=playback_hour)
render_header(f"Historical Playback | Rain {selected_day:%b %d} to Response {response_day:%b %d, %Y}")

control_cols = st.columns([1.4, 3.6, 1.0])
with control_cols[0]:
    st.page_link("pages/2_Wet_Weather_Analytics.py", label="Open Wet Weather Analytics", icon="🌧️")
with control_cols[1]:
    st.caption(f"Playback time: **{as_of:%a %b %d, %Y at %I:%M %p}** | Window: event start through +72 hr")
with control_cols[2]:
    st.caption("Use the sidebar slider to replay the event")

rain_val = pd.to_numeric(row.get("rain_in"), errors="coerce")
plant_flow = pd.to_numeric(row.get("plant_peak_mgd"), errors="coerce")
storm_score = pd.to_numeric(row.get("storm_score"), errors="coerce")
response_lag = pd.to_numeric(row.get("response_lag_hr"), errors="coerce")
snap = latest_snapshot(collection, as_of)
max_level = snap.level_in.max() if not snap.empty else np.nan
running = int(((snap.get("pump1_status", 0).fillna(0) + snap.get("pump2_status", 0).fillna(0)) > 0).sum()) if not snap.empty else 0
severity = "ALARM" if storm_score >= .8 else "ELEVATED" if storm_score >= .55 else "WATCH" if storm_score >= .3 else "NORMAL"
sev_color = {"NORMAL": GREEN, "WATCH": LIME, "ELEVATED": AMBER, "ALARM": RED}[severity]
rain_text = fmt(rain_val, 2, " in") if pd.notna(rain_val) else "Rainfall unavailable"
st.markdown(
    f'<div class="status-strip" style="border-left-color:{sev_color}"><b style="color:{NAVY}">{severity} WET WEATHER STATUS</b>'
    f' &nbsp; Storm score {fmt(storm_score*100,0,"%") if pd.notna(storm_score) else "—"} | {rain_text} | '
    f'plant response peak +{fmt(response_lag,0," hr")} | {running}/7 telemetry stations operating at playback time</div>',
    unsafe_allow_html=True,
)

kpis = st.columns(6)
items = [
    ("Rainfall trigger", rain_text, "Event-day total"),
    ("Peak plant influent", fmt(plant_flow, 2, " MGD"), "Trigger + following 48 hr"),
    ("Highest wet well", fmt(max_level, 1, " in"), "At playback time"),
    ("Stations running", f"{running} / 7", "At playback time"),
    ("System runtime", fmt(row.get("total_runtime_72h", np.nan), 1, " hr"), "Trigger + following 48 hr"),
    ("Storm score", fmt(storm_score * 100, 0, "%"), "Rain + hydraulic response"),
]
for c, (lab, val, sub) in zip(kpis, items):
    c.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

assets = load_asset_locations()
if not snap.empty:
    cols = [c for c in ["asset_id", "flow_gpm", "level_in", "pump1_status", "pump2_status", "interceptor_level"] if c in snap]
    assets = assets.merge(snap[cols], on="asset_id", how="left")
else:
    for col in ["flow_gpm", "level_in", "pump1_status", "pump2_status", "interceptor_level"]:
        assets[col] = np.nan
assets["utilization"] = (assets.level_in / 84).clip(0, 1.2)
assets["status"] = assets.utilization.apply(status_from_utilization)
assets.loc[assets.asset_type == "Treatment Plant", "status"] = "Plant"
colors = {"Normal": GREEN, "Watch": LIME, "Warning": AMBER, "Alarm": RED, "No Data": "#98A5B3", "Plant": NAVY}

left, right = st.columns([2.45, 1], gap="medium")
with left:
    fig = px.scatter_map(
        assets, lat="lat", lon="lon", color="status", color_discrete_map=colors,
        size=assets.asset_type.map({"Treatment Plant": 28, "Pump Station": 18}).fillna(18),
        hover_name="display_name",
        hover_data={"address": True, "flow_gpm": ":.0f", "level_in": ":.1f", "lat": False, "lon": False, "status": False},
        zoom=12.0, center={"lat": 42.286, "lon": -70.882}, height=650,
    )
    fig.update_layout(
        map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.01, x=.01, bgcolor="rgba(255,255,255,.86)"), clickmode="event+select",
    )
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="system_map")
    selected_asset = st.session_state.get("selected_asset", "PS 3")
    try:
        points = event.selection.points if event and event.selection else []
        if points:
            name = points[0].get("hovertext") or points[0].get("customdata", [None])[0]
            match = assets.loc[assets.display_name == name]
            if not match.empty:
                selected_asset = match.iloc[0].asset_id
                st.session_state.selected_asset = selected_asset
    except Exception:
        pass

with right:
    choices = assets.asset_id.tolist()
    selected_asset = st.selectbox("Selected asset", choices, index=choices.index(selected_asset) if selected_asset in choices else 0, label_visibility="collapsed")
    st.session_state.selected_asset = selected_asset
    a = assets.loc[assets.asset_id == selected_asset].iloc[0]
    p1 = 0 if pd.isna(a.get("pump1_status")) else int(a.get("pump1_status", 0))
    p2 = 0 if pd.isna(a.get("pump2_status")) else int(a.get("pump2_status", 0))
    capacity = fmt(a.capacity_gpm, 0, " gpm") if pd.notna(a.capacity_gpm) else "Not available"
    st.markdown(
        f'<div class="panel"><div class="station-title">{a.display_name}</div><div style="color:#73808c;font-size:.82rem;margin:3px 0 10px">{a.address}</div>'
        f'<span class="pill">{a.status}</span><hr style="border:none;border-top:1px solid #E8EDF2;margin:12px 0"><b>Playback snapshot</b><br><br>'
        f'Wet well <b style="float:right">{fmt(a.level_in,1," in")}</b><br>Flow <b style="float:right">{fmt(a.flow_gpm,0," gpm")}</b><br>'
        f'Pumps running <b style="float:right">{p1+p2}</b><br>Design capacity <b style="float:right">{capacity}</b></div>', unsafe_allow_html=True,
    )
    st.page_link("pages/1_Pump_Station_Detail.py", label="Open dedicated asset page", icon="🔎", use_container_width=True)
    st.markdown("#### What needs attention")
    alerts = [f"**{x.asset_id}** | {x.status} | {fmt(x.level_in,1,' in')} wet well" for _, x in assets[assets.asset_type == "Pump Station"].sort_values("utilization", ascending=False).head(4).iterrows()]
    st.info("\n\n".join(alerts) if alerts else "No station data available")

# Deterministic operations narrative.
cycles = station_cycle_summary(count_pump_cycles(collection, event_start, event_end))
baseline = build_dry_weather_baseline(influent, rain)
ii_ts = estimate_ii_timeseries(influent, baseline, event_start, event_end)
ii_summary = summarize_ii_event(ii_ts, rain_val, event_start)
summary = build_operations_summary(row, cycles, ii_summary, snap)
st.markdown("### Operations summary")
st.markdown(f'<div class="panel"><b style="color:{NAVY}">{summary["condition"]}</b><br><br>' + "<br>".join(f"• {x}" for x in summary["findings"]) + "</div>", unsafe_allow_html=True)

st.markdown("### Coordinated storm response")
start = event_start
c = collection[(collection.timestamp >= start) & (collection.timestamp <= event_end)].groupby("timestamp", as_index=False).agg(collection_flow_gpm=("flow_gpm", "sum"), max_wetwell_in=("level_in", "max"))
i = influent[(influent.timestamp >= start) & (influent.timestamp <= event_end)][["timestamp", "influent_total_mgd"]]
rh = historical_hourly_precip(start.date(), event_end.date())
t = historical_tides(start.date(), event_end.date() + timedelta(days=1))
fig = go.Figure()
fig.add_vline(x=as_of, line_width=2, line_dash="dash", line_color=RED, annotation_text="Playback")
if not rh.empty:
    fig.add_trace(go.Bar(x=rh.timestamp, y=rh.actual_precip_in * 1200, name="Rainfall (scaled)", marker_color=MUTED_BLUE, opacity=.35))
fig.add_trace(go.Scatter(x=c.timestamp, y=c.collection_flow_gpm, name="Collection flow", line=dict(color=BLUE, width=2)))
fig.add_trace(go.Scatter(x=i.timestamp, y=i.influent_total_mgd * 694.444, name="Plant influent (gpm equivalent)", line=dict(color=NAVY, width=2.5)))
fig.add_trace(go.Scatter(x=c.timestamp, y=c.max_wetwell_in * 35, name="Max wet well (scaled)", line=dict(color=AMBER, width=1.5, dash="dot")))
if not t.empty:
    fig.add_trace(go.Scatter(x=t.t, y=t.v * 250, name="Tide (scaled)", line=dict(color=GREEN, width=1, dash="dash")))
fig.update_layout(height=410, margin=dict(l=15, r=15, t=10, b=20), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.13), hovermode="x unified", xaxis=dict(showgrid=False), yaxis=dict(title="Operational response index", gridcolor="#EDF1F5"))
st.plotly_chart(fig, use_container_width=True)

with st.expander("External API status and live context"):
    weather, tides, marine = nws_bundle(), tide_predictions(), marine_forecast()
    a1, b1, c1 = st.columns(3)
    a1.write("**National Weather Service**"); a1.success(f"Connected | {len(weather['hourly'])} hourly periods") if weather.get("ok") else a1.warning("Unavailable; historical dashboard remains operational")
    b1.write("**NOAA Tides & Currents**"); b1.success(f"Connected | {len(tides)} tide events") if not tides.empty else b1.warning("Unavailable; cached/historical view used")
    c1.write("**Marine forecast**"); c1.success(f"Connected | {len(marine)} hourly periods") if not marine.empty else c1.warning("Unavailable; dashboard degrades gracefully")
