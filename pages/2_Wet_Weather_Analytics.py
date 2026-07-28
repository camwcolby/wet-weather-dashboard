from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Wet Weather Analytics | Hull", page_icon="🌧️", layout="wide")

from components.style import inject_css
from components.header import render_header
from config.theme import NAVY, BLUE, AMBER, MUTED_BLUE, GREEN
from models.pump_cycles import count_pump_cycles, station_cycle_summary
from models.ii_estimation import build_dry_weather_baseline, estimate_ii_timeseries, summarize_ii_event
from models.operations_summary import build_operations_summary
from models.storm_selection import rank_events
from services.data_loader import load_collection, load_influent, load_process_summary, load_station_runtimes, latest_snapshot
from services.weather import historical_precip, historical_hourly_precip, forecast_precipitation, recent_actual_precipitation
from services.tides import historical_tides
from utils.formatting import fmt

inject_css()
render_header("Wet Weather Analytics | Pump Cycling, Rainfall and Estimated I/I")

with st.spinner("Calculating 2026 wet-weather analytics..."):
    collection, influent = load_collection(), load_influent()
    process, runtimes = load_process_summary(), load_station_runtimes()
    min_date = min(collection.timestamp.min().date(), influent.timestamp.min().date())
    max_date = max(collection.timestamp.max().date(), influent.timestamp.max().date())
    rain = historical_precip(min_date, max_date)
    ranked, significant = rank_events(process, runtimes, rain)
    baseline = build_dry_weather_baseline(influent, rain)

st.caption("Estimated I/I is a screening-level estimate of wet-weather-derived flow above the 2026 dry-weather baseline. It is not a regulatory flow isolation study or a substitute for basin metering, smoke testing, or CCTV investigations.")

options = significant.copy() if not significant.empty else ranked.head(20).copy()
options = options.sort_values("event_date", ascending=False)
labels = {r.event_date: f"{r.event_date:%b %d, %Y} trigger to {pd.Timestamp(r.response_date):%b %d} response | score {r.storm_score:.0%}" for _, r in options.iterrows()}
selected_date = st.selectbox("Wet-weather event", options=options.event_date.tolist(), format_func=lambda d: labels.get(d, str(d)))
row = ranked.loc[ranked.event_date.eq(pd.Timestamp(selected_date))].iloc[0]
event_start, event_end = pd.Timestamp(selected_date), pd.Timestamp(selected_date) + pd.Timedelta(hours=72)
rain_total = pd.to_numeric(row.get("rain_in"), errors="coerce")

cycles = count_pump_cycles(collection, event_start, event_end)
station_cycles = station_cycle_summary(cycles)
ii_ts = estimate_ii_timeseries(influent, baseline, event_start, event_end)
ii_summary = summarize_ii_event(ii_ts, rain_total, event_start)
snapshot = latest_snapshot(collection, event_end)
summary = build_operations_summary(row, station_cycles, ii_summary, snapshot)

kpis = st.columns(6)
items = [
    ("Actual rainfall", fmt(rain_total, 2, " in") if pd.notna(rain_total) else "Unavailable", "Trigger-day total"),
    ("Estimated excess volume", fmt(ii_summary["excess_volume_mg"], 2, " MG"), "Above dry-weather baseline"),
    ("Peak excess flow", fmt(ii_summary["peak_excess_mgd"], 2, " MGD"), "Estimated wet-weather component"),
    ("Plant response lag", fmt(ii_summary["lag_hr"], 0, " hr"), "Event start to peak excess"),
    ("System pump starts", fmt(station_cycles.starts.sum() if not station_cycles.empty else np.nan, 0, ""), "Telemetry stations"),
    ("I/I response factor", fmt(ii_summary["mg_per_in"], 2, " MG/in"), "Excess volume per inch"),
]
for c, (lab, val, sub) in zip(kpis, items):
    c.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown("### Event interpretation")
st.markdown(f'<div class="panel"><b style="color:{NAVY}">{summary["condition"]}</b><br><br>' + "<br>".join(f"• {x}" for x in summary["findings"]) + "</div>", unsafe_allow_html=True)

st.markdown("### Rainfall and delayed plant response")
rain_hourly = historical_hourly_precip(event_start.date(), event_end.date())
tides = historical_tides(event_start.date(), event_end.date() + pd.Timedelta(days=1))
fig = go.Figure()
if not rain_hourly.empty:
    fig.add_trace(go.Bar(x=rain_hourly.timestamp, y=rain_hourly.actual_precip_in, name="Actual rainfall", marker_color=MUTED_BLUE, opacity=.55, yaxis="y2"))
if not ii_ts.empty:
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.influent_total_mgd, name="Observed influent", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.baseline_mgd, name="Expected dry-weather flow", line=dict(color=GREEN, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.excess_mgd, name="Estimated wet-weather-derived flow", fill="tozeroy", line=dict(color=BLUE, width=1.5)))
if not tides.empty:
    fig.add_trace(go.Scatter(x=tides.t, y=tides.v, name="Tide", line=dict(color=AMBER, dash="dot"), yaxis="y3"))
fig.update_layout(
    height=440, margin=dict(l=10, r=60, t=15, b=15), hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white",
    yaxis=dict(title="Flow (MGD)", gridcolor="#EDF1F5"),
    yaxis2=dict(title="Rain (in/hr)", overlaying="y", side="right", rangemode="tozero", showgrid=False),
    yaxis3=dict(title="Tide (ft)", overlaying="y", side="right", position=.94, showgrid=False),
    legend=dict(orientation="h", y=1.15), xaxis=dict(showgrid=False),
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.markdown("### Pump on/off cycles by station")
    if station_cycles.empty:
        st.info("No pump status transitions were available for this event window.")
    else:
        chart = px.bar(station_cycles, x="asset_id", y=["complete_cycles", "short_cycles"], barmode="group", labels={"value": "Cycle count", "asset_id": "Station", "variable": "Metric"})
        chart.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=15), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.12), yaxis=dict(gridcolor="#EDF1F5"))
        st.plotly_chart(chart, use_container_width=True)
        display = station_cycles.rename(columns={"asset_id": "Station", "starts": "Starts", "stops": "Stops", "complete_cycles": "Complete cycles", "short_cycles": "Short cycles", "runtime_hr_from_cycles": "Runtime from cycles (hr)", "median_cycle_min": "Median cycle (min)"})
        st.dataframe(display, hide_index=True, use_container_width=True)
with right:
    st.markdown("### Event I/I screening summary")
    summary_table = pd.DataFrame({
        "Metric": ["Observed peak influent", "Estimated dry-weather flow at peak", "Peak excess flow", "Estimated excess volume", "Response lag", "Response factor"],
        "Value": [fmt(ii_summary["peak_observed_mgd"], 2, " MGD"), fmt(ii_summary["peak_observed_mgd"] - ii_summary["peak_excess_mgd"], 2, " MGD"), fmt(ii_summary["peak_excess_mgd"], 2, " MGD"), fmt(ii_summary["excess_volume_mg"], 2, " MG"), fmt(ii_summary["lag_hr"], 0, " hr"), fmt(ii_summary["mg_per_in"], 2, " MG/in")],
    })
    st.dataframe(summary_table, hide_index=True, use_container_width=True)

st.markdown("### Current forecast versus recent actual rainfall")
forecast, actual = forecast_precipitation(7), recent_actual_precipitation(10)
forecast_total = forecast.forecast_precip_in.sum() if not forecast.empty else np.nan
actual_total = actual.actual_precip_in.sum() if not actual.empty else np.nan
fa, fb = st.columns(2)
fa.metric("Next 7-day forecast rainfall", fmt(forecast_total, 2, " in"))
fb.metric("Recent 10-day actual rainfall", fmt(actual_total, 2, " in"))
if not forecast.empty or not actual.empty:
    rf = go.Figure()
    if not actual.empty: rf.add_trace(go.Bar(x=actual.timestamp, y=actual.actual_precip_in, name="Actual", marker_color=NAVY))
    if not forecast.empty: rf.add_trace(go.Bar(x=forecast.timestamp, y=forecast.forecast_precip_in, name="Forecast", marker_color=BLUE))
    rf.update_layout(height=300, barmode="overlay", hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", yaxis_title="Rainfall (in/hr)", legend=dict(orientation="h", y=1.14), yaxis=dict(gridcolor="#EDF1F5"))
    st.plotly_chart(rf, use_container_width=True)
else:
    st.warning("Rainfall services are currently unavailable. Once either source connects, the app caches the series for repeatable demonstrations.")
