from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Wet Weather Analytics | Hull", page_icon="🌧️", layout="wide")

from components.style import inject_css
from components.header import render_header
from config.theme import NAVY, BLUE, GREEN, AMBER, RED, MUTED_BLUE
from models.pump_cycles import count_pump_cycles, station_cycle_summary
from models.ii_estimation import build_dry_weather_baseline, estimate_ii_timeseries, summarize_ii_event
from models.storm_selection import rank_events
from services.data_loader import load_collection, load_influent, load_process_summary, load_station_runtimes
from services.weather import historical_precip, forecast_precipitation, recent_actual_precipitation
from utils.formatting import fmt

inject_css()
render_header("Wet Weather Analytics · Pump Cycling, Rainfall & Estimated I/I")

with st.spinner("Calculating 2026 wet-weather analytics..."):
    collection = load_collection()
    influent = load_influent()
    process = load_process_summary()
    runtimes = load_station_runtimes()
    min_date = min(collection.timestamp.min().date(), influent.timestamp.min().date())
    max_date = max(collection.timestamp.max().date(), influent.timestamp.max().date())
    rain = historical_precip(min_date, max_date)
    ranked, significant = rank_events(process, runtimes, rain)
    baseline = build_dry_weather_baseline(influent, rain)

st.caption(
    "Estimated I/I is a screening-level estimate of wet-weather-derived flow above the 2026 dry-weather baseline. "
    "It is not a regulatory flow isolation study or a substitute for basin metering, smoke testing, or CCTV investigations."
)

# Event selection
options = significant.copy() if not significant.empty else ranked.head(20).copy()
options = options.sort_values("event_date", ascending=False)
labels = {
    r.event_date: f"{r.event_date:%b %d, %Y} rain → {pd.Timestamp(r.response_date):%b %d} response · score {r.storm_score:.0%}"
    for _, r in options.iterrows()
}
selected_date = st.selectbox(
    "Wet-weather event",
    options=options["event_date"].tolist(),
    format_func=lambda d: labels.get(d, str(d)),
)
row = ranked.loc[ranked["event_date"].eq(pd.Timestamp(selected_date))].iloc[0]
event_start = pd.Timestamp(selected_date)
event_end = event_start + pd.Timedelta(hours=72)
rain_total = float(row.get("rain_in", np.nan))

# Event analytics
cycles = count_pump_cycles(collection, event_start, event_end)
station_cycles = station_cycle_summary(cycles)
ii_ts = estimate_ii_timeseries(influent, baseline, event_start, event_end)
ii_summary = summarize_ii_event(ii_ts, rain_total, event_start)

kpis = st.columns(6)
items = [
    ("Actual rainfall", fmt(rain_total, 2, " in"), "Trigger-day total"),
    ("Estimated excess volume", fmt(ii_summary["excess_volume_mg"], 2, " MG"), "Above dry-weather baseline"),
    ("Peak excess flow", fmt(ii_summary["peak_excess_mgd"], 2, " MGD"), "Estimated wet-weather component"),
    ("Plant response lag", fmt(ii_summary["lag_hr"], 0, " hr"), "Rain date to peak excess"),
    ("System pump starts", fmt(station_cycles["starts"].sum() if not station_cycles.empty else np.nan, 0, ""), "Seven telemetry stations"),
    ("I/I response factor", fmt(ii_summary["mg_per_in"], 2, " MG/in"), "Excess volume per inch"),
]
for c, (lab, val, sub) in zip(kpis, items):
    c.markdown(
        f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True,
    )

st.markdown("### Rainfall and delayed plant response")
fig = go.Figure()
if not ii_ts.empty:
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.influent_total_mgd, name="Observed influent", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.baseline_mgd, name="Expected dry-weather flow", line=dict(color=MUTED_BLUE, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.excess_mgd, name="Estimated wet-weather-derived flow", fill="tozeroy", line=dict(color=BLUE, width=1.5)))
fig.update_layout(height=390, margin=dict(l=10, r=10, t=15, b=15), hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white", yaxis_title="Flow (MGD)", legend=dict(orientation="h", y=1.12), xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#EDF1F5"))
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
        display = station_cycles.rename(columns={
            "asset_id": "Station", "starts": "Starts", "stops": "Stops",
            "complete_cycles": "Complete cycles", "short_cycles": "Short cycles",
            "runtime_hr_from_cycles": "Runtime from cycles (hr)", "median_cycle_min": "Median cycle (min)",
        })
        st.dataframe(display, hide_index=True, use_container_width=True, column_config={
            "Runtime from cycles (hr)": st.column_config.NumberColumn(format="%.1f"),
            "Median cycle (min)": st.column_config.NumberColumn(format="%.1f"),
        })
with right:
    st.markdown("### Event I/I screening summary")
    summary = pd.DataFrame({
        "Metric": ["Observed peak influent", "Estimated dry-weather baseline at peak", "Peak estimated excess flow", "Total estimated excess volume", "Rain-to-peak lag", "Excess volume per inch"],
        "Value": [
            fmt(ii_summary["peak_observed_mgd"], 2, " MGD"),
            fmt(ii_ts.loc[ii_ts.excess_mgd.idxmax(), "baseline_mgd"] if not ii_ts.empty else np.nan, 2, " MGD"),
            fmt(ii_summary["peak_excess_mgd"], 2, " MGD"),
            fmt(ii_summary["excess_volume_mg"], 2, " MG"),
            fmt(ii_summary["lag_hr"], 0, " hr"),
            fmt(ii_summary["mg_per_in"], 2, " MG/in"),
        ],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.info(
        "Use this estimate to compare storm response, identify unusually responsive periods, and prioritize tributary areas. "
        "The next refinement should incorporate local rain gauges, antecedent groundwater conditions, and basin-level metering."
    )

st.markdown("### Current forecast rainfall versus recent actual rainfall")
forecast = forecast_precipitation(7)
actual = recent_actual_precipitation(10)
rf = go.Figure()
if not actual.empty:
    rf.add_trace(go.Bar(x=actual.timestamp, y=actual.actual_precip_in, name="Actual rainfall", marker_color=BLUE))
if not forecast.empty:
    rf.add_trace(go.Bar(x=forecast.timestamp, y=forecast.forecast_precip_in, name="Forecast rainfall", marker_color=GREEN))
    rf.add_trace(go.Scatter(x=forecast.timestamp, y=forecast.precip_probability_pct / 100.0, name="Forecast probability (scaled)", yaxis="y2", line=dict(color=AMBER, dash="dot")))
rf.update_layout(
    height=360, margin=dict(l=10, r=10, t=10, b=15), barmode="overlay", hovermode="x unified",
    paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.12),
    yaxis=dict(title="Rainfall (in/hr)", gridcolor="#EDF1F5"),
    yaxis2=dict(title="Probability", overlaying="y", side="right", tickformat=".0%", range=[0, 1]),
)
st.plotly_chart(rf, use_container_width=True)
st.caption(
    "The current prototype compares the live forward forecast with recent actual rainfall. Historical forecast accuracy requires "
    "saving each forecast snapshot as it is issued; the app's external-cache framework is ready for that next step."
)

with st.expander("Individual pump detail"):
    if cycles.empty:
        st.write("No cycle records available.")
    else:
        st.dataframe(cycles, hide_index=True, use_container_width=True)
