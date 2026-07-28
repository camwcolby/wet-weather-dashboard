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
from models.pump_cycles import count_pump_cycles, station_cycle_summary, daily_pump_starts
from models.ii_estimation import build_dry_weather_baseline, estimate_ii_timeseries, summarize_ii_event
from models.operations_summary import build_operations_summary
from models.storm_selection import rank_events
from services.data_loader import (
    load_collection, load_influent, load_historical_influent_rain,
    load_station_runtimes, latest_snapshot,
)
from services.weather import forecast_precipitation, recent_actual_precipitation
from services.tides import historical_tides
from utils.formatting import fmt

inject_css()
render_header("Wet Weather Analytics | Pump Cycling, Rainfall and Estimated I/I")

with st.spinner("Calculating wet-weather analytics..."):
    collection = load_collection()
    influent = load_influent()
    runtimes = load_station_runtimes()
    history = load_historical_influent_rain()
    min_date = max(collection.timestamp.min().normalize(), history.date.min())
    max_date = min(collection.timestamp.max().normalize(), history.date.max())
    history_2026 = history[(history.date >= min_date) & (history.date <= max_date)].copy()
    rain = history_2026[["date", "rain_in"]].copy()
    ranked, significant = rank_events(history_2026, runtimes)
    telemetry_dates = set(collection["timestamp"].dt.normalize().unique())
    covered = significant[significant["event_date"].isin(telemetry_dates)].copy()
    if not covered.empty:
        significant = covered
    baseline = build_dry_weather_baseline(influent, rain)
    starts_daily = daily_pump_starts(collection)

st.caption(
    "Estimated I/I is a screening-level estimate of wet-weather-derived flow above the 2026 dry-weather baseline. "
    "It is not a regulatory flow-isolation study or a substitute for basin metering, smoke testing, or CCTV investigations."
)

options = significant.copy() if not significant.empty else ranked.head(20).copy()
options = options.sort_values("event_date", ascending=False)
labels = {
    r.event_date: f"{r.event_date:%b %d, %Y} rainfall → {pd.Timestamp(r.response_date):%b %d} response | "
                  f"{r.rain_in:.2f} in | score {r.storm_score:.0%}"
    for _, r in options.iterrows()
}
selected_date = st.selectbox(
    "Wet-weather event",
    options=options.event_date.tolist(),
    format_func=lambda d: labels.get(d, str(d)),
)
row = ranked.loc[ranked.event_date.eq(pd.Timestamp(selected_date))].iloc[0]
event_start = pd.Timestamp(selected_date)
event_end = event_start + pd.Timedelta(hours=72)
rain_total = float(row["rain_in"]) if pd.notna(row["rain_in"]) else np.nan

cycles = count_pump_cycles(collection, event_start, event_end)
station_cycles = station_cycle_summary(cycles)
ii_ts = estimate_ii_timeseries(influent, baseline, event_start, event_end)
ii_summary = summarize_ii_event(ii_ts, rain_total, event_start)
snapshot = latest_snapshot(collection, event_end)
summary = build_operations_summary(row, station_cycles, ii_summary, snapshot)

kpis = st.columns(6)
items = [
    ("Actual rainfall", fmt(rain_total, 2, " in"), "Local plant record"),
    ("Estimated excess volume", fmt(ii_summary["excess_volume_mg"], 2, " MG"), "Above dry-weather baseline"),
    ("Peak excess flow", fmt(ii_summary["peak_excess_mgd"], 2, " MGD"), "Estimated wet-weather component"),
    ("Plant response lag", fmt(ii_summary["lag_hr"], 0, " hr"), "Rainfall date to peak excess"),
    ("System pump starts", fmt(station_cycles.starts.sum() if not station_cycles.empty else np.nan, 0, ""), "72-hour event window"),
    ("I/I response factor", fmt(ii_summary["mg_per_in"], 2, " MG/in"), "Excess volume per inch"),
]
for c, (lab, val, sub) in zip(kpis, items):
    c.markdown(
        f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True,
    )

st.markdown("### Event interpretation")
st.markdown(
    f'<div class="panel"><b style="color:{NAVY}">{summary["condition"]}</b><br><br>'
    + "<br>".join(f"• {x}" for x in summary["findings"]) + "</div>",
    unsafe_allow_html=True,
)

st.markdown("### Rainfall and delayed plant response")
event_rain = history_2026[(history_2026.date >= event_start) & (history_2026.date <= event_end.normalize())]
tides = historical_tides(event_start.date(), event_end.date() + pd.Timedelta(days=1))
fig = go.Figure()
if not event_rain.empty:
    fig.add_trace(go.Bar(
        x=event_rain.date + pd.Timedelta(hours=12), y=event_rain.rain_in,
        name="Daily rainfall", marker_color=MUTED_BLUE, opacity=.60, yaxis="y2",
    ))
if not ii_ts.empty:
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.influent_total_mgd, name="Observed influent", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.baseline_mgd, name="Expected dry-weather flow", line=dict(color=GREEN, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ii_ts.timestamp, y=ii_ts.excess_mgd, name="Estimated wet-weather-derived flow", fill="tozeroy", line=dict(color=BLUE, width=1.5)))
if not tides.empty:
    fig.add_trace(go.Scatter(x=tides.t, y=tides.v, name="Tide", line=dict(color=AMBER, dash="dot"), yaxis="y3"))
fig.update_layout(
    height=440, margin=dict(l=10, r=70, t=15, b=15), hovermode="x unified", paper_bgcolor="white", plot_bgcolor="white",
    yaxis=dict(title="Flow (MGD)", gridcolor="#EDF1F5"),
    yaxis2=dict(title="Daily rain (in)", overlaying="y", side="right", rangemode="tozero", showgrid=False),
    yaxis3=dict(title="Tide (ft)", overlaying="y", side="right", position=.94, showgrid=False),
    legend=dict(orientation="h", y=1.15), xaxis=dict(showgrid=False),
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.markdown("### Pump turn-ons during selected event")
    if station_cycles.empty:
        st.info("No pump status transitions were available for this event window.")
    else:
        chart = px.bar(
            station_cycles, x="asset_id", y="starts",
            labels={"starts": "Pump turn-ons", "asset_id": "Station"},
        )
        chart.update_layout(height=315, margin=dict(l=10, r=10, t=10, b=15), paper_bgcolor="white", plot_bgcolor="white", yaxis=dict(gridcolor="#EDF1F5"))
        st.plotly_chart(chart, use_container_width=True)
        display = station_cycles.rename(columns={
            "asset_id": "Station", "starts": "Turn-ons", "stops": "Turn-offs",
            "complete_cycles": "Complete cycles", "short_cycles": "Short cycles",
            "runtime_hr_from_cycles": "Runtime from cycles (hr)", "median_cycle_min": "Median run (min)",
        })
        st.dataframe(display, hide_index=True, use_container_width=True)
with right:
    st.markdown("### Event I/I screening summary")
    summary_table = pd.DataFrame({
        "Metric": ["Observed peak influent", "Estimated dry-weather flow at peak", "Peak excess flow", "Estimated excess volume", "Response lag", "Response factor"],
        "Value": [
            fmt(ii_summary["peak_observed_mgd"], 2, " MGD"),
            fmt(ii_summary["peak_observed_mgd"] - ii_summary["peak_excess_mgd"], 2, " MGD"),
            fmt(ii_summary["peak_excess_mgd"], 2, " MGD"),
            fmt(ii_summary["excess_volume_mg"], 2, " MG"),
            fmt(ii_summary["lag_hr"], 0, " hr"),
            fmt(ii_summary["mg_per_in"], 2, " MG/in"),
        ],
    })
    st.dataframe(summary_table, hide_index=True, use_container_width=True)

st.markdown("### Daily pump turn-ons by station")
if starts_daily.empty:
    st.info("Daily pump turn-ons could not be calculated from the status telemetry.")
else:
    station_options = sorted(starts_daily.asset_id.unique())
    selected_station = st.selectbox("Station", station_options, index=0, key="daily_starts_station")
    station_daily = starts_daily[starts_daily.asset_id.eq(selected_station)].copy()
    pivot = station_daily.pivot_table(index="date", columns="pump", values="starts", aggfunc="sum", fill_value=0).reset_index()
    pump_cols = [c for c in pivot.columns if c != "date"]
    pivot["Total"] = pivot[pump_cols].sum(axis=1)
    starts_fig = px.bar(pivot, x="date", y=pump_cols, barmode="stack", labels={"value": "Turn-ons per day", "date": "Date", "variable": "Pump"})
    starts_fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", yaxis=dict(gridcolor="#EDF1F5"), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(starts_fig, use_container_width=True)
    st.dataframe(pivot.sort_values("date", ascending=False), hide_index=True, use_container_width=True)

st.markdown("### Current forecast versus recent observed rainfall")
forecast, actual = forecast_precipitation(7), recent_actual_precipitation(10)
forecast_total = forecast.forecast_precip_in.sum() if not forecast.empty else np.nan
actual_total = actual.actual_precip_in.sum() if not actual.empty else np.nan
fa, fb = st.columns(2)
fa.metric("Next 7-day forecast rainfall", fmt(forecast_total, 2, " in"))
fb.metric("Recent 10-day observed rainfall", fmt(actual_total, 2, " in"))
if not forecast.empty or not actual.empty:
    rf = go.Figure()
    if not actual.empty:
        rf.add_trace(go.Bar(x=actual.timestamp, y=actual.actual_precip_in, name="Observed", marker_color=NAVY))
    if not forecast.empty:
        rf.add_trace(go.Bar(x=forecast.timestamp, y=forecast.forecast_precip_in, name="Forecast", marker_color=BLUE))
    rf.update_layout(height=300, barmode="overlay", hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", yaxis_title="Rainfall (in/hr)", legend=dict(orientation="h", y=1.14), yaxis=dict(gridcolor="#EDF1F5"))
    st.plotly_chart(rf, use_container_width=True)
else:
    st.warning("Live rainfall services are currently unavailable. Historical event analytics remain available from the local plant record.")
