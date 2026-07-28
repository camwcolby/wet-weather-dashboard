from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Hull Asset Detail",
    page_icon="⚙️",
    layout="wide",
)

from components.header import render_header
from components.style import inject_css
from config.assets import ASSETS
from config.theme import AMBER, BLUE, GREEN, NAVY
from services.data_loader import (
    load_collection,
    load_station_runtimes,
)


inject_css()
render_header("Asset Drill-down")


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
collection = load_collection()
runtimes = load_station_runtimes()

if collection.empty:
    st.error("Collection-system telemetry could not be loaded.")
    st.stop()


# ------------------------------------------------------------
# Read the date and playback selections from app.py
# ------------------------------------------------------------
collection_max = pd.to_datetime(
    collection["timestamp"],
    errors="coerce",
).max()

fallback_day = collection_max.normalize()

selected_day = pd.Timestamp(
    st.session_state.get(
        "playback_selected_day",
        fallback_day,
    )
).normalize()

event_start = pd.Timestamp(
    st.session_state.get(
        "playback_event_start",
        selected_day,
    )
)

event_end = pd.Timestamp(
    st.session_state.get(
        "playback_event_end",
        event_start + pd.Timedelta(hours=72),
    )
)

as_of = pd.Timestamp(
    st.session_state.get(
        "playback_as_of",
        event_end,
    )
)

# Do not show data beyond the selected playback moment.
chart_end = min(as_of, event_end)


# ------------------------------------------------------------
# Pump-station selector
# ------------------------------------------------------------
station_ids = [
    asset["asset_id"]
    for asset in ASSETS
    if asset["asset_type"] == "Pump Station"
]

default_station = st.session_state.get(
    "selected_asset",
    "PS 3",
)

if default_station not in station_ids:
    default_station = "PS 3"

station = st.selectbox(
    "Pump station",
    station_ids,
    index=station_ids.index(default_station),
)

st.session_state["selected_asset"] = station

asset = next(
    item
    for item in ASSETS
    if item["asset_id"] == station
)


# ------------------------------------------------------------
# Filter minute telemetry to the same playback window
# selected on the main page
# ------------------------------------------------------------
station_data = collection.loc[
    collection["asset_id"].eq(station)
].copy()

station_data["timestamp"] = pd.to_datetime(
    station_data["timestamp"],
    errors="coerce",
)

station_data = station_data.loc[
    station_data["timestamp"].between(
        event_start,
        chart_end,
        inclusive="both",
    )
].sort_values("timestamp")


# ------------------------------------------------------------
# Page heading
# ------------------------------------------------------------
st.markdown(f"## {asset['display_name']}")

st.caption(
    f"{asset['address']} · "
    f"Design capacity {asset['capacity_gpm']:,} gpm · "
    f"Force main {asset['force_main']}"
)

st.caption(
    f"Playback window: "
    f"**{event_start:%b %d, %Y %I:%M %p}** through "
    f"**{chart_end:%b %d, %Y %I:%M %p}**"
)


# ------------------------------------------------------------
# Exact playback snapshot
# ------------------------------------------------------------
exact_snapshot = station_data.loc[
    station_data["timestamp"].eq(
        chart_end.floor("min")
    )
]

if exact_snapshot.empty:
    st.warning(
        "No exact telemetry record exists for this station at "
        f"{chart_end:%b %d, %Y %I:%M %p}. "
        "No nearest timestamp has been substituted."
    )

    wet_well_text = "Unavailable"
    flow_text = "Unavailable"
    pump_1_text = "Unavailable"
    pump_2_text = "Unavailable"

else:
    latest = exact_snapshot.iloc[-1]

    wet_well_value = pd.to_numeric(
        latest.get("level_in"),
        errors="coerce",
    )

    flow_value = pd.to_numeric(
        latest.get("flow_gpm"),
        errors="coerce",
    )

    pump_1_value = pd.to_numeric(
        latest.get("pump1_status"),
        errors="coerce",
    )

    pump_2_value = pd.to_numeric(
        latest.get("pump2_status"),
        errors="coerce",
    )

    wet_well_text = (
        f"{wet_well_value:.1f} in"
        if pd.notna(wet_well_value)
        else "Unavailable"
    )

    flow_text = (
        f"{flow_value:,.0f} gpm"
        if pd.notna(flow_value)
        else "Unavailable"
    )

    pump_1_text = (
        "RUNNING"
        if pd.notna(pump_1_value) and pump_1_value > 0
        else "OFF"
        if pd.notna(pump_1_value)
        else "Unavailable"
    )

    pump_2_text = (
        "RUNNING"
        if pd.notna(pump_2_value) and pump_2_value > 0
        else "OFF"
        if pd.notna(pump_2_value)
        else "Unavailable"
    )


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Wet-well level",
    wet_well_text,
)

metric_2.metric(
    "Reported flow",
    flow_text,
)

metric_3.metric(
    "Pump 1",
    pump_1_text,
)

metric_4.metric(
    "Pump 2",
    pump_2_text,
)


# ------------------------------------------------------------
# Minute telemetry chart
# ------------------------------------------------------------
st.markdown("### Wet-well level and flow")

if station_data.empty:
    st.info(
        "No station telemetry exists within the selected "
        "playback window."
    )

else:
    telemetry_figure = go.Figure()

    if "level_in" in station_data.columns:
        telemetry_figure.add_trace(
            go.Scatter(
                x=station_data["timestamp"],
                y=pd.to_numeric(
                    station_data["level_in"],
                    errors="coerce",
                ),
                name="Wet well (in)",
                line=dict(
                    color=AMBER,
                    width=2,
                ),
            )
        )

    if "flow_gpm" in station_data.columns:
        telemetry_figure.add_trace(
            go.Scatter(
                x=station_data["timestamp"],
                y=pd.to_numeric(
                    station_data["flow_gpm"],
                    errors="coerce",
                ),
                name="Flow (gpm)",
                yaxis="y2",
                line=dict(
                    color=BLUE,
                    width=2,
                ),
            )
        )

    telemetry_figure.add_vline(
        x=chart_end,
        line_width=2,
        line_dash="dash",
        line_color=NAVY,
        annotation_text="Playback",
    )

    telemetry_figure.update_layout(
        height=430,
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            range=[event_start, chart_end],
            showgrid=False,
        ),
        yaxis=dict(
            title="Wet well (in)",
            gridcolor="#EDF1F5",
        ),
        yaxis2=dict(
            title="Flow (gpm)",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="h",
            y=1.10,
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
    )

    st.plotly_chart(
        telemetry_figure,
        use_container_width=True,
    )


# ------------------------------------------------------------
# Daily runtime and pumped-volume chart
# ------------------------------------------------------------
st.markdown("### Daily runtime and pumped volume")

runtime_data = runtimes.loc[
    runtimes["asset_id"].eq(station)
].copy()

if not runtime_data.empty:
    runtime_data["date"] = pd.to_datetime(
        runtime_data["date"],
        errors="coerce",
    ).dt.normalize()

    runtime_start_date = event_start.normalize()
    runtime_end_date = chart_end.normalize()

    runtime_data = runtime_data.loc[
        runtime_data["date"].between(
            runtime_start_date,
            runtime_end_date,
            inclusive="both",
        )
    ].sort_values("date")


if runtime_data.empty:
    st.info(
        "No daily runtime data exists for this station "
        "within the selected playback window."
    )

else:
    runtime_figure = go.Figure()

    if "total_runtime_hr" in runtime_data.columns:
        runtime_figure.add_bar(
            x=runtime_data["date"],
            y=pd.to_numeric(
                runtime_data["total_runtime_hr"],
                errors="coerce",
            ),
            name="Runtime (hr)",
            marker_color=NAVY,
        )

    if "flow_kgal" in runtime_data.columns:
        runtime_figure.add_trace(
            go.Scatter(
                x=runtime_data["date"],
                y=pd.to_numeric(
                    runtime_data["flow_kgal"],
                    errors="coerce",
                ),
                name="Flow (kgal)",
                yaxis="y2",
                line=dict(
                    color=GREEN,
                    width=3,
                ),
            )
        )

    runtime_figure.update_layout(
        height=340,
        hovermode="x unified",
        xaxis=dict(
            range=[
                event_start.normalize(),
                chart_end.normalize(),
            ],
            showgrid=False,
        ),
        yaxis=dict(
            title="Runtime (hr)",
            gridcolor="#EDF1F5",
        ),
        yaxis2=dict(
            title="Pumped volume (kgal)",
            overlaying="y",
            side="right",
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
        legend=dict(
            orientation="h",
            y=1.12,
        ),
    )

    st.plotly_chart(
        runtime_figure,
        use_container_width=True,
    )


st.page_link(
    "app.py",
    label="← Return to system overview",
)