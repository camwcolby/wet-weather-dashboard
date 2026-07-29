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
from models.plant_response import estimate_plant_response
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
from services.playback import (
    initialize_playback,
    set_playback,
)

MAX_PLAYBACK_HOURS = 72
WET_WELL_REFERENCE_DEPTH_IN = 84.0
SANITARY_TELEMETRY_STATIONS = 7

st.markdown(
    """
    <style>
    .leaflet-tile-pane img {
        image-rendering: auto !important;
        image-rendering: smooth !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def percentile_score(
    value,
    reference: pd.Series,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.95,
) -> float:
    """Normalize a value to 0–1 using reference percentiles."""

    numeric_value = pd.to_numeric(value, errors="coerce")
    numeric_reference = pd.to_numeric(reference, errors="coerce").dropna()

    if pd.isna(numeric_value) or numeric_reference.empty:
        return np.nan

    lower = numeric_reference.quantile(lower_quantile)
    upper = numeric_reference.quantile(upper_quantile)

    if pd.isna(lower) or pd.isna(upper) or upper <= lower:
        return np.nan

    return float(np.clip((numeric_value - lower) / (upper - lower), 0, 1))


def calculate_operational_response_index(
    collection: pd.DataFrame,
    influent: pd.DataFrame,
    assets: pd.DataFrame,
    snapshot: pd.DataFrame,
    as_of: pd.Timestamp,
    rain_value,
    rain_reference: pd.Series,
) -> dict:
    """Calculate a 0–100 screening index from five operational components.

    No nearest timestamp, carry-forward value, or interpolated value is used.
    If any required component is unavailable at the exact playback time, the
    composite index is reported as incomplete.
    """

    components = {
        "Collection excess flow": np.nan,
        "Pumps operating": np.nan,
        "Wet-well utilization": np.nan,
        "Plant influent": np.nan,
        "Rainfall": np.nan,
    }

    # --------------------------------------------------------------
    # Collection-system flow above an hour-of-week dry-weather proxy
    # --------------------------------------------------------------
    if (
        collection is not None
        and not collection.empty
        and {"timestamp", "flow_gpm"}.issubset(collection.columns)
    ):
        total_flow = collection[["timestamp", "flow_gpm"]].copy()
        total_flow["timestamp"] = pd.to_datetime(
            total_flow["timestamp"],
            errors="coerce",
        ).dt.floor("min")
        total_flow["flow_gpm"] = pd.to_numeric(
            total_flow["flow_gpm"],
            errors="coerce",
        )
        total_flow = (
            total_flow.dropna(subset=["timestamp"])
            .groupby("timestamp", as_index=False)["flow_gpm"]
            .sum(min_count=1)
        )

        if not total_flow.empty:
            total_flow["hour_of_week"] = (
                total_flow["timestamp"].dt.dayofweek * 24
                + total_flow["timestamp"].dt.hour
            )
            hourly_baseline = total_flow.groupby(
                "hour_of_week"
            )["flow_gpm"].median()

            total_flow["baseline_gpm"] = total_flow[
                "hour_of_week"
            ].map(hourly_baseline)
            total_flow["excess_gpm"] = (
                total_flow["flow_gpm"] - total_flow["baseline_gpm"]
            ).clip(lower=0)

            exact_total = total_flow.loc[
                total_flow["timestamp"].eq(
                    pd.Timestamp(as_of).floor("min")
                )
            ]

            if not exact_total.empty:
                exact_excess = exact_total.iloc[-1]["excess_gpm"]
                components["Collection excess flow"] = percentile_score(
                    exact_excess,
                    total_flow["excess_gpm"],
                )

    # --------------------------------------------------------------
    # Share of available station pumps operating at exact playback
    # --------------------------------------------------------------
    if snapshot is not None and not snapshot.empty:
        pump_columns = [
            column
            for column in ("pump1_status", "pump2_status")
            if column in snapshot.columns
        ]

        if pump_columns:
            pump_values = pd.concat(
                [
                    pd.to_numeric(
                        snapshot[column],
                        errors="coerce",
                    )
                    for column in pump_columns
                ],
                ignore_index=True,
            ).dropna()

            if not pump_values.empty:
                components["Pumps operating"] = float(
                    np.clip((pump_values > 0).mean(), 0, 1)
                )

        if "level_in" in snapshot.columns:
            levels = pd.to_numeric(
                snapshot["level_in"],
                errors="coerce",
            ).dropna()

            if not levels.empty:
                components["Wet-well utilization"] = float(
                    np.clip(
                        levels.max() / WET_WELL_REFERENCE_DEPTH_IN,
                        0,
                        1,
                    )
                )

    # --------------------------------------------------------------
    # Exact-minute plant influent relative to observed distribution
    # --------------------------------------------------------------
    if (
        influent is not None
        and not influent.empty
        and {"timestamp", "influent_total_mgd"}.issubset(influent.columns)
    ):
        influent_data = influent[
            ["timestamp", "influent_total_mgd"]
        ].copy()
        influent_data["timestamp"] = pd.to_datetime(
            influent_data["timestamp"],
            errors="coerce",
        ).dt.floor("min")
        influent_data["influent_total_mgd"] = pd.to_numeric(
            influent_data["influent_total_mgd"],
            errors="coerce",
        )

        exact_influent = influent_data.loc[
            influent_data["timestamp"].eq(
                pd.Timestamp(as_of).floor("min")
            )
        ]

        if not exact_influent.empty:
            components["Plant influent"] = percentile_score(
                exact_influent.iloc[-1]["influent_total_mgd"],
                influent_data["influent_total_mgd"],
            )

    # --------------------------------------------------------------
    # Selected-day rainfall relative to the historical daily range
    # --------------------------------------------------------------
    components["Rainfall"] = percentile_score(
        rain_value,
        rain_reference,
    )

    weights = {
        "Collection excess flow": 0.30,
        "Pumps operating": 0.20,
        "Wet-well utilization": 0.20,
        "Plant influent": 0.20,
        "Rainfall": 0.10,
    }

    complete = all(pd.notna(value) for value in components.values())

    if complete:
        index_value = 100 * sum(
            components[name] * weights[name]
            for name in weights
        )
    else:
        index_value = np.nan

    if pd.isna(index_value):
        label = "INCOMPLETE"
        color = "#98A5B3"
    elif index_value >= 80:
        label = "CRITICAL RESPONSE"
        color = RED
    elif index_value >= 60:
        label = "HIGH SYSTEM STRESS"
        color = AMBER
    elif index_value >= 40:
        label = "WET WEATHER RESPONSE"
        color = LIME
    elif index_value >= 20:
        label = "ELEVATED"
        color = MUTED_BLUE
    else:
        label = "NORMAL"
        color = GREEN

    return {
        "value": index_value,
        "label": label,
        "color": color,
        "complete": complete,
        "components": components,
        "weights": weights,
    }


def build_ori_timeline_svg(
    timeline: pd.DataFrame,
    current_label: str,
) -> str:
    """Return a compact inline SVG for the ORI hover preview."""

    width = 380
    height = 176
    left = 38
    right = 14
    top = 18
    bottom = 34
    plot_width = width - left - right
    plot_height = height - top - bottom

    clean = timeline.copy()
    clean["timestamp"] = pd.to_datetime(
        clean.get("timestamp"),
        errors="coerce",
    )
    clean["ori"] = pd.to_numeric(clean.get("ori"), errors="coerce")
    clean = clean.dropna(subset=["timestamp", "ori"]).sort_values("timestamp")

    if clean.empty:
        return (
            '<div class="ori-tooltip-empty">'
            'ORI history is unavailable for this playback interval.'
            '</div>'
        )

    # Keep the hover chart light even when a live feed contains many readings.
    if len(clean) > 96:
        positions = np.linspace(0, len(clean) - 1, 96).round().astype(int)
        clean = clean.iloc[np.unique(positions)].copy()

    start_time = clean["timestamp"].iloc[0]
    end_time = clean["timestamp"].iloc[-1]
    span_seconds = max((end_time - start_time).total_seconds(), 1)

    def x_position(timestamp) -> float:
        elapsed = (pd.Timestamp(timestamp) - start_time).total_seconds()
        return left + (elapsed / span_seconds) * plot_width

    def y_position(value) -> float:
        return top + (1 - np.clip(float(value), 0, 100) / 100) * plot_height

    bands = [
        (60, 100, "rgba(213,67,67,.08)"),
        (40, 60, "rgba(239,159,39,.10)"),
        (0, 40, "rgba(65,170,92,.08)"),
    ]
    band_rects = []
    for low, high, fill in bands:
        y_top = y_position(high)
        y_bottom = y_position(low)
        band_rects.append(
            f'<rect x="{left}" y="{y_top:.1f}" width="{plot_width}" '
            f'height="{(y_bottom-y_top):.1f}" fill="{fill}" />'
        )

    grid = []
    for tick in (0, 20, 40, 60, 80, 100):
        y = y_position(tick)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="currentColor" stroke-opacity=".12" stroke-width="1" />'
        )
        if tick in (0, 40, 60, 100):
            grid.append(
                f'<text x="{left-7}" y="{y+3.5:.1f}" text-anchor="end" '
                'font-size="9" fill="currentColor" opacity=".62">'
                f'{tick}</text>'
            )

    points = " ".join(
        f'{x_position(row.timestamp):.1f},{y_position(row.ori):.1f}'
        for row in clean.itertuples(index=False)
    )

    last = clean.iloc[-1]
    last_x = x_position(last["timestamp"])
    last_y = y_position(last["ori"])
    start_label = start_time.strftime("%-I %p") if hasattr(start_time, 'strftime') else "Start"
    end_label = end_time.strftime("%-I:%M %p") if hasattr(end_time, 'strftime') else "Current"
    # Windows-compatible fallback for strftime without %-I.
    start_label = start_time.strftime("%I %p").lstrip("0")
    end_label = end_time.strftime("%I:%M %p").lstrip("0")

    return (
        f'<svg class="ori-mini-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Operational Response Index timeline">'
        + "".join(band_rects)
        + "".join(grid)
        + f'<polyline points="{points}" fill="none" stroke="currentColor" '
          'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
        + f'<line x1="{last_x:.1f}" y1="{top}" x2="{last_x:.1f}" '
          f'y2="{height-bottom}" stroke="currentColor" stroke-opacity=".25" '
          'stroke-dasharray="3 3" />'
        + f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" '
          'fill="white" stroke="currentColor" stroke-width="3" />'
        + f'<text x="{left}" y="{height-10}" font-size="9" fill="currentColor" '
          f'opacity=".62">{escape(start_label)}</text>'
        + f'<text x="{width-right}" y="{height-10}" text-anchor="end" '
          f'font-size="9" fill="currentColor" opacity=".62">{escape(end_label)}</text>'
        + f'<text x="{last_x-7:.1f}" y="{max(last_y-9, 12):.1f}" '
          'text-anchor="end" font-size="10" font-weight="700" '
          f'fill="currentColor">{last["ori"]:.0f}</text>'
        + '</svg>'
        + f'<div class="ori-tooltip-caption">Storm start to {escape(current_label)}</div>'
    )


@st.cache_data(show_spinner=False, max_entries=24)
def build_full_ori_timeline(
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    rain_value: float,
    rain_reference_q10: float,
    rain_reference_q95: float,
    data_version: str,
    _collection: pd.DataFrame,
    _influent: pd.DataFrame,
) -> pd.DataFrame:
    """Build one vectorized ORI series for the entire playback event.

    The large DataFrames use leading-underscore arguments so Streamlit does
    not re-hash every telemetry row on each slider movement. ``data_version``
    is the explicit cache key that changes when the underlying source grows or
    its latest timestamp changes. The result is calculated once per event and
    then sliced to the selected playback time.
    """
    del data_version  # Used only as an explicit cache key.

    columns = [
        "timestamp",
        "ori",
        "collection_score",
        "pumps_score",
        "wet_well_score",
        "influent_score",
        "rain_score",
    ]
    if _collection is None or _collection.empty:
        return pd.DataFrame(columns=columns)

    event_start = pd.Timestamp(event_start).floor("min")
    event_end = pd.Timestamp(event_end).floor("min")

    collection = _collection.copy()
    collection["timestamp"] = pd.to_datetime(
        collection["timestamp"], errors="coerce"
    ).dt.floor("min")
    collection = collection.dropna(subset=["timestamp"])

    # --------------------------------------------------------------
    # Collection-system excess flow and its historical reference
    # --------------------------------------------------------------
    flow = collection[["timestamp", "flow_gpm"]].copy()
    flow["flow_gpm"] = pd.to_numeric(flow["flow_gpm"], errors="coerce")
    flow = (
        flow.groupby("timestamp", as_index=False)["flow_gpm"]
        .sum(min_count=1)
        .sort_values("timestamp")
    )
    flow["hour_of_week"] = (
        flow["timestamp"].dt.dayofweek * 24 + flow["timestamp"].dt.hour
    )
    hourly_baseline = flow.groupby("hour_of_week")["flow_gpm"].median()
    flow["baseline_gpm"] = flow["hour_of_week"].map(hourly_baseline)
    flow["excess_gpm"] = (flow["flow_gpm"] - flow["baseline_gpm"]).clip(lower=0)

    def scaled(values: pd.Series, reference: pd.Series) -> pd.Series:
        numeric_reference = pd.to_numeric(reference, errors="coerce").dropna()
        if numeric_reference.empty:
            return pd.Series(np.nan, index=values.index, dtype="float64")
        lower = numeric_reference.quantile(0.10)
        upper = numeric_reference.quantile(0.95)
        if pd.isna(lower) or pd.isna(upper) or upper <= lower:
            return pd.Series(np.nan, index=values.index, dtype="float64")
        return ((pd.to_numeric(values, errors="coerce") - lower) / (upper - lower)).clip(0, 1)

    event_flow = flow.loc[
        flow["timestamp"].between(event_start, event_end),
        ["timestamp", "excess_gpm"],
    ].copy()
    event_flow["collection_score"] = scaled(
        event_flow["excess_gpm"], flow["excess_gpm"]
    )

    # --------------------------------------------------------------
    # Exact-minute pump share and maximum wet-well utilization
    # --------------------------------------------------------------
    event_collection = collection.loc[
        collection["timestamp"].between(event_start, event_end)
    ].copy()

    pump_columns = [
        column
        for column in ("pump1_status", "pump2_status")
        if column in event_collection.columns
    ]
    if pump_columns:
        pump_long = event_collection[["timestamp", *pump_columns]].melt(
            id_vars="timestamp", value_vars=pump_columns, value_name="pump_status"
        )
        pump_long["pump_status"] = pd.to_numeric(
            pump_long["pump_status"], errors="coerce"
        )
        pump_scores = (
            pump_long.dropna(subset=["pump_status"])
            .assign(is_running=lambda frame: (frame["pump_status"] > 0).astype(float))
            .groupby("timestamp", as_index=False)["is_running"]
            .mean()
            .rename(columns={"is_running": "pumps_score"})
        )
    else:
        pump_scores = pd.DataFrame(columns=["timestamp", "pumps_score"])

    if "level_in" in event_collection.columns:
        level_data = event_collection[["timestamp", "level_in"]].copy()
        level_data["level_in"] = pd.to_numeric(level_data["level_in"], errors="coerce")
        wet_well = (
            level_data.groupby("timestamp", as_index=False)["level_in"]
            .max()
            .rename(columns={"level_in": "max_level_in"})
        )
        wet_well["wet_well_score"] = (
            wet_well["max_level_in"] / WET_WELL_REFERENCE_DEPTH_IN
        ).clip(0, 1)
    else:
        wet_well = pd.DataFrame(columns=["timestamp", "wet_well_score"])

    # --------------------------------------------------------------
    # Exact-minute plant influent and its historical reference
    # --------------------------------------------------------------
    if (
        _influent is not None
        and not _influent.empty
        and {"timestamp", "influent_total_mgd"}.issubset(_influent.columns)
    ):
        influent = _influent[["timestamp", "influent_total_mgd"]].copy()
        influent["timestamp"] = pd.to_datetime(
            influent["timestamp"], errors="coerce"
        ).dt.floor("min")
        influent["influent_total_mgd"] = pd.to_numeric(
            influent["influent_total_mgd"], errors="coerce"
        )
        influent = influent.dropna(subset=["timestamp"])
        influent_reference = influent["influent_total_mgd"]
        event_influent = (
            influent.loc[influent["timestamp"].between(event_start, event_end)]
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        event_influent["influent_score"] = scaled(
            event_influent["influent_total_mgd"], influent_reference
        )
        event_influent = event_influent[["timestamp", "influent_score"]]
    else:
        event_influent = pd.DataFrame(columns=["timestamp", "influent_score"])

    # --------------------------------------------------------------
    # Daily rainfall score is constant for the selected event
    # --------------------------------------------------------------
    rain_numeric = pd.to_numeric(rain_value, errors="coerce")
    if (
        pd.isna(rain_numeric)
        or pd.isna(rain_reference_q10)
        or pd.isna(rain_reference_q95)
        or rain_reference_q95 <= rain_reference_q10
    ):
        rain_score = np.nan
    else:
        rain_score = float(
            np.clip(
                (rain_numeric - rain_reference_q10)
                / (rain_reference_q95 - rain_reference_q10),
                0,
                1,
            )
        )

    timeline = event_flow[["timestamp", "collection_score"]].copy()
    timeline = timeline.merge(pump_scores, on="timestamp", how="left")
    timeline = timeline.merge(
        wet_well[["timestamp", "wet_well_score"]], on="timestamp", how="left"
    )
    timeline = timeline.merge(event_influent, on="timestamp", how="left")
    timeline["rain_score"] = rain_score

    component_columns = [
        "collection_score",
        "pumps_score",
        "wet_well_score",
        "influent_score",
        "rain_score",
    ]
    complete = timeline[component_columns].notna().all(axis=1)
    timeline["ori"] = np.where(
        complete,
        100
        * (
            timeline["collection_score"] * 0.30
            + timeline["pumps_score"] * 0.20
            + timeline["wet_well_score"] * 0.20
            + timeline["influent_score"] * 0.20
            + timeline["rain_score"] * 0.10
        ),
        np.nan,
    )
    return timeline[columns].sort_values("timestamp").reset_index(drop=True)


def build_ori_timeline(
    collection: pd.DataFrame,
    influent: pd.DataFrame,
    assets: pd.DataFrame,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    as_of: pd.Timestamp,
    rain_value,
    rain_reference: pd.Series,
) -> pd.DataFrame:
    """Return the cached full-event ORI timeline through ``as_of`` only."""
    del assets  # Retained in the signature for backward compatibility.

    collection_max = pd.to_datetime(
        collection.get("timestamp"), errors="coerce"
    ).max() if collection is not None and not collection.empty else pd.NaT
    influent_max = pd.to_datetime(
        influent.get("timestamp"), errors="coerce"
    ).max() if influent is not None and not influent.empty else pd.NaT
    data_version = (
        f"c:{len(collection) if collection is not None else 0}:"
        f"{collection_max}|i:{len(influent) if influent is not None else 0}:"
        f"{influent_max}"
    )

    rain_numeric = pd.to_numeric(rain_reference, errors="coerce").dropna()
    rain_q10 = rain_numeric.quantile(0.10) if not rain_numeric.empty else np.nan
    rain_q95 = rain_numeric.quantile(0.95) if not rain_numeric.empty else np.nan

    full_timeline = build_full_ori_timeline(
        event_start=pd.Timestamp(event_start),
        event_end=pd.Timestamp(event_end),
        rain_value=float(rain_value) if pd.notna(rain_value) else np.nan,
        rain_reference_q10=float(rain_q10) if pd.notna(rain_q10) else np.nan,
        rain_reference_q95=float(rain_q95) if pd.notna(rain_q95) else np.nan,
        data_version=data_version,
        _collection=collection,
        _influent=influent,
    )

    cutoff = pd.Timestamp(as_of).floor("min")
    return full_timeline.loc[
        pd.to_datetime(full_timeline["timestamp"], errors="coerce") <= cutoff
    ].copy()


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

initialize_playback(
    default_day=latest_storm,
    default_playback_hour=48,
)

mode = st.sidebar.radio(
    "Operating view",
    ["Latest significant storm", "Latest common-data date", "Custom date"],
    key="_playback_mode_widget",
)

if mode == "Latest significant storm":
    selected_day = pd.Timestamp(latest_storm).normalize()

elif mode == "Latest common-data date":
    selected_day = pd.Timestamp(
        max(common_dates) if common_dates else max_date
    ).normalize()

else:
    selected_day = pd.Timestamp(
        st.sidebar.date_input(
            "Date",
            min_value=min_date,
            max_value=max_date,
            key="_custom_date_widget",
        )
    ).normalize()

row_match = ranked.loc[
    ranked["event_date"].dt.normalize() == selected_day
]
row = (
    row_match.iloc[0]
    if not row_match.empty
    else pd.Series(dtype="object")
)

saved_hour = int(
    st.session_state.get(
        "playback_hour",
        48,
    )
)
saved_hour = max(
    0,
    min(saved_hour, MAX_PLAYBACK_HOURS),
)

if "_playback_hour_widget" not in st.session_state:
    st.session_state["_playback_hour_widget"] = saved_hour
else:
    current_widget_hour = int(
        st.session_state["_playback_hour_widget"]
    )
    if current_widget_hour < 0 or current_widget_hour > MAX_PLAYBACK_HOURS:
        st.session_state["_playback_hour_widget"] = saved_hour

playback_hour = st.sidebar.slider(
    "Storm playback hour",
    min_value=0,
    max_value=MAX_PLAYBACK_HOURS,
    step=1,
    key="_playback_hour_widget",
)

playback = set_playback(
    selected_day=selected_day,
    playback_hour=playback_hour,
    mode=mode,
)

selected_day = playback.selected_day
event_start = playback.event_start
event_end = playback.event_end
as_of = playback.as_of

render_header(
    f"Historical Playback | Trigger {selected_day:%b %d, %Y} | As of {as_of:%b %d, %I:%M %p}"
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

plant_response = estimate_plant_response(
    influent=influent,
    history=history_2026,
    event_start=event_start,
    as_of=as_of,
    event_hours=MAX_PLAYBACK_HOURS,
)

if plant_response.response_detected:
    response_primary = f"Detected after {plant_response.response_lag_hr:.1f} hr"
    response_secondary = (
        plant_response.response_onset.strftime("Response began %b %d at %I:%M %p")
        if pd.notna(plant_response.response_onset)
        else "Sustained plant-flow increase detected"
    )
else:
    response_primary = "Not yet detected"
    if pd.notna(plant_response.response_threshold_mgd):
        response_secondary = (
            f"Watching for sustained flow above "
            f"{plant_response.response_threshold_mgd:.2f} MGD"
        )
    else:
        response_secondary = "Insufficient influent data at this playback time"

if pd.notna(plant_response.estimated_peak_time):
    estimated_peak_time_text = plant_response.estimated_peak_time.strftime(
        "%b %d at %I:%M %p"
    )
    if plant_response.estimated_hours_to_peak <= 0.05:
        peak_time_secondary = "Peak window reached or passed"
    else:
        peak_time_secondary = (
            f"About {plant_response.estimated_hours_to_peak:.1f} hr remaining"
        )
else:
    estimated_peak_time_text = "Unavailable"
    peak_time_secondary = "Not enough earlier telemetry for an analog estimate"

estimated_peak_flow_text = display_value(
    plant_response.estimated_peak_flow_mgd,
    2,
    " MGD",
)
peak_flow_secondary = (
    f"{plant_response.confidence} confidence | "
    f"{plant_response.analog_count} prior analog"
    f"{'s' if plant_response.analog_count != 1 else ''}"
)

# Asset names are loaded before the operational cards so station IDs can be
# translated into operator-friendly labels.
assets = load_asset_locations()
asset_name_lookup = (
    assets.set_index("asset_id")["display_name"].astype(str).to_dict()
    if not assets.empty
    else {}
)

# Exact-minute collection snapshot. Empty means genuinely unavailable.
snap = exact_snapshot(collection, as_of)
telemetry_available = not snap.empty

max_level = np.nan
highest_wet_well_station = "Unavailable"

pump_on_names: list[str] = []
pump_off_names: list[str] = []
pump_unknown_names: list[str] = []

generator_on_names: list[str] = []
generator_off_names: list[str] = []
generator_unknown_names: list[str] = []

generator_status_candidates = (
    "generator_status",
    "emergency_generator_status",
    "genset_status",
    "generator_running",
)
generator_status_column = next(
    (
        column
        for column in generator_status_candidates
        if column in snap.columns
    ),
    None,
)

if telemetry_available:
    station_snapshot = snap.copy()
    station_snapshot["_display_name"] = station_snapshot["asset_id"].map(
        asset_name_lookup
    ).fillna(station_snapshot["asset_id"].astype(str))

    if "level_in" in station_snapshot.columns:
        station_snapshot["_level_numeric"] = pd.to_numeric(
            station_snapshot["level_in"],
            errors="coerce",
        )
        valid_levels = station_snapshot.dropna(
            subset=["_level_numeric"]
        )
        if not valid_levels.empty:
            highest_row = valid_levels.sort_values(
                "_level_numeric",
                ascending=False,
            ).iloc[0]
            max_level = float(highest_row["_level_numeric"])
            highest_wet_well_station = str(
                highest_row["_display_name"]
            )

    for _, station_row in station_snapshot.iterrows():
        station_name = str(station_row["_display_name"])

        p1 = pd.to_numeric(
            station_row.get("pump1_status"),
            errors="coerce",
        )
        p2 = pd.to_numeric(
            station_row.get("pump2_status"),
            errors="coerce",
        )

        if pd.isna(p1) and pd.isna(p2):
            pump_unknown_names.append(station_name)
        elif (0 if pd.isna(p1) else p1) + (0 if pd.isna(p2) else p2) > 0:
            pump_on_names.append(station_name)
        else:
            pump_off_names.append(station_name)

        if generator_status_column is not None:
            generator_value = pd.to_numeric(
                station_row.get(generator_status_column),
                errors="coerce",
            )
            if pd.isna(generator_value):
                generator_unknown_names.append(station_name)
            elif generator_value > 0:
                generator_on_names.append(station_name)
            else:
                generator_off_names.append(station_name)

running = len(pump_on_names)
reporting_stations = len(pump_on_names) + len(pump_off_names)
running_text = (
    f"{running} ON / {len(pump_off_names)} OFF"
    if telemetry_available
    else "Unavailable"
)

pump_on_text = ", ".join(pump_on_names) if pump_on_names else "None"
pump_off_text = ", ".join(pump_off_names) if pump_off_names else "None"

if generator_status_column is None:
    generator_summary = "Signal not included in prototype data"
    generator_detail = (
        "This card will populate when the live SCADA feed provides a "
        "normalized generator-status field."
    )
else:
    generator_summary = (
        f"{len(generator_on_names)} ON / "
        f"{len(generator_off_names)} OFF"
    )
    generator_detail = (
        f"ON: {', '.join(generator_on_names) if generator_on_names else 'None'}"
        f"<br>OFF: {', '.join(generator_off_names) if generator_off_names else 'None'}"
    )


ori = calculate_operational_response_index(
    collection=collection,
    influent=influent,
    assets=assets if "assets" in locals() else pd.DataFrame(),
    snapshot=snap,
    as_of=as_of,
    rain_value=rain_val,
    rain_reference=history_2026["rain_in"],
)

ori_timeline = build_ori_timeline(
    collection=collection,
    influent=influent,
    assets=assets,
    event_start=event_start,
    event_end=event_end,
    as_of=as_of,
    rain_value=rain_val,
    rain_reference=history_2026["rain_in"],
)
ori_timeline_svg = build_ori_timeline_svg(
    ori_timeline,
    as_of.strftime("%b %d at %I:%M %p").replace(" 0", " "),
)

ori_value = ori["value"]
ori_text = (
    f"{ori_value:.0f}"
    if pd.notna(ori_value)
    else "Incomplete"
)

rain_text = display_value(rain_val, 2, " in", "Unavailable")

st.markdown(
    f'<div class="section-intro">'
    f'<div><div class="section-kicker">Current system snapshot</div>'
    f'<div class="section-note">Exact-minute operating conditions and storm outlook</div></div>'
    f'<div class="status-badge" style="color:{ori["color"]};border-color:{ori["color"]};">'
    f'{escape(ori["label"].title())} · ORI {ori_text}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Three-level operator color treatment for the summary card.
if pd.isna(ori_value):
    ori_card_color = "#98A5B3"
    ori_card_background = "#F3F4F6"
elif ori_value < 40:
    ori_card_color = GREEN
    ori_card_background = "#ECFDF3"
elif ori_value < 60:
    ori_card_color = AMBER
    ori_card_background = "#FFF7E6"
else:
    ori_card_color = RED
    ori_card_background = "#FEF0F0"

overview_cols = st.columns(4, gap="medium")

with overview_cols[0]:
    st.markdown(
        f'<div class="kpi kpi-primary">'
        f'<div class="kpi-label">Pump stations</div>'
        f'<div class="split-count">'
        f'<div><span class="count-number">{running}</span><span class="count-label">ON</span></div>'
        f'<div class="count-divider"></div>'
        f'<div><span class="count-number">{len(pump_off_names)}</span><span class="count-label">OFF</span></div>'
        f'</div>'
        f'<div class="status-list"><span class="status-key">ON</span>{escape(pump_on_text)}</div>'
        f'<div class="status-list"><span class="status-key">OFF</span>{escape(pump_off_text)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with overview_cols[1]:
    generator_value = (
        generator_summary
        if generator_status_column is not None
        else "Awaiting SCADA signal"
    )
    generator_sub = (
        generator_detail
        if generator_status_column is not None
        else "Generator state will appear automatically when the live feed is connected."
    )
    st.markdown(
        f'<div class="kpi kpi-primary">'
        f'<div class="kpi-label">Emergency generators</div>'
        f'<div class="kpi-value kpi-value-compact">{escape(generator_value)}</div>'
        f'<div class="kpi-sub">{generator_sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with overview_cols[2]:
    highest_level_text = display_value(
        max_level,
        1,
        " in",
        "Unavailable",
    )
    st.markdown(
        f'<div class="kpi kpi-primary">'
        f'<div class="kpi-label">Highest wet well</div>'
        f'<div class="kpi-value">{highest_level_text}</div>'
        f'<div class="kpi-sub kpi-sub-strong">{escape(highest_wet_well_station)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with overview_cols[3]:
    st.markdown(
        f'<div class="ori-hover-wrap" tabindex="0" '
        f'aria-label="Operational response. Hover or focus to view timeline.">'
        f'<div class="kpi kpi-primary ori-hover-card" '
        f'style="border-color:{ori_card_color};background:{ori_card_background};color:{ori_card_color}">'
        f'<div class="kpi-label">Operational response</div>'
        f'<div class="kpi-value" style="color:{ori_card_color}">{ori_text}'
        f'<span class="value-denominator"> / 100</span></div>'
        f'<div class="response-pill" style="color:{ori_card_color};border-color:{ori_card_color}">'
        f'{escape(ori["label"].title())}</div>'
        f'<div class="hover-hint">Hover for storm timeline</div>'
        f'</div>'
        f'<div class="ori-tooltip" style="color:{ori_card_color}">'
        f'<div class="ori-tooltip-title">Operational response timeline</div>'
        f'{ori_timeline_svg}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

supporting_kpis = st.columns(4, gap="medium")
supporting_items = [
    ("Rainfall trigger", rain_text, "Selected trigger date"),
    ("Plant response", response_primary, response_secondary),
    ("Estimated peak time", estimated_peak_time_text, peak_time_secondary),
    ("Estimated peak flow", estimated_peak_flow_text, peak_flow_secondary),
]

for column, (label, value, subtitle) in zip(
    supporting_kpis,
    supporting_items,
):
    column.markdown(
        f'<div class="kpi kpi-secondary">'
        f'<div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value kpi-value-secondary">{escape(str(value))}</div>'
        f'<div class="kpi-sub">{escape(str(subtitle))}</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("How the Operational Response Index is calculated"):
    st.markdown(
        """
The **Operational Response Index (ORI)** is a 0–100 screening indicator of
how strongly the collection and treatment system is responding at the exact
playback moment. It is not a permit limit, failure probability, percent of
system capacity, or calibrated hydraulic-model result.

The current weighting is:

- **30%** collection-system flow above its hour-of-week baseline
- **20%** share of available station pumps operating
- **20%** highest wet-well utilization
- **20%** exact-minute plant influent relative to observed conditions
- **10%** selected-day rainfall relative to the historical daily range

The ORI response bands are:

- **Normal:** below 20
- **Elevated:** 20 to less than 40
- **Wet Weather Response:** 40 to less than 60
- **High System Stress:** 60 to less than 80
- **Critical Response:** 80 or greater

No nearest timestamp, carry-forward value, or interpolation is used. The
composite is shown as **Incomplete** when any required component is unavailable.

**Pump-station map status is separate from the ORI.** It currently divides
wet-well level by a provisional 84-inch reference depth. Station-specific
operating and alarm elevations should replace that reference when available.

**I/I response factor is also separate from the ORI.** It represents estimated
excess influent volume divided by rainfall, expressed as MG per inch.
        """
    )

    component_rows = []

    for component_name, component_score in ori["components"].items():
        component_rows.append(
            {
                "Component": component_name,
                "Weight": f'{ori["weights"][component_name] * 100:.0f}%',
                "Score": (
                    f"{component_score * 100:.0f}"
                    if pd.notna(component_score)
                    else "Unavailable"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(component_rows),
        hide_index=True,
        use_container_width=True,
    )

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
    yaxis=dict(title="Scaled operational signals", gridcolor="#EDF1F5"),
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