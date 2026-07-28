from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config.assets import HULL_LAT, HULL_LON


MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


@st.cache_data(ttl=1800, show_spinner=False)
def marine_forecast() -> pd.DataFrame:
    """Retrieve the Hull-area marine forecast.

    Tries Hull first, then a nearby offshore point if the shoreline grid cell
    does not return marine data.
    """
    candidate_points = [
        (HULL_LAT, HULL_LON),
        (HULL_LAT, HULL_LON + 0.05),
    ]

    errors: list[str] = []

    for latitude, longitude in candidate_points:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "wave_height,"
                "wave_direction,"
                "wave_period,"
                "wind_wave_height"
            ),
            "length_unit": "imperial",
            "timezone": "America/New_York",
            "forecast_days": 3,
            "cell_selection": "sea",
        }

        try:
            response = requests.get(
                MARINE_URL,
                params=params,
                timeout=20,
                headers={
                    "User-Agent": (
                        "Inframark Hull Wet Weather Dashboard "
                        "operations-dashboard"
                    )
                },
            )

            response.raise_for_status()
            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(
                    payload.get("reason", "Unknown API error")
                )

            hourly = payload.get("hourly")

            if not hourly or not hourly.get("time"):
                raise RuntimeError("No hourly marine forecast returned")

            frame = pd.DataFrame(hourly).rename(
                columns={"time": "timestamp"}
            )

            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"],
                errors="coerce",
            )

            frame = frame.dropna(subset=["timestamp"])
            frame.attrs["source_latitude"] = payload.get("latitude")
            frame.attrs["source_longitude"] = payload.get("longitude")
            frame.attrs["error"] = None

            return frame

        except Exception as exc:
            errors.append(
                f"{latitude:.4f}, {longitude:.4f}: {exc}"
            )

    empty = pd.DataFrame()
    empty.attrs["error"] = " | ".join(errors)
    return empty
