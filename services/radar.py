from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st


RAINVIEWER_METADATA_URL = (
    "https://api.rainviewer.com/public/weather-maps.json"
)


@st.cache_data(ttl=300, show_spinner=False)
def radar_frames() -> list[dict]:
    """Return available RainViewer radar frames.

    RainViewer generally provides recent historical radar frames.
    The newest available frame is used for the operational map.
    """
    try:
        response = requests.get(
            RAINVIEWER_METADATA_URL,
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

        host = payload.get("host")
        past_frames = payload.get("radar", {}).get("past", [])

        if not host or not past_frames:
            return []

        frames: list[dict] = []

        for frame in past_frames:
            frame_time = frame.get("time")
            frame_path = frame.get("path")

            if frame_time is None or not frame_path:
                continue

            frames.append(
                {
                    "time": int(frame_time),
                    "timestamp": pd.Timestamp(
                        datetime.fromtimestamp(
                            int(frame_time),
                            tz=timezone.utc,
                        )
                    ),
                    "tile_url": (
                        f"{host}{frame_path}"
                        "/256/{z}/{x}/{y}/2/1_1.png"
                    ),
                }
            )

        return sorted(frames, key=lambda item: item["time"])

    except requests.RequestException:
        return []
    except (TypeError, ValueError, KeyError):
        return []


def latest_radar_frame() -> dict | None:
    """Return the most recent available radar frame."""
    frames = radar_frames()

    if not frames:
        return None

    return frames[-1]