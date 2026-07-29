from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class PlaybackState:
    selected_day: pd.Timestamp
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    playback_hour: int
    as_of: pd.Timestamp
    mode: str


DEFAULT_MODE = "Latest significant storm"
DEFAULT_PLAYBACK_HOUR = 48


def initialize_playback(
    default_day,
    default_mode: str = DEFAULT_MODE,
    default_playback_hour: int = DEFAULT_PLAYBACK_HOUR,
) -> None:
    """Create durable playback values once per Streamlit session."""

    normalized_day = pd.Timestamp(default_day).normalize()

    if "playback_mode" not in st.session_state:
        st.session_state["playback_mode"] = default_mode

    if "playback_selected_day" not in st.session_state:
        st.session_state["playback_selected_day"] = normalized_day

    if "playback_hour" not in st.session_state:
        st.session_state["playback_hour"] = int(default_playback_hour)

    _restore_widget_keys()


def _restore_widget_keys() -> None:
    """Re-create temporary widget keys after returning from another page."""

    if "_playback_mode_widget" not in st.session_state:
        st.session_state["_playback_mode_widget"] = st.session_state.get(
            "playback_mode",
            DEFAULT_MODE,
        )

    if "_custom_date_widget" not in st.session_state:
        selected_day = pd.Timestamp(
            st.session_state.get(
                "playback_selected_day",
                pd.Timestamp.today().normalize(),
            )
        )
        st.session_state["_custom_date_widget"] = selected_day.date()

    if "_playback_hour_widget" not in st.session_state:
        st.session_state["_playback_hour_widget"] = int(
            st.session_state.get(
                "playback_hour",
                DEFAULT_PLAYBACK_HOUR,
            )
        )



def set_playback(
    *,
    selected_day,
    playback_hour: int,
    mode: str,
) -> PlaybackState:
    """Store one shared playback range for all pages."""

    selected_day = pd.Timestamp(selected_day).normalize()
    playback_hour = max(0, int(playback_hour))

    event_start = selected_day
    as_of = event_start + pd.Timedelta(hours=playback_hour)

    # The visible event range is the selected start through playback time.
    event_end = as_of

    st.session_state["playback_mode"] = mode
    st.session_state["playback_selected_day"] = selected_day
    st.session_state["playback_hour"] = playback_hour
    st.session_state["playback_window_hours"] = playback_hour
    st.session_state["playback_event_start"] = event_start
    st.session_state["playback_event_end"] = event_end
    st.session_state["playback_as_of"] = as_of

    return PlaybackState(
        selected_day=selected_day,
        event_start=event_start,
        event_end=event_end,
        playback_hour=playback_hour,
        as_of=as_of,
        mode=mode,
    )


def get_playback() -> PlaybackState | None:
    """Return the saved playback state, or None if the overview has not set it."""

    required = (
        "playback_selected_day",
        "playback_event_start",
        "playback_event_end",
        "playback_hour",
        "playback_as_of",
        "playback_mode",
    )

    if any(key not in st.session_state for key in required):
        return None

    return PlaybackState(
        selected_day=pd.Timestamp(
            st.session_state["playback_selected_day"]
        ).normalize(),
        event_start=pd.Timestamp(
            st.session_state["playback_event_start"]
        ),
        event_end=pd.Timestamp(
            st.session_state["playback_event_end"]
        ),
        playback_hour=int(
            st.session_state["playback_hour"]
        ),
        as_of=pd.Timestamp(
            st.session_state["playback_as_of"]
        ),
        mode=str(
            st.session_state["playback_mode"]
        ),
    )
