# Hull Wet Weather Operations Dashboard

Operator-first Streamlit prototype for Hull, Massachusetts. It combines 2026 collection-system and WWTF operating data with National Weather Service forecasts, NOAA tide predictions, and marine forecasts.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app defaults to the latest significant wet-weather event found by combining available rainfall, plant flow, station runtime, and pumped-volume signals. External APIs degrade gracefully when unavailable.

## Data notes

Asset coordinates are approximate and should be replaced with surveyed/GIS coordinates when available. Force-main and sewer geometries are schematic until shapefiles are provided.
