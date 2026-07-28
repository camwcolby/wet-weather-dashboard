from __future__ import annotations


def nws_radar_layer() -> dict[str, str]:
    """Return the official NOAA/NWS CONUS MRMS radar WMS configuration."""

    return {
        "wms_url": (
            "https://opengeo.ncep.noaa.gov/geoserver/"
            "conus/conus_bref_qcd/ows"
        ),
        "layers": "conus_bref_qcd",
        "styles": "",
        "format": "image/png",
        "version": "1.3.0",
        "attribution": "Radar: NOAA/NWS MRMS",
    }
