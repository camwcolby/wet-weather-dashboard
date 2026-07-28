from __future__ import annotations


def nws_radar_layer() -> dict[str, str]:
    """Return a single, consistent configuration for NOAA/NWS MRMS radar.

    Both ``wms_url`` and ``url`` are supplied as aliases so an older local map
    block cannot fail with a KeyError while files are being replaced.
    """
    endpoint = (
        "https://opengeo.ncep.noaa.gov/geoserver/"
        "conus/conus_bref_qcd/ows"
    )

    return {
        "wms_url": endpoint,
        "url": endpoint,
        "layers": "conus_bref_qcd",
        "styles": "",
        "version": "1.1.1",
        "attribution": "Radar: NOAA/NWS MRMS",
    }
