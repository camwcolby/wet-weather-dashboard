from __future__ import annotations


def nws_radar_layer() -> dict[str, str]:
    """Return the NOAA/NWS MRMS base-reflectivity WMS tile configuration.

    MapLibre replaces {bbox-epsg-3857} for each requested map tile.
    The WMS image itself is transparent, allowing the basemap to remain visible.
    """
    tile_url = (
        "https://opengeo.ncep.noaa.gov/geoserver/"
        "conus/conus_bref_qcd/ows"
        "?service=WMS"
        "&version=1.1.1"
        "&request=GetMap"
        "&layers=conus_bref_qcd"
        "&styles="
        "&format=image/png"
        "&transparent=true"
        "&srs=EPSG:3857"
        "&bbox={bbox-epsg-3857}"
        "&width=256"
        "&height=256"
        "&tiled=true"
    )

    return {
        "tile_url": tile_url,
        "attribution": "Radar: NOAA/NWS MRMS",
    }