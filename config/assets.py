"""Hull asset registry and mapping constants.

Coordinates are loaded from data/reference/locations.xlsx by the data loader.
The records below are corrected fallbacks so the app remains usable if the
reference workbook is unavailable.
"""
from __future__ import annotations

ASSETS = [
    {"asset_id":"LS A","source_name":"L.S.A.","display_name":"Lift Station A","asset_type":"Pump Station","address":"42 Valley Beach Road, Hull, MA","lat":42.268075862249496,"lon":-70.84569953165689,"capacity_gpm":200,"force_main":"4 in / 840 ft"},
    {"asset_id":"PS 1","source_name":"P.S.#1","display_name":"Pump Station 1","asset_type":"Pump Station","address":"157 Atlantic Avenue, Hull, MA","lat":42.267246611002776,"lon":-70.83845900733085,"capacity_gpm":450,"force_main":"8 in / 2,050 ft"},
    {"asset_id":"PS 3","source_name":"P.S.#3","display_name":"Pump Station 3","asset_type":"Pump Station","address":"George Washington Boulevard, Hull, MA","lat":42.26176646571457,"lon":-70.86162590794065,"capacity_gpm":1700,"force_main":"14 in / 4,625 ft"},
    {"asset_id":"PS 4","source_name":"P.S.#4","display_name":"Pump Station 4","asset_type":"Pump Station","address":"13A Marginal Road, Hull, MA","lat":42.27203165624249,"lon":-70.86677506545786,"capacity_gpm":800,"force_main":"8 in / 1,000 ft"},
    {"asset_id":"PS 5","source_name":"P.S.#5","display_name":"Pump Station 5","asset_type":"Pump Station","address":"70 Draper Avenue, Hull, MA","lat":42.28348210452741,"lon":-70.87702876408703,"capacity_gpm":1600,"force_main":"14 in / 530 ft"},
    {"asset_id":"PS 6","source_name":"P.S.#6","display_name":"Pump Station 6","asset_type":"Pump Station","address":"L Street Playground / 765 Nantasket Avenue, Hull, MA","lat":42.298521647935246,"lon":-70.88314532889895,"capacity_gpm":670,"force_main":"6 in / 60 ft"},
    {"asset_id":"PS 9","source_name":"P.S.#9","display_name":"Pump Station 9","asset_type":"Pump Station","address":"165 Main Street, Hull, MA","lat":42.30473361284288,"lon":-70.91853779233249,"capacity_gpm":650,"force_main":"14 in / 5,030 ft"},
    {"asset_id":"PS D","source_name":"P.S. D","display_name":"D Street Pump Station","asset_type":"Pump Station","address":"25 Cadish Avenue, Hull, MA","lat":42.29389293767283,"lon":-70.8859238066091,"capacity_gpm":None,"force_main":None},
    {"asset_id":"WWTP","source_name":"WWTP","display_name":"Hull Wastewater Treatment Plant","asset_type":"Treatment Plant","address":"1111 Nantasket Avenue, Hull, MA","lat":42.30569560981366,"lon":-70.89991779233718,"capacity_gpm":None,"force_main":"Receiving facility"},
]

HULL_LAT = 42.285
HULL_LON = -70.880
NOAA_TIDE_STATION = "8443970"  # Boston, MA; prototype reference station
