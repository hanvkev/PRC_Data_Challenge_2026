# Data sources

The challenge asks for external data that are openly accessible, published under an open licence and documented.
The sources below are all that the pipeline uses.

| Source | Used for | Licence | Notebook |
|---|---|---|---|
| [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/data.html): training 2025, `ranking`, `final_ranking` | labels, movements, schedule, NM flight times | provided by the organisers | `data/01_download` |
| [ADSB.lol `globe_history`](https://github.com/adsblol) (2025 and 2026 repositories) | first ground movement of each departure | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), stated in the repositories' README | `data/03_adsb_fetch`, `data/04_adsb_movement` |
| [NOAA/NCEI GHCNh v1](https://doi.org/10.25921/jp3d-3v19) | hourly airport weather (temperature, visibility); Zurich uses station SZI0000LSMD (Dübendorf airfield), because GHCNh has no current station at the airport | CC0-1.0, stated in the NOAA metadata ("NOAA waives any potential copyright and related rights in these data worldwide") | `data/05_features` |
| [OpenStreetMap](https://www.openstreetmap.org/copyright), via the `traffic` library | stand and gate positions | ODbL 1.0, © OpenStreetMap contributors | `data/04_adsb_movement`, `data/05_features` |
| [X-Plane Scenery Gateway](https://gateway.x-plane.com/), recommended scenery per airport via `xplane-airports` | runway thresholds and lengths, taxiway network, stand positions OSM does not know | GNU GPL v2 or later: every scenery pack ships a `COPYING` file with the GPL v2 and a README stating "free software … either version 2 of the License, or (at your option) any later version" (both are cached in `data/xplane_<ICAO>_<scenery>.json`) | `data/04_adsb_movement`, `data/05_features` |

No OpenSky Network data, no EUROCONTROL performance files (OPDI, daily delay series, taxi-time statistics) and no
data published by other teams are used.

The downloaded and derived data are not stored in the repository. ADS-B-derived values stay local; anyone who shares
them keeps the ADSB.lol attribution as the ODbL asks.
