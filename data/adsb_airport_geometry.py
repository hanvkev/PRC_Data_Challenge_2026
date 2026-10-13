"""Geometry helpers copied unchanged from 04_add_features.ipynb (formerly 05).
Matching parity is checked by tests; no notebook cells execute at import.
"""
from pathlib import Path
import numpy as np
import pandas as pd
data_dir = Path(__file__).resolve().parent
import json
import re
import networkx as nx
from pyproj import Geod
from xplane_airports import gateway
from traffic.data import airports as osm_airports

GEOMETRY_FEATURES = [
    "stand_to_runway_distance_m", "stand_runway_parallel_m",
    "stand_runway_perpendicular_m", "taxi_route_distance_m",
    "runway_length_m", "runway_width_m",
]
GEOMETRY_KEYS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
GEOD = Geod(ellps="WGS84")
STAND_SNAP_M, RUNWAY_SNAP_M = 250.0, 500.0
STAND_CONFLICT_M = 100.0  # Maximum pairwise distance among OSM or X-Plane candidates.


def geometry_name(value):
    return None if pd.isna(value) else " ".join(str(value).strip().upper().split()) or None


def geometry_runway(value):
    value = geometry_name(value)
    match = re.fullmatch(r"(?:RWY\s*)?(\d{1,2})([LRCWST]?)(?:\.0)?", value or "")
    if match and 1 <= int(match[1]) <= 36:
        return f"{int(match[1]):02d}{match[2]}"
    return value


def geometry_keys(inputs):
    keys = inputs[GEOMETRY_KEYS].copy()
    keys["ADEP_mvt"] = keys.ADEP_mvt.map(geometry_name)
    keys["STAND_mvt"] = keys.STAND_mvt.map(geometry_name)
    keys["RUNWAY_mvt"] = keys.RUNWAY_mvt.map(geometry_runway)
    return keys


def geo_distance(a, b):
    return GEOD.inv(*a, *b)[2]  # coordinates are (longitude, latitude)


def load_gateway_geometry(icao):
    metadata = gateway.airport(icao, retries_on_error=2)
    scenery_id = metadata.get("recommendedSceneryId")
    if not scenery_id:
        print(icao, "has no recommended scenery; geometry remains missing")
        return ""
    cache = data_dir / f"xplane_{icao}_{int(scenery_id)}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
        assert payload["icao"] == icao and payload["scenery_id"] == int(scenery_id)
    else:
        pack = gateway.scenery_pack(int(scenery_id), retries_on_error=2)
        payload = {"icao": icao, "scenery_id": int(scenery_id),
                   "apt_dat": str(pack.apt), "readme": pack.readme, "copying": pack.copying}
        temporary = cache.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload))
        temporary.replace(cache)
    print(icao, "recommended scenery", scenery_id, "cache:", cache.name)
    return payload["apt_dat"]


def parse_airport_geometry(text):
    stands, runways, nodes, edges = {}, {}, {}, []
    ambiguous = set()
    stand_candidates = []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] in ("1300", "15"):
            name = geometry_name(" ".join(t[6:] if t[0] == "1300" else t[4:]))
            point = (float(t[2]), float(t[1]))
            if name is not None and np.isfinite(point).all():
                stand_candidates.append({"ref": name, "longitude": point[0], "latitude": point[1]})
            if name in stands and stands[name] != point:
                ambiguous.add(name)
            stands[name] = point
        elif t[0] == "100":
            ends = [geometry_runway(t[8]), geometry_runway(t[17])]
            points = [(float(t[10]), float(t[9])), (float(t[19]), float(t[18]))]
            length = geo_distance(*points)
            displacements = [float(t[11]), float(t[20])]
            assert length > sum(displacements) and min(displacements) >= 0
            thresholds = []
            for i in (0, 1):
                bearing = GEOD.inv(*points[i], *points[1-i])[0]
                thresholds.append(GEOD.fwd(*points[i], bearing, displacements[i])[:2])
            for i, end in enumerate(ends):
                assert end not in runways, f"Duplicate runway end {end}"
                runways[end] = {"pair": tuple(sorted(ends)), "threshold": thresholds[i],
                    "opposite": thresholds[1-i], "runway_length_m": length,
                    "runway_width_m": float(t[1])}
        elif t[0] == "1201":
            nodes[int(t[4])] = (float(t[2]), float(t[1]))
        elif t[0] == "1202":
            edges.append((int(t[1]), int(t[2]), t[3], t[4], " ".join(t[5:])))
    for name in ambiguous | {None}:
        stands.pop(name, None)
    graph, taxi_nodes, runway_nodes = nx.DiGraph(), set(), {}
    for u, v, direction, kind, name in edges:
        if u not in nodes or v not in nodes:
            raise ValueError(f"Taxi edge references missing node: {u}, {v}")
        if direction not in ("oneway", "twoway"):
            raise ValueError(f"Unknown taxi direction: {direction}")
        weight = geo_distance(nodes[u], nodes[v])
        for a, b in ([(u, v), (v, u)] if direction == "twoway" else [(u, v)]):
            if not graph.has_edge(a, b) or weight < graph[a][b]["weight"]:
                graph.add_edge(a, b, weight=weight)
        if kind.startswith("taxiway"):
            taxi_nodes.update((u, v))
        elif kind == "runway":
            pair = tuple(sorted(geometry_runway(x) for x in name.split("/")))
            runway_nodes.setdefault(pair, set()).update((u, v))
    starts = sorted(u for u in taxi_nodes if graph.out_degree(u) > 0)
    for runway in runways.values():
        candidates = sorted(runway_nodes.get(runway["pair"], set()) & taxi_nodes)
        destination, connector = nearest_geometry_node(runway["threshold"], candidates, nodes, RUNWAY_SNAP_M)
        runway["connector"] = connector
        runway["paths"] = (nx.single_source_dijkstra_path_length(
            graph.reverse(copy=False), destination, weight="weight") if destination is not None else {})
    return {"stands": stands, "runways": runways, "nodes": nodes, "starts": starts,
            "ambiguous": sorted(x for x in ambiguous if x is not None),
            "stand_candidates": pd.DataFrame(stand_candidates, columns=["ref", "longitude", "latitude"])}


def nearest_geometry_node(point, candidates, nodes, limit):
    if not candidates:
        return None, np.nan
    distances = [geo_distance(point, nodes[node]) for node in candidates]
    i = int(np.argmin(distances))
    return (candidates[i], distances[i]) if distances[i] <= limit else (None, np.nan)


def geometry_combination(airport, stand_name, runway_name):
    result = dict.fromkeys(GEOMETRY_FEATURES, np.nan)
    runway = airport["runways"].get(runway_name)
    if runway is None:
        return result
    for column in GEOMETRY_FEATURES[-2:]:
        result[column] = runway[column]
    stand = airport["stands"].get(stand_name)
    if stand is None:
        return result
    threshold = runway["threshold"]
    azimuth, _, distance = GEOD.inv(*threshold, *stand)
    axis = GEOD.inv(*threshold, *runway["opposite"])[0]
    angle = np.deg2rad(azimuth - axis)
    result.update(stand_to_runway_distance_m=distance,
                  stand_runway_parallel_m=distance * np.cos(angle),
                  stand_runway_perpendicular_m=abs(distance * np.sin(angle)))
    start, connector = airport["stand_snaps"][stand_name]
    if start in runway["paths"]:
        result["taxi_route_distance_m"] = connector + runway["paths"][start] + runway["connector"]
    return result

def normalize_identifier(values):
    return values.astype("string").str.strip().str.upper()


def normalize_runway(values):
    return normalize_identifier(values).str.replace(r"^(\d)([LRC]?)$", r"0\1\2", regex=True)


def canonical_stand(values):
    # Only simple stand identifiers: preserve compound refs and variant suffixes.
    values = normalize_identifier(values).str.replace(r"\s+", "", regex=True)
    return values.str.replace(r"^([A-Z]*)0+(?=\d)", r"\1", regex=True)


def resolve_stands(raw, requested, icao, gates=None):
    raw = raw.copy()
    raw[["latitude", "longitude"]] = raw[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").astype(float)
    raw["stand"] = normalize_identifier(raw["ref"])
    raw = raw.dropna(subset=["stand", "latitude", "longitude"])
    raw = raw.loc[np.isfinite(raw["latitude"]) & np.isfinite(raw["longitude"])]
    exact = raw.groupby("stand")[["latitude", "longitude"]].mean()
    raw["canonical"] = canonical_stand(raw["stand"])
    gates = raw.iloc[:0].copy() if gates is None else gates.copy()
    gates[["latitude", "longitude"]] = gates[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").astype(float)
    gates["stand"] = normalize_identifier(gates["ref"])
    gates = gates.dropna(subset=["stand", "latitude", "longitude"])
    gates = gates.loc[np.isfinite(gates.latitude) & np.isfinite(gates.longitude)]
    gates["canonical"] = canonical_stand(gates["stand"])
    records = []
    for stand in requested.dropna().unique():
        canonical = canonical_stand(pd.Series([stand])).iloc[0]
        source = "missing"
        candidates = raw.iloc[:0]
        if stand in exact.index:
            candidates = raw.loc[raw["stand"].eq(stand)]
            source = "exact" if len(candidates) == 1 else "mean_same_ref"
        else:
            candidates = raw.loc[raw["canonical"].eq(canonical)]
            if not candidates.empty:
                source = "normalized_ref"
            elif re.fullmatch(r"[A-Z]*\d+", str(canonical)):
                # Parent stand: average its available letter-suffixed variants.
                candidates = raw.loc[raw["canonical"].str.fullmatch(
                    re.escape(canonical) + r"[A-Z]", na=False
                )]
                if not candidates.empty:
                    source = "mean_variants"
            elif re.fullmatch(r"[A-Z]*\d+[A-Z]", str(canonical)):
                # Missing subposition: approximate with the existing parent stand.
                candidates = raw.loc[raw["canonical"].eq(canonical[:-1])]
                if not candidates.empty:
                    source = "parent_stand_estimate"
            if candidates.empty and re.fullmatch(r"[A-Z]*\d+[A-Z]?", str(canonical)):
                candidates = gates.loc[gates["canonical"].eq(canonical)]
                if not candidates.empty:
                    source = "gate_estimate"
        spread_m = np.nan
        if not candidates.empty:
            points = list(zip(candidates.longitude, candidates.latitude))
            spread_m = max(geo_distance(a, b) for a in points for b in points)
            # Apply the same conflict rule to exact refs and inferred matches.
            if spread_m > STAND_CONFLICT_M:
                source = "ambiguous_spread"
            coordinates = candidates[["latitude", "longitude"]].mean()
        else:
            coordinates = pd.Series({"latitude": np.nan, "longitude": np.nan})
        usable = source not in ("missing", "ambiguous_spread")
        records.append({
            "airport": icao, "stand": stand,
            "latitude": coordinates.latitude if usable else np.nan,
            "longitude": coordinates.longitude if usable else np.nan,
            "stand_position_source": source,
            "matched_refs": "; ".join(sorted(candidates.stand.unique())),
            "position_spread_m": spread_m,
        })
    return pd.DataFrame(records, columns=["airport", "stand", "latitude", "longitude",
                                         "stand_position_source", "matched_refs", "position_spread_m"])




def load_osm_stands(icao):
    # Cache the same OSM parking positions and gates used by the old notebook.
    cache_dir = data_dir / "04_additional_data" / "osm_stands"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"osm_stands_{icao}.json"
    if cache.exists():
        return pd.read_json(cache)
    airport = osm_airports[icao]
    if airport is None:
        raise ValueError(f"Unknown OSM airport: {icao}")
    osm = airport._openstreetmap().data
    columns = ["aeroway", "ref", "latitude", "longitude"]
    osm = osm.reindex(columns=columns)
    positions = osm.loc[osm.aeroway.isin(["parking_position", "gate"])].copy()
    temporary = cache.with_suffix(".json.part")
    positions.to_json(temporary, orient="records", double_precision=15)
    temporary.replace(cache)
    return positions


def xplane_stand_key(value, icao):
    name = geometry_name(value)
    if name is None:
        return None
    # Only strip known scenery decoration around a complete stand identifier.
    name = re.sub(r"^(?:GATE|RAMP|STAND|PARKING)\s+", "", name)
    if icao == "LTFM":
        name = re.sub(r"-\([A-F]\)$", "", name)
    compact = re.sub(r"\s+", "", name)
    if re.fullmatch(r"[A-Z]*\d+[A-Z]?", compact):
        return re.sub(r"^([A-Z]*)0+(?=\d)", r"\1", compact)
    return geometry_name(value)


def resolve_xplane_stand(icao, stand, candidates):
    key = xplane_stand_key(stand, icao)
    matches = candidates.loc[candidates["key"].eq(key)]
    method = "normalized_ref"
    if not matches.empty and matches.ref.eq(stand).all():
        method = "exact" if len(matches) == 1 else "mean_same_ref"
    if matches.empty and re.fullmatch(r"[A-Z]*\d+", key or ""):
        matches = candidates.loc[candidates["key"].str.fullmatch(
            re.escape(key) + r"[A-Z]", na=False)]
        method = "mean_variants"
    if matches.empty:
        return None, "missing", "", np.nan
    points = list(zip(matches.longitude, matches.latitude))
    spread = max(geo_distance(a, b) for a in points for b in points)
    refs = "; ".join(sorted(matches.ref.unique()))
    if spread > STAND_CONFLICT_M:
        return None, "ambiguous_spread", refs, spread
    return tuple(matches[["longitude", "latitude"]].mean()), method, refs, spread


def select_stand_positions(icao, requested, osm, xplane_candidates):
    resolved = resolve_stands(
        osm.loc[osm.aeroway.eq("parking_position")], requested, icao,
        gates=osm.loc[osm.aeroway.eq("gate")],
    )
    resolved["osm_position_source"] = resolved.stand_position_source
    resolved["xplane_match_method"] = "not_used"
    resolved["xplane_matched_refs"] = ""
    resolved["xplane_position_spread_m"] = np.nan
    candidates = xplane_candidates.copy()
    candidates["key"] = candidates.ref.map(lambda value: xplane_stand_key(value, icao))
    for i, row in resolved.iterrows():
        if row.stand_position_source in ("missing", "ambiguous_spread"):
            point, method, refs, spread = resolve_xplane_stand(icao, row.stand, candidates)
            resolved.loc[i, "xplane_match_method"] = method
            resolved.loc[i, "xplane_matched_refs"] = refs
            resolved.loc[i, "xplane_position_spread_m"] = spread
            if point is not None:
                resolved.loc[i, ["longitude", "latitude"]] = point
                resolved.loc[i, "stand_position_source"] = "xplane_fallback"
    stands = {row.stand: (row.longitude, row.latitude)
              for row in resolved.itertuples() if np.isfinite([row.longitude, row.latitude]).all()}
    return stands, resolved
