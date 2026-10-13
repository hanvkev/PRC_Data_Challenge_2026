"""Additional 04 features, evaluated in the same pass as movement detection."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import adsb_cleanup_worker as w
import adsb_airport_geometry as g

DISTANCE_COLUMNS = [f"adsb_movement_to_{target}_{kind}_m"
                    for target in ("gate", "runway")
                    for kind in ("direct", "parallel", "perpendicular")]
PROGRESS_COLUMNS = [f"adsb_movement_progress_{kind}" for kind in ("direct", "parallel", "perpendicular")]
EXTRA_SCHEMA = pa.schema([
    ("adsb_first_observation_time", w.UTC), ("adsb_first_observation_to_mvt_sec", pa.float64()),
    ("adsb_movement_longitude", pa.float64()), ("adsb_movement_latitude", pa.float64()),
    ("adsb_geometry_status", pa.string()),
    *[(c, pa.float64()) for c in DISTANCE_COLUMNS + PROGRESS_COLUMNS],
])
SCHEMA = pa.schema(list(w.FEATURE_SCHEMA) + list(EXTRA_SCHEMA))


def canonical_mvt_id(value):
    """Match clean float IDs to downloader integer strings without float parsing.

    Opaque IDs remain unchanged. Numeric decimal strings must be integral;
    unsafe float integers are rejected because their original digits may be lost.
    """
    if value is None or pd.isna(value):
        raise ValueError('Missing movement identity')
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not value.is_integer() or abs(value) > 2**53-1:
            raise ValueError(f'Unsafe/nonintegral numeric movement identity: {value!r}')
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError('Empty movement identity')
    if re.fullmatch(r'[+-]?\d+(?:\.0+)?', text):
        return str(int(text.split('.')[0]))
    if re.fullmatch(r'[+-]?\d+\.\d+', text):
        raise ValueError(f'Nonintegral movement identity: {value!r}')
    return text


def projected(point, runway):
    azimuth, _, distance = g.GEOD.inv(*runway['threshold'], *point)
    axis = g.GEOD.inv(*runway['threshold'], *runway['opposite'])[0]
    angle = np.deg2rad(azimuth-axis)
    return np.array([distance*np.cos(angle), distance*np.sin(angle)])


def point_features(point, stand, runway):
    result = dict.fromkeys(DISTANCE_COLUMNS + PROGRESS_COLUMNS, np.nan)
    if stand is not None:
        result['adsb_movement_to_gate_direct_m'] = g.geo_distance(point, stand)
    if runway is None:
        return result
    p = projected(point, runway)
    remaining = [g.geo_distance(point, runway['threshold']), p[0], abs(p[1])]
    for kind, value in zip(('direct', 'parallel', 'perpendicular'), remaining):
        result[f'adsb_movement_to_runway_{kind}_m'] = value
    if stand is None:
        return result
    s = projected(stand, runway)
    # Both points share one threshold-centred coordinate system.
    result['adsb_movement_to_gate_parallel_m'] = p[0]-s[0]
    result['adsb_movement_to_gate_perpendicular_m'] = abs(p[1]-s[1])
    baseline = [g.geo_distance(stand, runway['threshold']), s[0], abs(s[1])]
    for kind, rest, total in zip(('direct', 'parallel', 'perpendicular'), remaining, baseline):
        if abs(total) > 1e-6:
            result[f'adsb_movement_progress_{kind}'] = 1-rest/total
    return result


def build_context(clean_paths, destination, n_buckets, *, refresh_geometry=False):
    """Small per-bucket metadata files; no targets or full trajectories in memory."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    columns = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'STAND_mvt', 'RUNWAY_mvt']
    frames = []
    for path in clean_paths:
        print(f'Clean metadata: {Path(path).name}', flush=True)
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError('No cleaned movement Parquets')
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    del frames
    data['MVT_ID_mvt'] = data.MVT_ID_mvt.map(canonical_mvt_id)
    if data.MVT_ID_mvt.isna().any() or data.MVT_ID_mvt.duplicated().any():
        raise ValueError('Missing or conflicting cleaned movement identities')
    data['MVT_TIME_UTC_mvt'] = pd.to_datetime(data.MVT_TIME_UTC_mvt, utc=True)
    for c, func in [('ADEP_mvt', g.geometry_name), ('STAND_mvt', g.geometry_name), ('RUNWAY_mvt', g.geometry_runway)]:
        data[c] = data[c].map(func)
    airports, audits, provenance = {}, [], []
    for icao, rows in data[data.PHASE_mvt.eq('DEP')].dropna(subset=['ADEP_mvt']).groupby('ADEP_mvt'):
        print(f'Geometry: {icao}, matching {rows.STAND_mvt.nunique():,} stands', flush=True)
        caches = sorted(g.data_dir.glob(f'xplane_{icao}_*.json'))
        if refresh_geometry or not caches:
            text = g.load_gateway_geometry(icao)
        elif len(caches) == 1:
            payload = json.loads(caches[0].read_text())
            if payload['icao'] != icao:
                raise ValueError(f'Invalid geometry cache {caches[0]}')
            text = payload['apt_dat']
            provenance.extend(w.snapshot(caches))
        else:
            raise ValueError(f'Multiple scenery caches for {icao}; use --refresh-geometry to resolve recommended scenery')
        airport = g.parse_airport_geometry(text)
        stands, audit = g.select_stand_positions(icao, rows.STAND_mvt, g.load_osm_stands(icao), airport['stand_candidates'])
        audits.append(audit)
        airports[icao] = (stands, airport['runways'])
        print(f'Geometry {icao}: {audit.stand_position_source.value_counts().to_dict()}', flush=True)
    data['_bucket'] = np.fromiter((w.stable_bucket(m, n_buckets) for m in data.MVT_ID_mvt),
                                  dtype=np.int32, count=len(data))
    populated = set()
    for bucket, group in data.groupby('_bucket', sort=True):
        records = {}
        for row in group.itertuples(index=False):
            stands, runways = airports.get(row.ADEP_mvt, ({}, {}))
            stand, runway = stands.get(row.STAND_mvt), runways.get(row.RUNWAY_mvt)
            eligible = row.PHASE_mvt == 'DEP'
            if not eligible:
                stand = runway = None
            status = ('not_departure' if not eligible else 'matched' if stand is not None and runway is not None
                      else 'missing_stand_and_runway' if stand is None and runway is None
                      else 'missing_stand' if stand is None else 'missing_runway')
            records[row.MVT_ID_mvt] = dict(
                airport=row.ADEP_mvt, mvt_time=str(row.MVT_TIME_UTC_mvt), stand=stand,
                runway={k: runway[k] for k in ('threshold', 'opposite')} if runway else None, status=status)
        w.json_write(destination / f'bucket-{bucket:04d}.json', records)
        populated.add(bucket)
        if len(populated) % 32 == 0:
            print(f'Geometry metadata: {len(populated)}/{n_buckets} buckets saved', flush=True)
    for bucket in set(range(n_buckets))-populated:
        w.json_write(destination / f'bucket-{bucket:04d}.json', {})
    if audits:
        pd.concat(audits, ignore_index=True).to_csv(destination/'stand_matching_details.csv', index=False)
    w.json_write(destination/'provenance.json', dict(clean_sources=w.snapshot(clean_paths), scenery=provenance,
        osm=w.snapshot(sorted((g.data_dir/'04_additional_data/osm_stands').glob('osm_stands_*.json')))))


def feature_partition(partition, context_dir):
    partition = Path(partition)
    context = json.loads((Path(context_dir)/(partition.name+'.json')).read_text())
    candidates = pq.read_table(partition/'candidates.parquet').to_pandas()
    selected = candidates.loc[candidates.selected]
    if selected.MVT_ID_mvt.duplicated().any():
        raise ValueError('Multiple selected flights')
    starts = selected.set_index('MVT_ID_mvt').flight_start.to_dict()
    records = {}
    audits = pq.read_table(partition/'audit.parquet').to_pylist()
    if audits and not any(canonical_mvt_id(r['MVT_ID_mvt']) in context for r in audits):
        raise ValueError(f'{partition.name}: no clean metadata matches; rebuild context with canonical MVT IDs')
    for row in audits:
        records[row['MVT_ID_mvt']] = dict(
            **{c: row[c] for c in w.META + ['traffic_flight_id', 'adsb_flight_selection_rule']},
            **dict.fromkeys(EXTRA_SCHEMA.names), adsb_movement_available=False,
            adsb_n_points_1s=0, adsb_movement_start_time=None, adsb_movement_to_mvt_sec=None,
            adsb_movement_status='unresolved_selection:'+str(row['unresolved_reason']),
        )
        records[row['MVT_ID_mvt']]['adsb_geometry_status'] = 'unresolved_selection'
    for frame in w.trajectory_groups(partition/'trajectories.parquet'):
        record = w.movement_record(frame)
        mvt = record['MVT_ID_mvt']
        first = starts[mvt]  # Original selected flight timestamp, not resampled/floored.
        record.update(dict.fromkeys(EXTRA_SCHEMA.names))
        record.update(adsb_first_observation_time=first,
                      adsb_first_observation_to_mvt_sec=(record['MVT_TIME_UTC_mvt']-first).total_seconds())
        geo = context.get(canonical_mvt_id(mvt))
        if geo and (geo['airport'] != record['ADEP_mvt'] or pd.Timestamp(geo['mvt_time']) != record['MVT_TIME_UTC_mvt']):
            raise ValueError(f'Clean/ADS-B metadata conflict for {mvt}')
        record['adsb_geometry_status'] = 'no_sustained_movement'
        if record['adsb_movement_available']:
            points = frame.loc[frame.timestamp.eq(record['adsb_movement_start_time'])]
            if len(points) != 1:
                raise ValueError(f'Movement start not uniquely present in trajectory: {mvt}')
            point = tuple(points.iloc[0][['longitude', 'latitude']].astype(float))
            record.update(adsb_movement_longitude=point[0], adsb_movement_latitude=point[1])
            record['adsb_geometry_status'] = geo['status'] if geo else 'missing_clean_metadata'
            if not np.isfinite(point).all():
                record['adsb_geometry_status'] = 'missing_movement_coordinates'
            elif geo:
                record.update(point_features(point, geo['stand'], geo['runway']))
        records[mvt] = record
    return list(records.values())


def validate_extra(path):
    detected = missing_metadata = 0
    with pq.ParquetFile(path) as source:
        for batch in source.iter_batches(batch_size=4096):
            f = batch.to_pandas()
            detected += int(f.adsb_movement_available.sum())
            missing_metadata += int(f.adsb_geometry_status.eq('missing_clean_metadata').sum())
            delta = (f.MVT_TIME_UTC_mvt-f.adsb_first_observation_time).dt.total_seconds()
            if not np.allclose(delta, f.adsb_first_observation_to_mvt_sec, equal_nan=True):
                raise ValueError('First observation feature arithmetic mismatch')
            if not f.adsb_first_observation_time.notna().eq(f.adsb_n_points_1s.gt(0)).all():
                raise ValueError('First observation coverage mismatch')
            if np.isinf(f[DISTANCE_COLUMNS+PROGRESS_COLUMNS].to_numpy()).any():
                raise ValueError('Infinite geometry features')
            unsigned = [c for c in DISTANCE_COLUMNS if '_parallel_' not in c]
            if f[unsigned].lt(0).any().any():
                raise ValueError('Negative unsigned distance')
            if f.loc[~f.adsb_movement_available, DISTANCE_COLUMNS+PROGRESS_COLUMNS].notna().any().any():
                raise ValueError('Geometry without movement detection')

    if detected and missing_metadata == detected:
        raise ValueError('All detected movements lack clean metadata; check canonical MVT IDs and hash buckets')
