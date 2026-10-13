"""Two-stage ADS-B preparation for notebook 04; importable spawn workers.

No feature detection occurs in Stage A. Only Stage B calls Flight.movement().
Output runs are immutable: an interrupted attempt stays in its private staging
folder; rerunning starts a new attempt and never publishes a partial dataset.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import sqlite3
import time
import tempfile
import uuid
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from traffic.core import Flight, Traffic

GIB = 1024**3
UTC = pa.timestamp("ns", tz="UTC")
RAW_SCHEMA = pa.schema([
    ("MVT_ID_mvt", pa.string()), ("FLIGHT_ID_mvt", pa.string()),
    ("ADEP_mvt", pa.string()), ("MVT_TIME_UTC_mvt", UTC),
    ("icao24", pa.string()), ("callsign", pa.string()), ("timestamp", UTC),
    *[(c, pa.float64()) for c in ["latitude", "longitude", "groundspeed", "track", "altitude", "vertical_rate"]],
    ("on_ground", pa.bool_()), ("match_method", pa.string()), ("source_date", pa.string()),
])
RAW_COLUMNS = ["ground_speed" if c == "groundspeed" else c for c in RAW_SCHEMA.names]
META = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
# FLIGHT_ID_mvt links the movement to the NM flight plan and is missing for ~1 % of the departures (no NM record).
# It may therefore be null, but must be the same (constant null) for all observations of a movement.
NULLABLE_META = {"FLIGHT_ID_mvt"}


def inconsistent_meta(frame):
    """Columns of META that are not constant within a movement, or null although required."""
    return [c for c in META if frame[c].nunique(dropna=False) != 1
            or (c not in NULLABLE_META and frame[c].isna().any())]
OUTPUT_SCHEMA = RAW_SCHEMA.append(pa.field("traffic_flight_id", pa.string())).append(
    pa.field("adsb_flight_selection_rule", pa.string()))
AUDIT_SCHEMA = pa.schema([*[RAW_SCHEMA.field(c) for c in META],
    ("traffic_flight_id", pa.string()), ("adsb_flight_selection_rule", pa.string()),
    ("unresolved_reason", pa.string()), ("n_physical_flights", pa.int64()),
    ("raw_rows", pa.int64()), ("exact_duplicates_removed", pa.int64()),
    ("unassigned_points", pa.int64()), ("adsb_n_points_1s", pa.int64()),
    ("same_second_observations", pa.int64()), ("max_observed_gap_sec", pa.float64()),
])
CANDIDATE_SCHEMA = pa.schema([*[RAW_SCHEMA.field(c) for c in META],
    ("traffic_flight_id", pa.string()), ("flight_start", UTC), ("flight_stop", UTC),
    ("n_adsb_points", pa.int64()), ("selection_rule", pa.string()), ("selected", pa.bool_()),
])
FEATURE_SCHEMA = pa.schema([*[RAW_SCHEMA.field(c) for c in META],
    ("traffic_flight_id", pa.string()), ("adsb_flight_selection_rule", pa.string()),
    ("adsb_movement_start_time", UTC), ("adsb_movement_to_mvt_sec", pa.float64()),
    ("adsb_movement_available", pa.bool_()), ("adsb_n_points_1s", pa.int64()),
    ("adsb_movement_status", pa.string()),
])


@dataclass(frozen=True)
class Config:
    max_workers: int = 16
    n_buckets: int = 512
    batch_rows: int = 262144
    staging_buffer_rows: int = 524288
    ram_budget_gib: float = 64
    reserve_gib: float = 8
    worker_budget_gib: float = 3
    max_bucket_rows: int = 1_500_000
    max_bucket_pandas_gib: float = 0.75
    max_movement_rows: int = 250_000
    max_resampled_points: int = 172_800

    def __post_init__(self):
        if any(v <= 0 for k, v in asdict(self).items() if k != "reserve_gib") or self.reserve_gib < 0:
            raise ValueError("Configuration limits must be positive")


def memory_status():
    try:
        import psutil
        return dict(available_ram_gib=psutil.virtual_memory().available / GIB,
                    process_rss_gib=psutil.Process().memory_info().rss / GIB)
    except ImportError:
        return dict(available_ram_gib=None, process_rss_gib=None)


def resource_report(config):
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    memory = memory_status()
    available, rss = memory["available_ram_gib"], memory["process_rss_gib"]
    budget = min(config.ram_budget_gib, available) if available is not None else config.ram_budget_gib
    workers = min(config.max_workers, len(affinity), int(slurm) if slurm else len(affinity),
                  int((budget - config.reserve_gib) // config.worker_budget_gib))
    if workers < 1:
        raise ValueError("RAM/CPU budget does not permit one worker")
    return dict(cpu_affinity=affinity, cpu_count=os.cpu_count(), SLURM_CPUS_PER_TASK=slurm,
                configured_MAX_WORKERS=config.max_workers, effective_workers=workers,
                available_ram_gib=available, process_rss_gib=rss, N_BUCKETS=config.n_buckets)


def initialize_worker():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    # traffic triggers this pandas deprecation warning once per trajectory; it does not change any result.
    warnings.filterwarnings("ignore", message="Downcasting object dtype arrays", category=FutureWarning)


def stable_bucket(mvt, n_buckets):
    # Explicit UTF-8 + BLAKE2b, independent of PYTHONHASHSEED, file and batch.
    return int.from_bytes(hashlib.blake2b(str(mvt).encode("utf-8"), digest_size=8).digest(), "big") % n_buckets


def bucket_ids(series, n_buckets):
    unique = series.unique()
    return series.map({s: stable_bucket(s, n_buckets) for s in unique}).to_numpy(dtype=np.int32)


def snapshot(paths):
    return [dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size,
                 mtime_ns=Path(p).stat().st_mtime_ns) for p in paths]


def inspect_sources(paths):
    result = []
    for path in paths:
        with pq.ParquetFile(path) as source:
            missing = set(RAW_COLUMNS) - set(source.schema_arrow.names)
            if missing:
                raise ValueError(f"{path}: missing required columns {sorted(missing)}")
            if "flight_id" in source.schema_arrow.names:
                raise ValueError(f"{path}: expected raw observations without flight_id")
            result.append(dict(file=Path(path).name, rows=source.metadata.num_rows,
                               bytes=Path(path).stat().st_size,
                               parquet_uncompressed_bytes=sum(source.metadata.row_group(i).total_byte_size
                                                              for i in range(source.num_row_groups))))
    if not result or sum(r["rows"] for r in result) == 0:
        raise ValueError("No raw ADS-B observations")
    return pd.DataFrame(result)


def normalize(frame):
    frame = frame.rename(columns={"ground_speed": "groundspeed"}).copy()
    for c in ["timestamp", "MVT_TIME_UTC_mvt"]:
        frame[c] = pd.to_datetime(frame[c], utc=True, errors="raise")
    if frame.MVT_ID_mvt.isna().any() or frame.MVT_ID_mvt.eq("").any():
        raise ValueError("Missing MVT_ID_mvt cannot be partitioned safely")
    # No cleaning/filtering in this pass. Null metadata is audited per complete MVT.
    return pa.Table.from_pandas(frame, schema=RAW_SCHEMA, preserve_index=False).to_pandas()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    os.replace(temporary, path)


@contextmanager
def output_lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise FileExistsError(f"Immutable output exists: {path}; choose a new RUN_NAME")
        yield


def bounded_map(function, tasks, workers, progress=None):
    """Only workers futures outstanding; arguments are paths/config, never buckets."""
    started = time.monotonic()
    for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker) as pool:
        iterator, pending, exhausted = iter(tasks), set(), False
        try:
            while pending or not exhausted:
                while len(pending) < workers and not exhausted:
                    task = next(iterator, None)
                    if task is None:
                        exhausted = True
                    else:
                        pending.add(pool.submit(function, *task))
                if not pending:
                    break
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                if progress:
                    progress(time.monotonic() - started)
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def repartition(paths, scratch, config):
    """Stream source batches to Parquet buckets; disk-backed boundary statistics."""
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    database = scratch / "index.sqlite"
    counts = np.zeros(config.n_buckets, dtype=np.int64)
    buffers = [[] for _ in counts]
    buffered = np.zeros(config.n_buckets, dtype=np.int64)
    writers = {}
    db = sqlite3.connect(database)
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE seen(mvt TEXT PRIMARY KEY, bucket INTEGER, n INTEGER, dmin TEXT, dmax TEXT, fmin TEXT, fmax TEXT)")

    def flush(bucket):
        if not buffered[bucket]:
            return
        if bucket not in writers:
            writers[bucket] = pq.ParquetWriter(scratch / f"bucket-{bucket:04d}.parquet", RAW_SCHEMA,
                                                compression="zstd", use_dictionary=False)
        writers[bucket].write_table(pa.concat_tables(buffers[bucket]))
        buffers[bucket].clear()
        buffered[bucket] = 0

    try:
        for path in paths:
            with pq.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=config.batch_rows, columns=RAW_COLUMNS, use_threads=False):
                    frame = normalize(batch.to_pandas())
                    frame["_bucket"] = bucket_ids(frame.MVT_ID_mvt, config.n_buckets)
                    # source_date is metadata: null is counted explicitly, never silently dropped.
                    stat = frame.assign(source_date=frame.source_date.fillna("<missing>"))
                    stat = stat.groupby("MVT_ID_mvt", sort=False).agg(
                        bucket=("_bucket", "first"), n=("_bucket", "size"),
                        dmin=("source_date", "min"), dmax=("source_date", "max"))
                    db.executemany("""INSERT INTO seen VALUES (?,?,?,?,?,?,?) ON CONFLICT(mvt) DO UPDATE SET
                        n=n+excluded.n, dmin=MIN(dmin,excluded.dmin), dmax=MAX(dmax,excluded.dmax),
                        fmin=MIN(fmin,excluded.fmin), fmax=MAX(fmax,excluded.fmax)""",
                        ((str(m), int(r.bucket), int(r.n), r.dmin, r.dmax, Path(path).name, Path(path).name)
                         for m, r in stat.iterrows()))
                    db.commit()
                    for bucket, group in frame.groupby("_bucket", sort=True):
                        bucket = int(bucket)
                        table = pa.Table.from_pandas(group.drop(columns="_bucket"), schema=RAW_SCHEMA, preserve_index=False)
                        buffers[bucket].append(table)
                        buffered[bucket] += len(group)
                        counts[bucket] += len(group)
                    while buffered.sum() >= config.staging_buffer_rows:
                        flush(int(buffered.argmax()))
            print(f"Hash pass: {Path(path).name}; {counts.sum():,} rows staged", flush=True)
        for bucket in range(config.n_buckets):
            flush(bucket)
    finally:
        for writer in writers.values():
            writer.close()
        db.close()
    json_write(scratch / "partition_counts.json", counts.tolist())
    return validate_buckets(scratch, config)


def validate_buckets(scratch, config):
    """Independently scan written buckets; PRIMARY KEY proves cross-bucket uniqueness."""
    scratch = Path(scratch)
    counts = np.zeros(config.n_buckets, dtype=np.int64)
    with sqlite3.connect(scratch / "index.sqlite") as db:
        db.execute("PRAGMA cache_size=-32768")
        db.execute("DROP TABLE IF EXISTS actual")
        db.execute("CREATE TABLE actual(mvt TEXT PRIMARY KEY, bucket INTEGER, n INTEGER)")
        for path in sorted(scratch.glob("bucket-*.parquet")):
            bucket = int(path.stem.split("-")[1])
            with pq.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=config.batch_rows, columns=["MVT_ID_mvt"], use_threads=False):
                    ids = batch.column(0).to_pandas()
                    if not (bucket_ids(ids, config.n_buckets) == bucket).all():
                        raise ValueError("MVT distributed to incorrect hash bucket")
                    for mvt, n in ids.value_counts().items():
                        existing = db.execute("SELECT bucket FROM actual WHERE mvt=?", (mvt,)).fetchone()
                        if existing and existing[0] != bucket:
                            raise ValueError(f"MVT spans multiple buckets: {mvt}")
                        db.execute("INSERT INTO actual VALUES (?,?,?) ON CONFLICT(mvt) DO UPDATE SET n=n+excluded.n",
                                   (mvt, bucket, int(n)))
                    counts[bucket] += len(ids)
            db.commit()
        mismatch = db.execute("""SELECT mvt,bucket,n FROM seen EXCEPT SELECT mvt,bucket,n FROM actual""").fetchone()
        extra = db.execute("SELECT mvt,bucket,n FROM actual EXCEPT SELECT mvt,bucket,n FROM seen").fetchone()
        if mismatch or extra:
            raise ValueError(f"Repartition count/identity mismatch: {mismatch}, {extra}")
        n, days, months = db.execute("SELECT COUNT(*), SUM(dmin<>dmax), SUM(fmin<>fmax) FROM seen").fetchone()
    nonempty = counts[counts > 0]
    if not len(nonempty):
        raise ValueError("No non-empty buckets")
    if counts.max() > config.max_bucket_rows:
        raise MemoryError(f"Bucket RAM guard: {counts.max():,} rows; increase N_BUCKETS and rerun")
    report = dict(total_raw_rows=int(counts.sum()), unique_MVT_ID_mvt=n, N_BUCKETS=config.n_buckets,
                  nonempty_buckets=len(nonempty), bucket_rows_min=int(nonempty.min()),
                  bucket_rows_median=float(np.median(nonempty)), bucket_rows_p95=float(np.quantile(nonempty, .95)),
                  bucket_rows_max=int(nonempty.max()), mvts_multiple_source_dates=int(days or 0),
                  mvts_multiple_monthly_files=int(months or 0), statistics_population="non-empty buckets")
    json_write(scratch / "VALIDATED.json", report)
    return report


def select_candidate(candidates, target):
    """Conservative temporal selection; exact temporal ties remain unresolved."""
    containing = candidates[(candidates.flight_start <= target) & (candidates.flight_stop >= target)]
    if len(containing):
        # Closest start at/before MVT favours departure over a long earlier arrival.
        best = containing[containing.flight_start == containing.flight_start.max()]
        rule = "contains_mvt"
    else:
        after = candidates[candidates.flight_start > target]
        if len(after):
            best = after[after.flight_start == after.flight_start.min()]
            rule = "starts_after_mvt"
        else:
            before = candidates[candidates.flight_stop < target]
            best = before[before.flight_stop == before.flight_stop.max()]
            rule = "ends_before_mvt"
    if len(best) != 1:
        return None, "unresolved", f"temporal_tie_{rule}"
    return best.traffic_flight_id.iloc[0], rule, None


def prepare_movement(frame, config):
    """One complete MVT: assign -> candidate audit -> select -> resample."""
    if len(frame) > config.max_movement_rows:
        raise MemoryError(f"Movement RAM guard: {frame.MVT_ID_mvt.iloc[0]} ({len(frame):,} rows)")
    raw_rows = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    meta = {c: frame[c].iloc[0] for c in META}
    audit = dict(**meta, traffic_flight_id=None, adsb_flight_selection_rule="unresolved",
                 unresolved_reason=None, n_physical_flights=0, raw_rows=raw_rows,
                 exact_duplicates_removed=raw_rows-len(frame), unassigned_points=0,
                 adsb_n_points_1s=0, same_second_observations=0, max_observed_gap_sec=None)
    candidates = pd.DataFrame(columns=CANDIDATE_SCHEMA.names)
    if inconsistent_meta(frame):
        # Do not choose an arbitrary conflicting challenge metadata value.
        for c in META:
            if frame[c].nunique(dropna=False) != 1:
                audit[c] = None
        audit["unresolved_reason"] = "inconsistent_or_missing_movement_metadata"
        return None, audit, candidates
    if frame[["timestamp", "icao24", "callsign"]].isna().any().any():
        audit["unresolved_reason"] = "missing_timestamp_or_aircraft_identity"
        return None, audit, candidates
    frame = frame.sort_values(["timestamp", "icao24", "callsign", "source_date"], kind="stable", na_position="last")
    frame = frame.reset_index(drop=True)
    # Temporary row numbers are an assignment-completeness check, NOT flight IDs.
    assigned = Traffic(frame.assign(_observation_row=np.arange(len(frame)))).assign_id().eval(max_workers=1)
    if assigned is None:
        audit.update(unassigned_points=len(frame), unresolved_reason="traffic_assigned_no_flights")
        return None, audit, candidates
    data = assigned.data.rename(columns={"flight_id": "traffic_flight_id"})
    if data._observation_row.duplicated().any():
        raise ValueError("traffic assigned an observation more than once")
    audit["unassigned_points"] = len(frame) - data._observation_row.nunique()
    candidates = data.groupby("traffic_flight_id", sort=True).agg(
        flight_start=("timestamp", "min"), flight_stop=("timestamp", "max"),
        n_adsb_points=("timestamp", "size")).reset_index()
    for c, value in meta.items():
        candidates[c] = value
    audit["n_physical_flights"] = len(candidates)
    selected, rule, reason = select_candidate(candidates, meta["MVT_TIME_UTC_mvt"])
    if audit["unassigned_points"]:
        selected, rule, reason = None, "unresolved", "traffic_omitted_observations_including_possible_singletons"
    candidates["selection_rule"], candidates["selected"] = rule, False
    if selected is None:
        audit["unresolved_reason"] = reason
        return None, audit, candidates
    chosen = data[data.traffic_flight_id == selected].drop(columns="_observation_row").sort_values("timestamp", kind="stable")
    # Multiple non-identical observations at one timestamp cannot be safely ordered.
    if chosen.timestamp.duplicated().any():
        candidates["selection_rule"] = "unresolved"
        audit["unresolved_reason"] = "conflicting_same_timestamp_observations"
        return None, audit, candidates
    points = int((chosen.timestamp.max().floor("s") - chosen.timestamp.min().floor("s")).total_seconds()) + 1
    if points > config.max_resampled_points:
        raise MemoryError(f"Resampling RAM guard: {meta['MVT_ID_mvt']} would have {points:,} points")
    candidates["selected"] = candidates.traffic_flight_id.eq(selected)
    # Remove challenge metadata before interpolation, restore constants afterwards.
    observations = chosen.drop(columns=META + ["traffic_flight_id"])
    result = Flight(observations).resample("1s").data
    for c, value in meta.items():
        result[c] = value
    result["traffic_flight_id"] = selected
    result["adsb_flight_selection_rule"] = rule
    audit.update(traffic_flight_id=selected, adsb_flight_selection_rule=rule,
                 adsb_n_points_1s=len(result),
                 same_second_observations=int(chosen.timestamp.dt.floor("s").duplicated().sum()),
                 max_observed_gap_sec=float(chosen.timestamp.diff().dt.total_seconds().max()))
    return result[OUTPUT_SCHEMA.names], audit, candidates


class RecordWriter:
    """Compact audit buffers with fixed schema even if all values are null."""
    def __init__(self, path, schema, limit=2048):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema, self.limit, self.records = schema, limit, []

    def add(self, records):
        for record in records:
            self.records.append(record)
            if len(self.records) >= self.limit:
                self.flush()

    def flush(self):
        if self.records:
            self.writer.write_table(pa.Table.from_pandas(pd.DataFrame(self.records, columns=self.schema.names), schema=self.schema, preserve_index=False))
            self.records.clear()

    def close(self):
        self.flush()
        self.writer.close()


def process_bucket(path, destination, config):
    path, destination = Path(path), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with pq.ParquetFile(path) as source:
        if source.metadata.num_rows > config.max_bucket_rows:
            raise MemoryError("Bucket row RAM guard")
        frame = source.read(use_threads=False).to_pandas()
    if frame.memory_usage(deep=True).sum() > config.max_bucket_pandas_gib * GIB:
        raise MemoryError("Bucket pandas RAM guard: increase N_BUCKETS")
    audits = RecordWriter(destination / "audit.parquet", AUDIT_SCHEMA)
    candidates = RecordWriter(destination / "candidates.parquet", CANDIDATE_SCHEMA)
    totals = dict(mvts=0, assigned_mvts=0, single=0, multiple=0, physical_flights=0,
                  max_physical_flights=0, selected=0, resampled=0, rows=0, unresolved=0,
                  contains_mvt=0, starts_after_mvt=0, ends_before_mvt=0)
    try:
        with pq.ParquetWriter(destination / "trajectories.parquet", OUTPUT_SCHEMA, compression="zstd") as writer:
            for _, group in frame.groupby("MVT_ID_mvt", sort=True):
                result, audit, table = prepare_movement(group, config)
                audits.add([audit])
                candidates.add(table.to_dict("records"))
                n = audit["n_physical_flights"]
                totals["mvts"] += 1
                totals["assigned_mvts"] += n > 0
                totals["single"] += n == 1
                totals["multiple"] += n > 1
                totals["physical_flights"] += n
                totals["max_physical_flights"] = max(totals["max_physical_flights"], n)
                totals[audit["adsb_flight_selection_rule"]] += 1
                if result is not None:
                    # Exactly one full trajectory per row group. Stage B reads one at a time.
                    writer.write_table(pa.Table.from_pandas(result, schema=OUTPUT_SCHEMA, preserve_index=False),
                                       row_group_size=len(result))
                    totals["selected"] += 1
                    totals["resampled"] += 1
                    totals["rows"] += len(result)
    finally:
        audits.close()
        candidates.close()
    return dict(path=str(destination), totals=totals)


def trajectory_groups(path):
    """Stage A guarantees one complete MVT per Parquet row group."""
    with pq.ParquetFile(path) as source:
        for index in range(source.num_row_groups):
            yield source.read_row_group(index, use_threads=False).to_pandas()


def validate_stage_a(output, config, expected_mvts, expected_rows, raw_index=None):
    """Read persisted output; verify coverage and one complete 1s trajectory per MVT."""
    output = Path(output)
    rows = 0
    with tempfile.TemporaryDirectory(prefix="adsb-validate-") as temporary, \
            sqlite3.connect(Path(temporary) / "validation.sqlite") as db:
        db.execute("CREATE TABLE audit(mvt TEXT PRIMARY KEY, bucket INTEGER, flight TEXT, n INTEGER, rule TEXT)")
        db.execute("CREATE TABLE observed(mvt TEXT PRIMARY KEY, bucket INTEGER, flight TEXT, n INTEGER, rule TEXT)")
        for partition in sorted((output / "buckets").glob("bucket-*")):
            bucket = int(partition.name.split("-")[1])
            with pq.ParquetFile(partition / "audit.parquet") as audit_file:
                if not audit_file.schema_arrow.equals(AUDIT_SCHEMA):
                    raise ValueError("Audit schema mismatch")
                for batch in audit_file.iter_batches(batch_size=4096):
                    for r in batch.to_pylist():
                        if stable_bucket(r["MVT_ID_mvt"], config.n_buckets) != bucket:
                            raise ValueError("Audit partition mismatch")
                        db.execute("INSERT INTO audit VALUES (?,?,?,?,?)", (r["MVT_ID_mvt"], bucket,
                            r["traffic_flight_id"], r["adsb_n_points_1s"], r["adsb_flight_selection_rule"]))
            with pq.ParquetFile(partition / "trajectories.parquet") as source:
                if not source.schema_arrow.equals(OUTPUT_SCHEMA):
                    raise ValueError("Processed schema mismatch")
            for frame in trajectory_groups(partition / "trajectories.parquet"):
                if inconsistent_meta(frame) or any(frame[c].isna().any() or frame[c].nunique() != 1
                                                   for c in ["traffic_flight_id", "adsb_flight_selection_rule"]):
                    raise ValueError("Trajectory has non-constant/null metadata")
                stamps = frame.timestamp
                if stamps.isna().any() or not stamps.is_monotonic_increasing or stamps.duplicated().any():
                    raise ValueError("Unordered or duplicated trajectory timestamps")
                if not stamps.diff().dropna().eq(pd.Timedelta(seconds=1)).all():
                    raise ValueError("Trajectory does not have exactly 1-second spacing")
                mvt = frame.MVT_ID_mvt.iloc[0]
                if stable_bucket(mvt, config.n_buckets) != bucket:
                    raise ValueError("Output hash partition mismatch")
                db.execute("INSERT INTO observed VALUES (?,?,?,?,?)", (mvt, bucket,
                    frame.traffic_flight_id.iloc[0], len(frame), frame.adsb_flight_selection_rule.iloc[0]))
                rows += len(frame)
            db.commit()
        missing = db.execute("SELECT * FROM audit WHERE n>0 EXCEPT SELECT * FROM observed").fetchone()
        extra = db.execute("SELECT * FROM observed EXCEPT SELECT * FROM audit WHERE n>0").fetchone()
        n = db.execute("SELECT COUNT(*) FROM observed").fetchone()[0]
        n_audit = db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
        invalid = db.execute("SELECT mvt FROM audit WHERE (n=0 AND (flight IS NOT NULL OR rule<>'unresolved')) OR (n>0 AND flight IS NULL)").fetchone()
        if missing or extra or invalid or n_audit != expected_mvts or rows != expected_rows:
            raise ValueError(f"Stage-A coverage/count mismatch: {missing}, {extra}, {invalid}")
        if raw_index is not None:
            db.execute("ATTACH DATABASE ? AS raw_index", (str(raw_index),))
            missing_raw = db.execute("SELECT mvt FROM raw_index.seen EXCEPT SELECT mvt FROM audit").fetchone()
            extra_raw = db.execute("SELECT mvt FROM audit EXCEPT SELECT mvt FROM raw_index.seen").fetchone()
            if missing_raw or extra_raw:
                raise ValueError("Stage-A audit differs from raw MVT population")
    return dict(rows=rows, persisted_mvts=n, raw_mvts=n_audit, schema_valid=True,
                exactly_one_flight_per_mvt=True, spacing_seconds=1, unique_partition_per_mvt=True)


def stage_a(paths, output, scratch_root, config, partitioned=None):
    """Publish Stage A only after a full persisted-data validation.

    Optional partitioned path is produced by the notebook's explicit pass 1.
    Its source/config fingerprint must match before reuse.
    """
    paths = [Path(p).resolve() for p in paths]
    output, scratch_root = Path(output).resolve(), Path(scratch_root).resolve()
    sources = snapshot(paths)
    inspect_sources(paths)
    resources = resource_report(config)
    print(json.dumps(resources, indent=2))
    with output_lock(output):
        attempt = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
        (attempt / "buckets").mkdir(parents=True)
        (attempt / "pending").mkdir()
        try:
            if partitioned is None:
                partitioned = scratch_root / f"hash-{uuid.uuid4().hex}"
                raw_report = repartition(paths, partitioned, config)
                json_write(Path(partitioned) / "INPUTS.json", dict(sources=sources, config=asdict(config)))
            else:
                partitioned = Path(partitioned)
                provenance = json.loads((partitioned / "INPUTS.json").read_text())
                if provenance != dict(sources=sources, config=asdict(config)):
                    raise ValueError("Hash-bucket source/config fingerprint mismatch")
                raw_report = validate_buckets(partitioned, config)
            totals, completed = {}, 0
            tasks = [(str(p), str(attempt / "pending" / p.stem), config)
                     for p in sorted(Path(partitioned).glob("bucket-*.parquet"))]

            def progress(elapsed):
                print(dict(buckets=f"{completed}/{len(tasks)}", **totals,
                           elapsed_sec=round(elapsed, 1), mvts_per_sec=round(totals.get("mvts", 0)/max(elapsed, .001), 2),
                           rss_gib=memory_status()["process_rss_gib"]), flush=True)

            for result in bounded_map(process_bucket, tasks, resources["effective_workers"], progress):
                temporary = Path(result["path"])
                # Only the coordinator commits; workers own disjoint private files.
                os.replace(temporary, attempt / "buckets" / temporary.name)
                completed += 1
                for k, v in result["totals"].items():
                    totals[k] = max(totals.get(k, 0), v) if k == "max_physical_flights" else totals.get(k, 0) + v
            validation = validate_stage_a(attempt, config, raw_report["unique_MVT_ID_mvt"], totals["rows"], Path(partitioned) / "index.sqlite")
            if sources != snapshot(paths):
                raise RuntimeError("Raw inputs changed during Stage A; unpublished attempt retained")
            manifest = dict(stage="A", config=asdict(config), resources=resources, sources=sources,
                            traffic_version=importlib.metadata.version("traffic"), raw=raw_report,
                            totals=totals, validation=validation,
                            assignment="Traffic.assign_id().eval(max_workers=1); default 10-minute split",
                            selection="contains: latest start; then earliest start after; then latest stop before; temporal ties unresolved")
            manifest["parquet_bytes"] = sum(p.stat().st_size for p in attempt.rglob("*.parquet"))
            json_write(attempt / "SUCCESS.json", manifest)
            os.replace(attempt, output)
            print(f"Stage A committed: {Path(output).name}; Parquet size {manifest['parquet_bytes']/GIB:.3f} GiB")
            return manifest
        except BaseException:
            print(f"Stage A not published. Incomplete attempt retained at {attempt}", flush=True)
            raise


def movement_record(frame):
    """Feature values use only the persisted 1-second trajectory."""
    row = frame.iloc[0]
    record = {c: row[c] for c in META + ["traffic_flight_id", "adsb_flight_selection_rule"]}
    # The installed default movement implementation resamples internally.
    movement = Flight(frame).movement(speed_threshold=2, time_threshold="30s", resample_rule="1s")
    start = movement.start if movement is not None else pd.NaT
    available = bool(pd.notna(start))
    record.update(adsb_movement_start_time=start,
                  adsb_movement_to_mvt_sec=(row.MVT_TIME_UTC_mvt-start).total_seconds() if available else None,
                  adsb_movement_available=available, adsb_n_points_1s=len(frame),
                  adsb_movement_status="detected" if available else "no_sustained_movement")
    return record


def feature_partition(partition):
    """A bounded bucket of compact records returned; trajectories stay in worker."""
    partition = Path(partition)
    # Audit is compact (one row per raw MVT); processed observations are streamed.
    audit = pq.read_table(partition / "audit.parquet").to_pylist()
    records = {}
    for row in audit:
        records[row["MVT_ID_mvt"]] = dict(
            **{c: row[c] for c in META + ["traffic_flight_id", "adsb_flight_selection_rule"]},
            adsb_movement_start_time=None, adsb_movement_to_mvt_sec=None,
            adsb_movement_available=False, adsb_n_points_1s=0,
            adsb_movement_status="unresolved_selection:" + str(row["unresolved_reason"]))
    for frame in trajectory_groups(partition / "trajectories.parquet"):
        record = movement_record(frame)
        records[record["MVT_ID_mvt"]] = record
    return list(records.values())


def stage_b(processed, feature_path, config, *, partition_function=feature_partition,
            partition_args=(), feature_schema=FEATURE_SCHEMA, extra_validator=None):
    """The only source is committed Stage A; write one standalone ZSTD Parquet."""
    processed, feature_path = Path(processed), Path(feature_path)
    marker = processed / "SUCCESS.json"
    if not marker.exists():
        raise ValueError("Stage B requires successfully persisted and validated Stage A")
    manifest = json.loads(marker.read_text())
    persisted_config = Config(**manifest["config"])
    # Revalidate persisted inputs even after a kernel restart; never trust a marker alone.
    validate_stage_a(processed, persisted_config, manifest["raw"]["unique_MVT_ID_mvt"], manifest["totals"]["rows"])
    resources = resource_report(config)
    with output_lock(feature_path):
        temporary = feature_path.with_name(f".{feature_path.name}.{uuid.uuid4().hex}.tmp")
        total = valid = completed = 0
        tasks = [(str(p), *partition_args) for p in sorted((processed / "buckets").glob("bucket-*"))]
        with pq.ParquetWriter(temporary, feature_schema, compression="zstd") as writer:
            for records in bounded_map(partition_function, tasks, resources["effective_workers"],
                                       lambda elapsed: print(dict(stage="B", buckets=f"{completed}/{len(tasks)}",
                                           mvts=total, detections=valid, elapsed_sec=round(elapsed, 1),
                                           mvts_per_sec=round(total/max(elapsed, .001), 2),
                                           rss_gib=memory_status()["process_rss_gib"]), flush=True)):
                writer.write_table(pa.Table.from_pandas(pd.DataFrame(records, columns=feature_schema.names), schema=feature_schema, preserve_index=False))
                total += len(records)
                valid += sum(r["adsb_movement_available"] for r in records)
                completed += 1
        validate_features(temporary, processed, manifest, feature_schema)
        if extra_validator is not None:
            extra_validator(temporary)
        os.replace(temporary, feature_path)
        result = dict(stage="B", processed=Path(processed).name, rows=total, valid_movement_detections=valid,
                      failed_movement_detections=manifest["validation"]["persisted_mvts"]-valid,
                      unresolved=manifest["totals"]["unresolved"], coverage_percent=100*valid/total,
                      speed_threshold=2, time_threshold="30s", resample_rule="1s",
                      traffic_version=importlib.metadata.version("traffic"), bytes=feature_path.stat().st_size)
        json_write(feature_path.with_suffix(".json"), result)
        return result


def validate_features(path, processed, manifest, feature_schema=FEATURE_SCHEMA):
    database = Path(str(path) + ".validation.sqlite")
    try:
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE expected(mvt TEXT PRIMARY KEY, flight TEXT, n INTEGER)")
            db.execute("CREATE TABLE actual(mvt TEXT PRIMARY KEY, flight TEXT, n INTEGER)")
            for partition in sorted((Path(processed) / "buckets").glob("bucket-*")):
                with pq.ParquetFile(partition / "audit.parquet") as source:
                    for batch in source.iter_batches(batch_size=4096):
                        db.executemany("INSERT INTO expected VALUES (?,?,?)", ((r["MVT_ID_mvt"], r["traffic_flight_id"],
                            r["adsb_n_points_1s"]) for r in batch.to_pylist()))
            with pq.ParquetFile(path) as source:
                if not source.schema_arrow.equals(feature_schema):
                    raise ValueError("Feature schema mismatch")
                for batch in source.iter_batches(batch_size=4096):
                    frame = batch.to_pandas()
                    if not frame.adsb_movement_available.eq(frame.adsb_movement_start_time.notna()).all():
                        raise ValueError("Invalid availability flag")
                    delta = (frame.MVT_TIME_UTC_mvt-frame.adsb_movement_start_time).dt.total_seconds()
                    if not np.allclose(delta, frame.adsb_movement_to_mvt_sec, equal_nan=True):
                        raise ValueError("Movement feature formula mismatch")
                    db.executemany("INSERT INTO actual VALUES (?,?,?)", ((r.MVT_ID_mvt, r.traffic_flight_id, r.adsb_n_points_1s)
                                                                        for r in frame.itertuples()))
            mismatch = db.execute("SELECT * FROM expected EXCEPT SELECT * FROM actual").fetchone()
            extra = db.execute("SELECT * FROM actual EXCEPT SELECT * FROM expected").fetchone()
            if mismatch or extra or db.execute("SELECT COUNT(*) FROM actual").fetchone()[0] != manifest["raw"]["unique_MVT_ID_mvt"]:
                raise ValueError("Feature identity/count mismatch")
    finally:
        if database.exists():
            database.unlink()


def diagnostics(processed, feature_path):
    """Compact statistics only; SQL keeps quantile sorting/feature population on disk."""
    processed, feature_path = Path(processed), Path(feature_path)
    manifest = json.loads((processed / "SUCCESS.json").read_text())
    database = feature_path.with_suffix(".diagnostics.sqlite")
    if database.exists():
        database.unlink()
    examples = {k: [] for k in ["single", "arrival_then_departure", "contains_mvt", "starts_after_mvt",
                                "ends_before_mvt", "unresolved", "multi_flight", "normal", "failed", "negative", "large_positive"]}
    try:
        with sqlite3.connect(database) as db:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-32768")
            db.execute("CREATE TABLE features(mvt TEXT, airport TEXT, month TEXT, value REAL, available INTEGER, points INTEGER)")
            count, mean, m2, valid, persisted = 0, 0., 0., 0, 0
            with pq.ParquetFile(feature_path) as source:
                for batch in source.iter_batches(batch_size=4096):
                    frame = batch.to_pandas()
                    db.executemany("INSERT INTO features VALUES (?,?,?,?,?,?)", ((r.MVT_ID_mvt, r.ADEP_mvt,
                        r.MVT_TIME_UTC_mvt.strftime("%Y-%m") if pd.notna(r.MVT_TIME_UTC_mvt) else None,
                        float(r.adsb_movement_to_mvt_sec) if pd.notna(r.adsb_movement_to_mvt_sec) else None,
                        int(r.adsb_movement_available), int(r.adsb_n_points_1s)) for r in frame.itertuples()))
                    values = frame.adsb_movement_to_mvt_sec.dropna().to_numpy()
                    if len(values):
                        n, avg = len(values), float(values.mean())
                        delta = avg-mean
                        m2 += float(((values-avg)**2).sum()) + delta**2 * count*n/(count+n)
                        mean += delta*n/(count+n)
                        count += n
                    valid += int(frame.adsb_movement_available.sum())
                    persisted += int(frame.adsb_n_points_1s.gt(0).sum())
                    masks = dict(normal=frame.adsb_movement_to_mvt_sec.between(0, 1800),
                                 failed=~frame.adsb_movement_available & frame.adsb_n_points_1s.gt(0),
                                 negative=frame.adsb_movement_to_mvt_sec.lt(0), large_positive=frame.adsb_movement_to_mvt_sec.gt(7200))
                    for key, mask in masks.items():
                        examples[key].extend(frame[mask].head(max(0, 5-len(examples[key]))).to_dict("records"))
            db.execute("CREATE INDEX values_idx ON features(value)")
            coverage = {}
            for group in ["airport", "month"]:
                coverage[group] = pd.read_sql_query(f"""SELECT COALESCE({group},'<missing>') AS {group}, COUNT(*) AS raw_mvts,
                    SUM(points>0) AS persisted_mvts, SUM(available) AS valid_detections,
                    100.0*SUM(available)/COUNT(*) AS raw_coverage_pct,
                    100.0*SUM(available)/NULLIF(SUM(points>0),0) AS persisted_coverage_pct
                    FROM features GROUP BY {group} ORDER BY {group}""", db)
            distribution = dict(count=count, mean=mean if count else None, std=(m2/(count-1))**.5 if count>1 else None)
            for name, q in [("min",0), ("p01",.01), ("p05",.05), ("p25",.25), ("median",.5), ("p75",.75), ("p95",.95), ("p99",.99), ("max",1)]:
                value = None
                if count:
                    position = (count-1)*q
                    low, high = int(np.floor(position)), int(np.ceil(position))
                    vals = [r[0] for r in db.execute("SELECT value FROM features WHERE value IS NOT NULL ORDER BY value LIMIT ? OFFSET ?", (high-low+1, low))]
                    value = vals[0] + (vals[-1]-vals[0])*(position-low)
                distribution[name] = value
            tails = []
            for label, clause in [("< 0 sec", "value<0"), ("> 30 min", "value>1800"), ("> 60 min", "value>3600"), ("> 120 min", "value>7200")]:
                n = db.execute(f"SELECT COUNT(*) FROM features WHERE {clause}").fetchone()[0]
                tails.append(dict(category=label, count=n, pct_valid=100*n/count if count else None))
        for partition in sorted((processed / "buckets").glob("bucket-*")):
            # Compact candidate/audit tables, at most one bounded bucket in RAM.
            audit = pq.read_table(partition / "audit.parquet").to_pandas()
            candidates = pq.read_table(partition / "candidates.parquet").to_pandas()
            candidate_groups = {mvt: group for mvt, group in candidates.groupby("MVT_ID_mvt", sort=False)}
            for row in audit.itertuples():
                group = candidate_groups.get(row.MVT_ID_mvt, candidates.iloc[:0])
                keys = []
                if row.n_physical_flights == 1:
                    keys.append("single")
                if row.n_physical_flights > 1:
                    keys.append("multi_flight")
                keys.append(row.adsb_flight_selection_rule)
                if row.n_physical_flights > 1 and row.adsb_flight_selection_rule in ["contains_mvt", "starts_after_mvt"] and (group.flight_stop < row.MVT_TIME_UTC_mvt).any():
                    keys.append("arrival_then_departure")
                for key in keys:
                    if key in examples and len(examples[key]) < 5:
                        examples[key].append(dict(audit=row._asdict(), candidates=group.to_dict("records")))
        totals = dict(total_raw_mvts=manifest["raw"]["unique_MVT_ID_mvt"], assigned_mvts=manifest["totals"]["assigned_mvts"],
                      multiple_flights=manifest["totals"]["multiple"], unresolved=manifest["totals"]["unresolved"],
                      persisted_mvts=persisted, valid_movement_detections=valid, failed_movement_detections=persisted-valid,
                      raw_coverage_pct=100*valid/manifest["raw"]["unique_MVT_ID_mvt"],
                      persisted_coverage_pct=100*valid/persisted if persisted else None)
        return dict(totals=totals, coverage=coverage, distribution=pd.Series(distribution), tails=pd.DataFrame(tails), examples=examples)
    finally:
        if database.exists():
            database.unlink()
