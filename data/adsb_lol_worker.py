"""Streaming day workers for 03_fetch_adsb_lol.ipynb.

Importable functions are required for ProcessPoolExecutor's spawn context in Jupyter.
Workers write only their private filtered daily staging files. The notebook owns the
monthly Parquet files, progress JSON, and run lock. No challenge data load on import.
"""
import collections
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import sqlite3
import tarfile
import time
from urllib.parse import urlparse
from concurrent.futures import CancelledError

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

BATCH_ROWS = 20_000
MAX_TRACE_BYTES = 512 * 1024 * 1024
HTTP_TIMEOUT = (30, 120)
PROGRESS_INTERVAL_SECONDS = 30
SESSION = None
_EVENTS = None
_STOP = None
_OWNER_PID = None

def initialize_worker(config, events, stop, owner_pid):
    global BATCH_ROWS, MAX_TRACE_BYTES, HTTP_TIMEOUT, PROGRESS_INTERVAL_SECONDS
    global SESSION, _EVENTS, _STOP, _OWNER_PID
    BATCH_ROWS = config["batch_rows"]
    MAX_TRACE_BYTES = config["max_trace_bytes"]
    HTTP_TIMEOUT = config["http_timeout"]
    PROGRESS_INTERVAL_SECONDS = config["progress_interval_seconds"]
    _EVENTS, _STOP, _OWNER_PID = events, stop, owner_pid
    # Progress messages are best-effort: never block worker shutdown on a full pipe.
    if events is not None:
        events.cancel_join_thread()
    SESSION = requests.Session()
    SESSION.headers.update({"User-Agent": "PRC-2026-filtered-ADSB-acquisition"})
    # Do not create an additional Arrow thread pool per CPU in every worker.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

def check_cancelled():
    if (_STOP is not None and _STOP.is_set()) or (_OWNER_PID is not None and os.getppid() != _OWNER_PID):
        raise CancelledError("Download run interrupted or parent process exited")

def log(*parts, flush=True):
    message = " ".join(str(part) for part in parts)
    if _EVENTS is None:
        print(message, flush=flush)
    else:
        try:
            _EVENTS.put_nowait(message)
        except queue.Full:
            pass  # Status must never block download workers while the parent saves a month.

def require(condition, message):
    if not condition:
        raise ValueError(message)

def norm(value):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().upper()
    return "" if value in {"", "NAN", "NONE", "UNKNOWN", "N/A"} else value

class DownloadProgress:
    """Rate-limited status; byte percentage describes transfer, not total processing."""
    def __init__(self, day, total_bytes):
        self.day = day
        self.total_bytes = total_bytes
        self.bytes = self.traces = self.candidates = self.rows = 0
        self.started = self.last_print = time.monotonic()
    def report(self, force=False):
        current = time.monotonic()
        if not force and current - self.last_print < PROGRESS_INTERVAL_SECONDS:
            return
        elapsed = max(current - self.started, 0.001)
        percent = 100 * self.bytes / self.total_bytes if self.total_bytes else 0
        log(f"{self.day} [3/6 Stream + Filter] "
              f"{self.bytes / 1e9:.2f}/{self.total_bytes / 1e9:.2f} GB ({percent:.1f}% transferred) | "
              f"{self.traces:,} traces checked | {self.candidates:,} candidates | "
              f"{self.rows:,} points queued | {elapsed / 60:.1f} min | "
              f"{self.bytes / elapsed / 1e6:.1f} MB/s incl. processing", flush=True)
        self.last_print = current

YEAR_INDEX = {}
def metadata_get(url, optional=False):
    check_cancelled()
    headers = {}
    if urlparse(url).netloc == "api.github.com" and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    response = SESSION.get(url, timeout=HTTP_TIMEOUT, headers=headers)
    with response:
        if optional and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json() if "api.github.com" in url else response.text

def release_tags(day):
    pattern = re.escape("v" + day.replace("-", ".")) + r"-planes-readsb-(?:prod|staging)-[0-9]+(?:tmp)?"
    # Publication can cross New Year: Dec 31 may live in the following year's repository.
    for year in [int(day[:4]), int(day[:4]) + 1]:
        repo = f"adsblol/globe_history_{year}"
        if year not in YEAR_INDEX:
            base = f"https://raw.githubusercontent.com/{repo}/main/"
            preferred = metadata_get(base + "PREFERRED_RELEASES.txt", optional=True) or ""
            alternatives = metadata_get(base + "RELEASES.md", optional=True) or ""
            YEAR_INDEX[year] = (preferred, alternatives)
        preferred, alternatives = YEAR_INDEX[year]
        tags = list(dict.fromkeys(re.findall(pattern, preferred) + re.findall(pattern, alternatives)))
        if tags:
            return repo, tags
    raise ValueError(f"No prod/staging releases published for {day} in its year or following year")

def release_assets(repo, tag):
    release = metadata_get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    assets = [a for a in release["assets"] if re.fullmatch(re.escape(tag) + r"\.tar(?:\.[a-z]{2})?", a["name"])]
    require(assets, f"No tar assets for {tag}")
    assets.sort(key=lambda a: a["name"])
    if len(assets) > 1 or not assets[0]["name"].endswith(".tar"):
        suffixes = [a["name"].rsplit(".", 1)[-1] for a in assets]
        expected = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(len(assets))]
        require(suffixes == expected, f"Missing/out-of-order split parts: {suffixes}")
    return assets

class HTTPParts(io.RawIOBase):
    """Forward-only concatenation; validates every part, including bytes after tar EOF."""
    def __init__(self, assets, progress=None):
        super().__init__()
        self.assets = iter(assets)
        self.progress = progress
        self.part = 0
        self.response = None
        self.finished = False
        self.total = 0
    def readable(self):
        return True
    def readinto(self, buffer):
        while not self.finished:
            check_cancelled()
            if self.response is None:
                asset = next(self.assets, None)
                if asset is None:
                    self.finished = True
                    return 0
                self.asset = asset
                self.part += 1
                if self.progress is not None:
                    log(f"{self.progress.day} [3/6 Connection] archive part {self.part}: "
                          f"{asset['name']} ({asset['size'] / 1e9:.2f} GB)", flush=True)
                self.response = SESSION.get(asset["browser_download_url"], stream=True, timeout=HTTP_TIMEOUT,
                                            headers={"Accept-Encoding": "identity"})
                self.response.raise_for_status()
                self.count = 0
                self.digest = hashlib.sha256()
            data = self.response.raw.read(len(buffer))
            if data:
                buffer[:len(data)] = data
                self.count += len(data)
                self.total += len(data)
                self.digest.update(data)
                if self.progress is not None:
                    self.progress.bytes = self.total
                    self.progress.report()
                return len(data)
            self.response.close()
            self.response = None
            require(self.count == self.asset["size"], f"Truncated asset {self.asset['name']}")
            expected = self.asset.get("digest")
            if expected and expected.startswith("sha256:"):
                require(self.digest.hexdigest() == expected.split(":", 1)[1], "Asset checksum mismatch")
        return 0
    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()

def trace_documents(stream, callsigns, progress=None):
    trace_count = 0
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            check_cancelled()
            # TarFile otherwise retains a TarInfo object for every member in Python 3.11.
            archive.members.clear()
            if not member.isfile() or not re.search(r"(?:^|/)trace_full_[~0-9a-fA-F]+\.json(?:\.gz)?$", member.name):
                continue
            trace_count += 1
            if progress is not None:
                progress.traces = trace_count
                progress.report()
            with archive.extractfile(member) as raw:
                with io.BufferedReader(raw) as buffered:
                    compressed = buffered.peek(2)[:2] == b"\x1f\x8b"
                    if compressed:
                        with gzip.GzipFile(fileobj=buffered) as decoded:
                            data = decoded.read(MAX_TRACE_BYTES + 1)
                    else:
                        data = buffered.read(MAX_TRACE_BYTES + 1)
            require(len(data) <= MAX_TRACE_BYTES, f"Oversized trace: {member.name}; decompressed size exceeds {MAX_TRACE_BYTES / 1024**2:.0f} MiB; increase MAX_TRACE_BYTES explicitly")
            # Readsb callsigns are ASCII. This screening is necessary because member names are hex only.
            found = {norm(v.decode("ascii")) for v in re.findall(rb'"flight"\s*:\s*"([^"\\]*)"', data)}
            if not (found & callsigns):
                continue
            document = json.loads(data)
            require(isinstance(document.get("trace"), list), f"Invalid trace: {member.name}")
            yield document
    # Do not silently accept missing later split parts or an HTTP failure beyond tar EOF.
    while stream.read(1024 * 1024):
        pass
    require(trace_count > 0, "Archive contains no full aircraft traces")

SCHEMA = pa.schema([
    ("MVT_ID_mvt", pa.string()), ("FLIGHT_ID_mvt", pa.string()), ("ADEP_mvt", pa.string()),
    ("MVT_TIME_UTC_mvt", pa.timestamp("us", tz="UTC")),
    ("icao24", pa.string()), ("callsign", pa.string()), ("timestamp", pa.timestamp("us", tz="UTC")),
    ("latitude", pa.float64()), ("longitude", pa.float64()), ("ground_speed", pa.float32()),
    ("track", pa.float32()), ("altitude", pa.float32()), ("vertical_rate", pa.float32()),
    ("on_ground", pa.bool_()), ("match_method", pa.string()), ("source_date", pa.string()),
])
# Readsb units: knots, degrees, feet, feet/minute. On ground, trace track is true heading.
# Altitude/rate follow readsb's barometric-or-geometric trace values, not a unified vertical datum.
# Missing source values stay null; altitude='ground' maps to on_ground=True and altitude=null.

def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def filtered_rows(document, candidates, day):
    hex_id = str(document.get("icao", "")).lower()
    require(re.fullmatch(r"[0-9a-f]{6}", hex_id) is not None, "Missing/invalid ICAO24 in candidate trace")
    base = finite(document.get("timestamp"))
    require(base is not None, "Invalid trace base timestamp")
    left = pd.Timestamp(day, tz="UTC").timestamp()
    right = left + 86400
    current_callsign, callsign_at = "", float("-inf")
    previous = float("-inf")
    for point_number, point in enumerate(document["trace"]):
        if point_number % 1024 == 0:
            check_cancelled()
        require(isinstance(point, list) and len(point) >= 8, "Malformed trace point")
        offset = finite(point[0])
        require(offset is not None, "Non-finite trace timestamp")
        stamp = base + offset
        require(stamp >= previous, "Trace timestamps are not ordered")
        previous = stamp
        flags = int(point[6] or 0)
        if flags & 2:
            current_callsign, callsign_at = "", float("-inf")
        detail = point[8] if len(point) > 8 and isinstance(point[8], dict) else {}
        if "flight" in detail:
            current_callsign, callsign_at = norm(detail["flight"]), stamp
        if not left <= stamp < right or stamp - callsign_at > 300:
            continue
        flights = candidates.get(current_callsign, [])
        if not flights:
            continue
        lat, lon = finite(point[1]), finite(point[2])
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        possible = [f for f in flights if f["window_start"].timestamp() <= stamp <= f["window_end"].timestamp()
                    and f["min_lon"] <= lon <= f["max_lon"] and f["min_lat"] <= lat <= f["max_lat"]]
        if len(possible) != 1:
            continue
        flight = possible[0]
        altitude = finite(point[3])
        yield {"MVT_ID_mvt": flight["MVT_ID_mvt"], "FLIGHT_ID_mvt": flight["FLIGHT_ID_mvt"],
               "ADEP_mvt": flight["ADEP_mvt"], "MVT_TIME_UTC_mvt": flight["MVT_TIME_UTC_mvt"],
               "icao24": hex_id, "callsign": current_callsign,
               "timestamp": pd.Timestamp(round(stamp * 1_000_000), unit="us", tz="UTC"),
               "latitude": lat, "longitude": lon, "ground_speed": finite(point[4]),
               "track": finite(point[5]), "altitude": altitude, "vertical_rate": finite(point[7]),
               "on_ground": True if point[3] == "ground" else (False if altitude is not None else None),
               "match_method": "exact_callsign+unique_window+airport", "source_date": day}

def staging_database(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE points (mvt TEXT, hex TEXT, ts TEXT, lat REAL, lon REAL, payload TEXT, "
               "PRIMARY KEY (mvt,hex,ts,lat,lon)) WITHOUT ROWID")
    return db

def insert_rows(db, rows):
    def values():
        for row in rows:
            serial = dict(row)
            for key in ["timestamp", "MVT_TIME_UTC_mvt"]:
                serial[key] = serial[key].isoformat()
            yield (row["MVT_ID_mvt"], row["icao24"], serial["timestamp"], row["latitude"], row["longitude"],
                   json.dumps(serial, allow_nan=False))
    db.executemany("INSERT OR IGNORE INTO points VALUES (?,?,?,?,?,?)", values())

def staged_batches(db):
    cursor = db.execute("SELECT payload FROM points ORDER BY mvt,hex,ts,lat,lon")
    while rows := cursor.fetchmany(BATCH_ROWS):
        items = [json.loads(r[0]) for r in rows]
        for item in items:
            for key in ["timestamp", "MVT_TIME_UTC_mvt"]:
                item[key] = dt.datetime.fromisoformat(item[key])
        yield pa.Table.from_pylist(items, schema=SCHEMA)


def fetch_day(day, flights, temp_dir):
    log(f"{day} [1/6 Preparation] {len(flights):,} relevant departures; mapping callsigns …", flush=True)
    candidates = collections.defaultdict(list)
    for flight in flights.to_dict("records"):
        if flight["match_callsign"]:
            candidates[flight["match_callsign"]].append(flight)
    log(f"{day} [2/6 Release search] Looking for the preferred release and alternatives …", flush=True)
    repo, tags = release_tags(day)
    errors = []
    for attempt, tag in enumerate(tags):
        log(f"{day} [2/6 Release {attempt + 1}/{len(tags)}] {repo}: {tag}; querying archive parts …", flush=True)
        path = Path(temp_dir) / f"day-{attempt}.sqlite"
        db = staging_database(path)
        try:
            assets = release_assets(repo, tag)
            status = DownloadProgress(day, sum(a["size"] for a in assets))
            log(f"{day} [3/6 Start] {len(assets)} archive part(s), "
                  f"{status.total_bytes / 1e9:.2f} GB; download and filtering run concurrently.", flush=True)
            candidate_aircraft = set()
            with HTTPParts(assets, progress=status) as raw, io.BufferedReader(raw, buffer_size=1024*1024) as stream:
                for document in trace_documents(stream, set(candidates), progress=status):
                    candidate_aircraft.add(document.get("icao"))
                    insert_rows(db, filtered_rows(document, candidates, day))
                    status.candidates = len(candidate_aircraft)
                    status.rows = db.total_changes
                    status.report()
            status.report(force=True)
            log(f"{day} [4/6 Matching] Removing ambiguous aircraft assignments and counting points …", flush=True)
            # Conflicting aircraft for one departure are not reliable callsign matches.
            ambiguous = db.execute("SELECT mvt FROM points GROUP BY mvt HAVING COUNT(DISTINCT hex)>1").fetchall()
            db.executemany("DELETE FROM points WHERE mvt=?", ambiguous)
            db.commit()
            counts = dict(db.execute("SELECT mvt,COUNT(*) FROM points GROUP BY mvt"))
            check_cancelled()
            daily_path = Path(temp_dir) / "filtered.parquet"
            log(f"{day} [4/6 Daily file] Writing filtered observations for the main process …")
            with pq.ParquetWriter(daily_path, SCHEMA, compression="zstd") as writer:
                for table in staged_batches(db):
                    check_cancelled()
                    writer.write_table(table)
            with open(daily_path, "rb") as handle:
                os.fsync(handle.fileno())
            return {"day_file": str(daily_path), "release": tag, "repository": repo, "candidate_aircraft": len(candidate_aircraft),
                    "matched_departures": len(counts), "retained_observations": sum(counts.values()),
                    "ambiguous_departures_discarded": len(ambiguous),
                    "matched_movement_ids": sorted(counts), "release_errors": errors}
        except CancelledError:
            raise
        except Exception as exc:
            message = f"{tag}: {type(exc).__name__}: {exc}"
            errors.append(message)
            log("Release failed:", message, flush=True)
            (Path(temp_dir) / "filtered.parquet").unlink(missing_ok=True)
        finally:
            db.close()
            path.unlink(missing_ok=True)
    raise RuntimeError("All available releases failed: " + " | ".join(errors))


def process_day(day, flights, temp_dir):
    check_cancelled()
    started = time.monotonic()
    result = fetch_day(day, flights, temp_dir)
    result["download_processing_seconds"] = round(time.monotonic() - started, 1)
    result["worker_pid"] = os.getpid()
    return result
