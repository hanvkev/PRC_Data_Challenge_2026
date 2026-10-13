"""Regression tests of the 04 ADS-B processing: .venv/bin/python -m unittest discover -s data/tests -v."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adsb_cleanup_worker as w


def frame(start="2025-01-31T23:59:40Z", *, periods=61, mvt="m1", aircraft="abc123",
          target="2025-02-01T00:00:10Z", moving=True):
    stamps = pd.date_range(start, periods=periods, freq="2s")
    x = np.maximum(np.arange(periods)-10, 0) * .00015 if moving else np.zeros(periods)
    return pd.DataFrame(dict(MVT_ID_mvt=mvt, FLIGHT_ID_mvt="challenge_"+mvt,
        ADEP_mvt="EGLL", MVT_TIME_UTC_mvt=pd.Timestamp(target), icao24=aircraft,
        callsign="TEST", timestamp=stamps, latitude=51.47, longitude=-.45+x,
        ground_speed=np.where(x>0, 10., 0.), track=90., altitude=0., vertical_rate=0.,
        on_ground=True, match_method="test", source_date=stamps.strftime("%Y-%m-%d")))


def prepared(data, config=None):
    return w.prepare_movement(w.normalize(data), config or w.Config())


class SelectionTests(unittest.TestCase):
    def test_previous_arrival_departure_and_exact_duplicates(self):
        arrival = frame("2025-01-31T22:00:00Z")
        departure = frame()
        result, audit, candidates = prepared(pd.concat([arrival, departure, departure.iloc[:2]]))
        self.assertEqual(audit["n_physical_flights"], 2)
        self.assertEqual(audit["adsb_flight_selection_rule"], "contains_mvt")
        self.assertEqual(audit["exact_duplicates_removed"], 2)
        self.assertEqual(candidates.selected.sum(), 1)
        self.assertEqual(result.timestamp.min(), departure.timestamp.min())
        self.assertEqual(len(result), 121)
        self.assertTrue(result.timestamp.diff().dropna().eq(pd.Timedelta("1s")).all())
        self.assertEqual(result.FLIGHT_ID_mvt.unique().tolist(), ["challenge_m1"])
        self.assertNotIn("flight_id", result)
        self.assertEqual(result.MVT_TIME_UTC_mvt.nunique(), 1)
        self.assertTrue(result.on_ground.all())

    def test_after_has_priority_over_closer_before(self):
        before = frame("2025-01-31T23:58:08Z")  # stop two seconds before target
        after = frame("2025-02-01T00:02:00Z", aircraft="bbb222")
        result, audit, _ = prepared(pd.concat([before, after]))
        self.assertEqual(audit["adsb_flight_selection_rule"], "starts_after_mvt")
        self.assertEqual(result.icao24.unique().tolist(), ["bbb222"])
        record = w.movement_record(result)
        self.assertTrue(record["adsb_movement_available"])
        self.assertLess(record["adsb_movement_to_mvt_sec"], 0)

    def test_latest_before_and_large_positive(self):
        data = frame("2025-01-31T20:00:00Z")
        result, audit, _ = prepared(data)
        self.assertEqual(audit["adsb_flight_selection_rule"], "ends_before_mvt")
        self.assertGreater(w.movement_record(result)["adsb_movement_to_mvt_sec"], 7200)

    def test_overlapping_contains_and_genuine_ties(self):
        older = frame("2025-01-31T23:59:00Z", periods=100)
        newer = frame(aircraft="bbb222")
        result, audit, _ = prepared(pd.concat([older, newer]))
        self.assertEqual(result.icao24.unique().tolist(), ["bbb222"])
        self.assertEqual(audit["adsb_flight_selection_rule"], "contains_mvt")
        result, audit, candidates = prepared(pd.concat([frame(), newer]))
        self.assertIsNone(result)
        self.assertEqual(audit["unresolved_reason"], "temporal_tie_contains_mvt")
        self.assertFalse(candidates.selected.any())

    def test_after_and_before_ties_unresolved(self):
        for start in ["2025-02-01T00:05:00Z", "2025-01-31T22:00:00Z"]:
            result, audit, _ = prepared(pd.concat([frame(start), frame(start, aircraft="bbb222")]))
            self.assertIsNone(result)
            self.assertIn("temporal_tie", audit["unresolved_reason"])

    def test_failed_movement_and_metadata(self):
        result, _, _ = prepared(frame(moving=False))
        record = w.movement_record(result)
        self.assertFalse(record["adsb_movement_available"])
        self.assertTrue(pd.isna(record["adsb_movement_start_time"]))
        self.assertIsNone(record["adsb_movement_to_mvt_sec"])
        data = frame()
        data.loc[0, "MVT_TIME_UTC_mvt"] += pd.Timedelta("1h")
        result, audit, _ = prepared(data)
        self.assertIsNone(result)
        self.assertEqual(audit["unresolved_reason"], "inconsistent_or_missing_movement_metadata")
        self.assertIsNone(audit["MVT_TIME_UTC_mvt"])

    def test_missing_flight_id_without_nm_record(self):
        # Departures without NM flight plan have no FLIGHT_ID_mvt; they are processed like the others.
        result, audit, _ = prepared(frame().assign(FLIGHT_ID_mvt=None))
        self.assertIsNotNone(result)
        self.assertIsNone(audit["unresolved_reason"])
        self.assertTrue(result.FLIGHT_ID_mvt.isna().all())
        self.assertTrue(w.movement_record(result)["adsb_movement_available"])
        data = frame()
        data.loc[0, "FLIGHT_ID_mvt"] = None  # partly missing stays inconsistent
        result, audit, _ = prepared(data)
        self.assertIsNone(result)
        self.assertEqual(audit["unresolved_reason"], "inconsistent_or_missing_movement_metadata")

    def test_singletons_and_conflicting_observations_are_explicit(self):
        result, audit, _ = prepared(frame(periods=1))
        self.assertIsNone(result)
        self.assertEqual(audit["unassigned_points"], 1)
        conflicting = frame().iloc[:1].copy()
        conflicting["longitude"] += .1
        result, audit, _ = prepared(pd.concat([frame(), conflicting]))
        self.assertIsNone(result)
        self.assertEqual(audit["unresolved_reason"], "conflicting_same_timestamp_observations")

    def test_guards_do_not_truncate(self):
        with self.assertRaisesRegex(MemoryError, "Movement RAM guard"):
            prepared(frame(), w.Config(max_movement_rows=3))
        with self.assertRaisesRegex(MemoryError, "Resampling RAM guard"):
            prepared(frame(), w.Config(max_resampled_points=3))

    def test_stable_hash_and_worker_caps(self):
        command = [sys.executable, "-c", "import sys; sys.path.insert(0,'data'); import adsb_cleanup_worker as w; print(w.stable_bucket('mvt跨月',512))"]
        results = [subprocess.check_output(command, env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip() for seed in ["1", "54321"]]
        self.assertEqual(results[0], results[1])
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}):
            self.assertLessEqual(w.resource_report(w.Config())["effective_workers"], 2)


class PipelineTests(unittest.TestCase):
    def inputs(self, root):
        departure = frame()
        arrival = frame("2025-01-31T22:00:00Z")
        stopped = frame(mvt="stopped", moving=False)
        tie = pd.concat([frame(mvt="tie"), frame(mvt="tie", aircraft="bbb222")])
        singleton = frame(mvt="singleton", periods=1)
        data = pd.concat([departure, arrival, stopped, tie, singleton, departure.iloc[:1]], ignore_index=True)
        paths = [root / "2025-01.parquet", root / "2025-02.parquet"]
        for month, path in zip([1, 2], paths):
            # MVTs deliberately span files and every 7-row read batch.
            data[data.timestamp.dt.month.eq(month)].to_parquet(path, index=False)
        return paths

    def test_end_to_end_spawn_and_persisted_only_stage_b(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.inputs(root)
            original = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            config = w.Config(max_workers=2, n_buckets=4, batch_rows=7, staging_buffer_rows=15)
            with self.assertRaisesRegex(ValueError, "requires successfully"):
                w.stage_b(root / "missing", root / "bad.parquet", config)
            report = w.stage_a(paths, root / "processed", root / "scratch", config)
            self.assertEqual(report["raw"]["unique_MVT_ID_mvt"], 4)
            self.assertEqual(report["raw"]["mvts_multiple_source_dates"], 3)
            self.assertEqual(report["raw"]["mvts_multiple_monthly_files"], 3)
            self.assertEqual(report["totals"]["unresolved"], 2)
            self.assertEqual(report["validation"]["persisted_mvts"], 2)
            self.assertEqual(original, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            for path in paths:
                path.unlink()  # proves Stage B never rereads raw sources
            feature_path = root / "features.parquet"
            summary = w.stage_b(root / "processed", feature_path, config)
            self.assertEqual(summary["rows"], 4)
            self.assertEqual(summary["valid_movement_detections"], 1)
            self.assertEqual(summary["failed_movement_detections"], 1)
            features = pd.read_parquet(feature_path)
            self.assertFalse(features.MVT_ID_mvt.duplicated().any())
            self.assertEqual(pq.ParquetFile(feature_path).metadata.row_group(0).column(0).compression, "ZSTD")
            diagnostics = w.diagnostics(root / "processed", feature_path)
            self.assertEqual(diagnostics["totals"]["raw_coverage_pct"], 25)
            self.assertEqual(diagnostics["distribution"]["count"], 1)
            self.assertTrue(diagnostics["examples"]["arrival_then_departure"])
            with self.assertRaises(FileExistsError):
                w.stage_b(root / "processed", feature_path, config)
            # Corrupt persisted timestamps: the success marker alone is insufficient.
            path = next(p for p in (root / "processed/buckets").glob("*/trajectories.parquet") if pq.ParquetFile(p).metadata.num_rows)
            table = pd.read_parquet(path)
            table.loc[1, "timestamp"] = table.timestamp.iloc[0]
            table.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicated trajectory"):
                w.stage_b(root / "processed", root / "corrupt.parquet", config)
            self.assertFalse((root / "corrupt.parquet").exists())

    def test_failed_run_is_not_published_and_can_be_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.inputs(root)
            with self.assertRaisesRegex(MemoryError, "Bucket RAM guard"):
                w.stage_a(paths, root / "out", root / "scratch", w.Config(n_buckets=1, max_bucket_rows=2))
            self.assertFalse((root / "out").exists())
            result = w.stage_a(paths, root / "out", root / "scratch", w.Config(n_buckets=4, max_workers=1))
            self.assertEqual(result["validation"]["raw_mvts"], 4)


if __name__ == "__main__":
    unittest.main()
