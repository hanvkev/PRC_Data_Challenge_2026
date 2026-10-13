"""Geometry endpoints, stand matching and spawned Stage-B integration."""
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from test_adsb_cleanup import frame
import adsb_cleanup_worker as w
import adsb_extra_features as e
import adsb_airport_geometry as g


class GeometryTests(unittest.TestCase):
    def test_endpoints_midpoint_and_zero_baseline(self):
        runway = dict(threshold=(0., 0.), opposite=(0., .03))
        stand = (.01, -.01)
        for point, expected in [(stand, 0.), (runway['threshold'], 1.)]:
            values = e.point_features(point, stand, runway)
            for c in e.PROGRESS_COLUMNS:
                self.assertAlmostEqual(values[c], expected, places=8)
        mid = e.point_features((.005, -.005), stand, runway)
        for c in e.PROGRESS_COLUMNS:
            self.assertAlmostEqual(mid[c], .5, places=6)
        zero = e.point_features((0., -.005), (0., -.01), runway)
        self.assertTrue(np.isnan(zero['adsb_movement_progress_perpendicular']))
        away = e.point_features((.02, -.02), stand, runway)
        self.assertLess(away['adsb_movement_progress_direct'], 0)
        missing = e.point_features(stand, None, runway)
        self.assertTrue(np.isnan(missing['adsb_movement_to_gate_direct_m']))
        self.assertGreater(missing['adsb_movement_to_runway_direct_m'], 0)

    def test_conflict_fallback_and_ltfm_suffix(self):
        osm = pd.DataFrame([dict(aeroway='parking_position', ref='01', longitude=0., latitude=0.),
                            dict(aeroway='parking_position', ref='01', longitude=1., latitude=0.)])
        xp = pd.DataFrame([dict(ref='GATE 001-(A)', longitude=.01, latitude=.02)])
        stands, audit = g.select_stand_positions('LTFM', pd.Series(['01']), osm, xp)
        self.assertEqual(stands['01'], (.01, .02))
        self.assertEqual(audit.osm_position_source.iloc[0], 'ambiguous_spread')
        self.assertEqual(audit.stand_position_source.iloc[0], 'xplane_fallback')


class IdentityTests(unittest.TestCase):
    def test_numeric_identity_formats_and_unsafe_values(self):
        for value in (182768532.0, 182768532, '182768532', '182768532.0', ' 0182768532.000 '):
            self.assertEqual(e.canonical_mvt_id(value), '182768532')
        self.assertEqual(e.canonical_mvt_id('m1'), 'm1')
        self.assertEqual(e.canonical_mvt_id('9007199254740993'), '9007199254740993')
        for bad in (None, np.nan, '', 1.5, '1.5', float(2**53)):
            with self.assertRaises(ValueError):
                e.canonical_mvt_id(bad)

    def test_float_dep_and_ranking_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = pd.DataFrame(dict(MVT_ID_mvt=[182768532.0, 202501153.0], PHASE_mvt='DEP',
                ADEP_mvt='EGLL', MVT_TIME_UTC_mvt=pd.Timestamp('2025-02-01T00:00:10Z'),
                STAND_mvt='001', RUNWAY_mvt='9L'))
            paths = [root/'train_DEP.parquet', root/'ranking.parquet']
            for i, path in enumerate(paths):
                rows.iloc[[i]].to_parquet(path, index=False)
            (root/'xplane_EGLL_1.json').write_text(json.dumps(dict(icao='EGLL', apt_dat='')))
            raw_stands = pd.DataFrame([dict(ref='001', longitude=-.45, latitude=51.47)])
            airport = dict(stand_candidates=raw_stands, runways={'09L': dict(threshold=(-.4, 51.47), opposite=(-.35, 51.47))})
            with patch.object(g, 'data_dir', root), patch.object(g, 'parse_airport_geometry', return_value=airport), patch.object(g, 'load_osm_stands', return_value=raw_stands.assign(aeroway='parking_position')):
                e.build_context(paths, root/'context', 4)
            for mvt in ['182768532', '202501153']:
                bucket = w.stable_bucket(mvt, 4)
                record = json.loads((root/'context'/f'bucket-{bucket:04d}.json').read_text())[mvt]
                self.assertEqual(record['status'], 'matched')
                self.assertEqual(record['stand'], [-.45, 51.47])


class IntegrationTests(unittest.TestCase):
    def test_spawned_extended_stage_b(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = pd.concat([frame('2025-01-31T22:00:00Z'),
                             frame('2025-01-31T23:59:40.250Z'),
                             frame(mvt='stationary', moving=False)], ignore_index=True)
            source = root/'2025-01.parquet'
            raw.to_parquet(source, index=False)
            config = w.Config(max_workers=1, n_buckets=2)
            processed = root/'processed'
            w.stage_a([source], processed, root/'scratch', config)
            context = root/'context'
            context.mkdir()
            records = {m: dict(airport='EGLL', mvt_time='2025-02-01T00:00:10Z',
                              stand=(-.45, 51.47), runway=dict(threshold=(-.4, 51.47), opposite=(-.35, 51.47)),
                              status='matched') for m in ('m1', 'stationary')}
            for bucket in range(2):
                w.json_write(context/f'bucket-{bucket:04d}.json', {m: r for m,r in records.items() if w.stable_bucket(m, 2)==bucket})
            output = root/'features.parquet'
            w.stage_b(processed, output, config, partition_function=e.feature_partition,
                      partition_args=(str(context),), feature_schema=e.SCHEMA, extra_validator=e.validate_extra)
            features = pd.read_parquet(output).set_index('MVT_ID_mvt')
            self.assertEqual(len(features), 2)
            self.assertEqual(features.loc['m1', 'adsb_first_observation_time'], pd.Timestamp('2025-01-31T23:59:40.250Z'))
            self.assertAlmostEqual(features.loc['m1', 'adsb_first_observation_to_mvt_sec'], 29.75)
            self.assertTrue(features.loc['m1', 'adsb_movement_available'])
            self.assertTrue(np.isfinite(features.loc['m1', e.DISTANCE_COLUMNS].astype(float)).all())
            self.assertFalse(features.loc['stationary', 'adsb_movement_available'])
            self.assertTrue(features.loc['stationary', e.DISTANCE_COLUMNS].isna().all())
            self.assertTrue(pd.notna(features.loc['stationary', 'adsb_first_observation_time']))
            broken = features.reset_index()
            broken.loc[broken.adsb_movement_available, 'adsb_geometry_status'] = 'missing_clean_metadata'
            broken[e.DISTANCE_COLUMNS+e.PROGRESS_COLUMNS] = np.nan
            pq.write_table(pa.Table.from_pandas(broken, schema=e.SCHEMA, preserve_index=False), root/'broken.parquet')
            with self.assertRaisesRegex(ValueError, 'All detected movements lack clean metadata'):
                e.validate_extra(root/'broken.parquet')
            # Additional validation happens before atomic publication.
            def reject(path):
                raise ValueError('reject extra features')
            with self.assertRaisesRegex(ValueError, 'reject extra'):
                w.stage_b(processed, root/'rejected.parquet', config,
                          partition_function=e.feature_partition, partition_args=(str(context),),
                          feature_schema=e.SCHEMA, extra_validator=reject)
            self.assertFalse((root/'rejected.parquet').exists())


if __name__ == '__main__':
    unittest.main()
