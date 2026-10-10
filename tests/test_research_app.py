import io
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from research_app.catalog import Catalog, annotation_votes, number
from research_app.server import handler_for

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog(ROOT)

    def test_manifest_and_evaluation_cohorts(self):
        meta = self.catalog.metadata()
        self.assertEqual(meta['splits'], {'train': {'videos': 19, 'frames': 863}, 'val': {'videos': 6, 'frames': 169}, 'test': {'videos': 3, 'frames': 218}})
        self.assertTrue(all(row['matches_test'] for row in meta['coverage'].values()))

    def test_metrics_are_csv_values_and_mode_specific(self):
        raw, phys = [next(b for b in self.catalog.benchmarks if b['model'] == 'fpn' and b['mode'] == m) for m in ['raw', 'physics_enforced']]
        self.assertAlmostEqual(raw['dice'], 0.9609261133841106)
        self.assertEqual(raw['dice'], phys['dice'])
        self.assertNotEqual(raw['war_mae'], phys['war_mae'])

    def test_verified_common_frame_and_missing_prediction(self):
        frame = self.catalog.frame('video 21', 'ezgif-frame-040')
        self.assertTrue(all(m['figure'] for m in frame['models']))
        for model in frame['models']:
            for mode in ['raw', 'physics_enforced']:
                image = Image.open(io.BytesIO(self.catalog.panel(model['id'], frame['video'], frame['stem'], 'prediction', mode)))
                self.assertEqual(image.width, image.height)
        missing = self.catalog.frame('video 1', 'ezgif-frame-001')
        self.assertIsNone(next(m for m in missing['models'] if m['id'] == 'fpn')['figure'])
        with self.assertRaises(KeyError):
            self.catalog.panel('fpn', 'video 1', 'ezgif-frame-001', 'prediction', 'raw')

    def test_error_maps_cannot_be_mislabelled(self):
        with self.assertRaises(KeyError):
            self.catalog.panel('unetpp', 'video 21', 'ezgif-frame-040', 'error', 'physics_enforced')
        with self.assertRaises(KeyError):
            self.catalog.panel('segformer_b1', 'video 21', 'ezgif-frame-040', 'error', 'raw')
        self.assertTrue(self.catalog.panel('segformer_b1', 'video 21', 'ezgif-frame-040', 'error', 'physics_enforced').startswith(b'\x89PNG'))

    def test_xai_run_is_separate_and_missing_figures_explicit(self):
        frame = self.catalog.frame('video 1', 'ezgif-frame-053')
        self.assertEqual(len(frame['xai_figures']), 1)
        self.assertAlmostEqual(frame['xai_metrics']['dice'], .9902459979)
        self.assertEqual(self.catalog.frame('video 21', 'ezgif-frame-040')['xai_figures'], [])

    def test_export_metrics_source_matches_segformer_figure(self):
        for key, row in self.catalog.seg_figures.items():
            model = next(m for m in self.catalog.frame(*key)['models'] if m['id'] == 'segformer_b1')
            self.assertEqual(model['values']['war_pred_raw'], float(row['war_pred_raw']))
            self.assertIn('figure-export WAR differs', ' '.join(self.catalog.warnings))

    def test_color_annotations_use_any_nonzero_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(4):
                a = np.zeros((2, 2, 3), dtype=np.uint8)
                a[0, 0, 2] = 1  # would disappear under grayscale >127
                if i < 2:
                    a[1, 1, 2] = 245
                path = Path(tmp) / f'{i}.png'
                Image.fromarray(a).save(path); paths.append(path)
            votes = annotation_votes(paths)
            self.assertEqual(votes.tolist(), [[4, 0], [0, 2]])
            with self.assertRaises(ValueError):
                annotation_votes(paths[:3])

    def test_stored_consensus_matches_actual_annotators(self):
        video, stem = 'video 21', 'ezgif-frame-040'
        votes = annotation_votes([ROOT / self.catalog.data / 'annotations' / f'annotator {n}' / video / f'{stem}.png' for n in range(1, 5)])
        with Image.open(ROOT / self.catalog.data / 'consensus_masks' / video / f'{stem}.png') as image:
            consensus = np.asarray(image.convert('L')) > 127
        np.testing.assert_array_equal(consensus, votes >= 2)

    def test_missing_and_malformed_files_do_not_crash_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'video_split_assignment.csv').write_text('wrong,header\nx,y\n')
            (root/'xai_gradcam_outputs/tables').mkdir(parents=True)
            (root/'xai_gradcam_outputs/tables/xai_summary.json').write_text('{ broken')
            catalog = Catalog(root)
            self.assertEqual(catalog.videos, [])
            self.assertEqual(catalog.xai_summary, {})
            self.assertTrue(catalog.warnings)
            json.dumps(catalog.metadata(), allow_nan=False)

    def test_nonfinite_metrics_are_unavailable(self):
        for value in ['nan', 'inf', '', None, 'not a number']:
            self.assertIsNone(number(value))

    def test_file_and_frame_access_are_allowlisted(self):
        for key in ['../../.git/config', '../README.md', 'not-a-file']:
            with self.assertRaises(KeyError):
                self.catalog.path_for_id(key)
        with self.assertRaises(KeyError):
            self.catalog.frame('../.git', 'config')


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(Catalog(ROOT)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_app_and_javascript_modules(self):
        for path, mime in [('/', 'text/html'), ('/app.js', 'text/javascript'), ('/state.mjs', 'text/javascript'), ('/styles.css', 'text/css'), ('/api/meta', 'application/json')]:
            with urlopen(self.base+path) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(mime, response.headers['Content-Type'])

    def test_unknown_paths_and_invalid_frame(self):
        for path in ['/api/files/../README.md', '/.git/config', '/api/frame?video=unknown&stem=unknown']:
            with self.assertRaises(HTTPError) as ctx:
                urlopen(self.base+path)
            self.assertEqual(ctx.exception.code, 404)

    def test_download_comparison(self):
        with urlopen(self.base+'/api/comparison?video=video+21&stem=ezgif-frame-040&mode=raw') as response:
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertEqual(Image.open(io.BytesIO(response.read())).size, (1440, 980))


if __name__ == '__main__':
    unittest.main()
