"""Small synthetic data checks; no model or numerical framework imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_vla_latency_v1 import analyze, summarize


def fixture(signal=True):
    rows = []
    for mode in ['disabled', 'fixed', 'adaptive']:
        for rep in range(3):
            for ep in range(3):
                for frame in range(8):
                    adaptive = mode == 'adaptive' and signal
                    r = {'episode': ep, 'frame': frame, 'repeat': rep, 'mode': mode,
                         'wall_seconds': .061 + (.02 * frame if adaptive else 0) + rep * .0001,
                         'retained_tokens': 50 + frame * 10 if adaptive else 100,
                         'controller_history_before': [float(frame)], 'max_reserved_bytes': 16 * 1024**3}
                    if mode == 'disabled':
                        r.update(native_tolerance_passed=True, native_exact=True, native_max_abs_error=0.)
                    rows.append(r)
    return rows


def write_fixture(root, rows):
    native = root / 'native.jsonl'
    native.write_text('{}\n' * 24)
    fast = root / 'fastvla.jsonl'
    fast.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    common = {'completed': True, 'cohort_sha256': 'synthetic_fixture_only', 'max_reserved_bytes': 16 * 1024**3}
    (root / 'native_summary.json').write_text(json.dumps(dict(common, rows=24, rows_sha256=hashlib.sha256(native.read_bytes()).hexdigest())))
    (root / 'fastvla_summary.json').write_text(json.dumps(dict(common, rows=len(rows), native_check_all=True,
        native_check_count=72, rows_sha256=hashlib.sha256(fast.read_bytes()).hexdigest())))


class AnalysisTests(unittest.TestCase):
    def test_known_signal_passes_and_constant_replay_stops(self):
        for signal in (True, False):
            with tempfile.TemporaryDirectory() as d:
                write_fixture(Path(d), fixture(signal))
                r = analyze(d)
                self.assertEqual(r['candidate_for_closed_loop_design'], signal)

    def test_missing_repeat_rejected(self):
        rows = [r for r in fixture() if r['mode'] == 'adaptive']
        with self.assertRaises(AssertionError): summarize(rows[1:])

    def test_controller_state_mismatch_rejected(self):
        rows = [r for r in fixture() if r['mode'] == 'adaptive']
        rows[0]['controller_history_before'] = [-1.]
        with self.assertRaises(AssertionError): summarize(rows)

    def test_invalid_native_control_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = fixture()
            rows[0]['native_tolerance_passed'] = False
            write_fixture(Path(d), rows)
            with self.assertRaises(AssertionError): analyze(d)

    def test_corrupt_rows_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_fixture(Path(d), fixture())
            p = Path(d) / 'fastvla.jsonl'
            p.write_text(p.read_text() + '\n')
            with self.assertRaises(AssertionError): analyze(d)


if __name__ == '__main__': unittest.main()
