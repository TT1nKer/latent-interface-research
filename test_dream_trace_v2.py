"""Receipt safety tests; no torch imports and no model loading."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from remote.dream_trace_gate_v2 import (
    EXPECTED_CHECKS, FINGERPRINT_KEYS, full_resource_gate,
    implementation_checks_pass, validate_prior,
)


def receipt():
    r = {key: 'same' for key in FINGERPRINT_KEYS}
    r.update(completed=True, stage='validity', examples=16,
             execution_policy='within_environment_v2', numerical_validity_passed=True,
             public_reference_all_equal=False,
             independent_author_checks=[dict(condition=c, rounds=d,
                 predictions_equal=True, message_hashes_equal=True)
                 for c, d in sorted(EXPECTED_CHECKS)],
             setup_seconds=31.44, elapsed_seconds=93.56,
             torch_peak_reserved_bytes=16 * 1024**3)
    return r


class ReceiptTests(unittest.TestCase):
    def test_reference_failure_allowed_but_implementation_failure_rejected(self):
        r = receipt()
        validate_prior(r, r, 'validity')
        r['independent_author_checks'][0]['predictions_equal'] = False
        with self.assertRaises(AssertionError):
            validate_prior(r, r, 'validity')

    def test_missing_duplicate_and_hash_checks_fail_closed(self):
        checks = receipt()['independent_author_checks']
        self.assertFalse(implementation_checks_pass([]))
        self.assertFalse(implementation_checks_pass(checks[:-1]))
        self.assertFalse(implementation_checks_pass([checks[0]] * 6))
        bad = copy.deepcopy(checks)
        next(c for c in bad if c['condition'] == 'self').pop('message_hashes_equal')
        self.assertFalse(implementation_checks_pass(bad))

    def test_stale_code_or_runtime_cannot_authorize_run(self):
        r = receipt()
        for key in FINGERPRINT_KEYS:
            changed = dict(r, **{key: 'changed'})
            with self.assertRaises(AssertionError):
                validate_prior(r, changed, 'validity')

    def test_budget_stops_slow_run(self):
        r = receipt()
        limits = {'budgets': {'full_wall_seconds': 7200}}
        self.assertLess(full_resource_gate(r, limits), 7200)
        r['elapsed_seconds'] = 160
        with self.assertRaises(AssertionError):
            full_resource_gate(r, limits)

    def test_original_protocol_and_numerical_helpers_preserved(self):
        base = Path(__file__).resolve().parent / 'remote'
        a = json.loads((base / 'dream_trace_amendment_v2.json').read_text())
        for filename, key in [('dream_trace_protocol_v1.json', 'base_protocol_sha256'),
                              ('dream_recurrent_trace_v1.py', 'base_driver_sha256')]:
            self.assertEqual(hashlib.sha256((base / filename).read_bytes()).hexdigest(), a[key])
        trees = [ast.parse((base / f'dream_recurrent_trace_v{v}.py').read_text()) for v in (1, 2)]
        for name in ('forward', 'classifier', 'observe', 'save_state', 'project'):
            funcs = [next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)
                          and n.name == name) for t in trees]
            self.assertEqual(ast.dump(funcs[0]), ast.dump(funcs[1]))


if __name__ == '__main__':
    unittest.main()
