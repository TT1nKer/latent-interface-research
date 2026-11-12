"""Correction-gate regressions using recorded failure/diagnostic evidence."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest

from remote.dream_readout_gate_v3 import (POLICY, FINGERPRINT_KEYS,
    validate_diagnostic, project_corrected_full_seconds, validate_probe_baseline,
    project_probe_seconds)
from remote.dream_trace_gate_v2 import EXPECTED_CHECKS

ROOT=Path(__file__).resolve().parent


def fixtures():
    return tuple(json.loads((ROOT/name).read_text()) for name in (
        'results/dream_trace_full_v2.failure.json',
        'results/dream_readout_diagnostic_v1.summary.json'))


def corrected_receipt():
    p={k:'fixed' for k in (*FINGERPRINT_KEYS,'legacy_full_summary_sha256','diagnostic_summary_sha256')}
    p.update(completed=True,stage='full',examples=1000,execution_policy=POLICY,
        numerical_validity_passed=True,numerical_flags={'bf16_reduced_precision_reduction':True},
        independent_author_checks=[dict(condition=c,rounds=d,predictions_equal=True,message_hashes_equal=True)
                                  for c,d in sorted(EXPECTED_CHECKS)],
        setup_seconds=40,terminal_seconds=[60]*26,forward_seconds=[60]*4,
        torch_peak_reserved_bytes=21*1024**3)
    return p


class CorrectionGateTests(unittest.TestCase):
    def test_recorded_diagnosis_supports_bounded_correction_but_not_old_validity(self):
        old,diag=fixtures()
        validate_diagnostic(old,diag)
        self.assertAlmostEqual(project_corrected_full_seconds(old,diag),6719.994564626366)
        self.assertIs(old['numerical_validity_passed'],False)
        bad=copy.deepcopy(diag);bad['comparisons']['cross1_cached_receiver_vs_fresh_author']['equal']=999
        with self.assertRaises(AssertionError):validate_diagnostic(old,bad)
        bad=copy.deepcopy(diag);bad['joint_replay_hidden_exact']=False
        with self.assertRaises(AssertionError):validate_diagnostic(old,bad)

    def test_resource_limits_reject_slow_or_large_run(self):
        old,diag=fixtures();old['elapsed_seconds']*=2
        with self.assertRaises(AssertionError):project_corrected_full_seconds(old,diag)
        old,diag=fixtures();diag['torch_peak_reserved_bytes']=22*1024**3
        with self.assertRaises(AssertionError):project_corrected_full_seconds(old,diag)

    def test_probe_rejects_failed_or_mismatched_corrected_baseline(self):
        p=corrected_receipt();validate_probe_baseline(p,p)
        old,_=fixtures()
        with self.assertRaises(AssertionError):validate_probe_baseline(old,p)
        for key in ('code_sha256','legacy_full_summary_sha256','diagnostic_summary_sha256'):
            with self.assertRaises(AssertionError):validate_probe_baseline(p,dict(p,**{key:'changed'}))
        bad=copy.deepcopy(p);bad['independent_author_checks']=[]
        with self.assertRaises(AssertionError):validate_probe_baseline(bad,p)
        legacy,_=fixtures();self.assertLess(project_probe_seconds(p,legacy),1200)
        bad=copy.deepcopy(p);bad['terminal_seconds']=[500]*26
        with self.assertRaises(AssertionError):project_probe_seconds(bad,legacy)

    def test_frozen_recurrence_helpers_and_primary_rules_unchanged(self):
        amendment=json.loads((ROOT/'remote/dream_readout_amendment_v3.json').read_text())
        for filename,key in [('dream_recurrent_trace_v2.py','base_driver_v2_sha256'),
                             ('dream_trace_protocol_v1.json','base_protocol_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT/'remote'/filename).read_bytes()).hexdigest(),amendment[key])
        trees=[ast.parse((ROOT/'remote'/f).read_text()) for f in ('dream_recurrent_trace_v2.py','dream_recurrent_readout_v3.py')]
        for name in ('forward','classifier','observe','save_state','project'):
            funcs=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            self.assertEqual(ast.dump(funcs[0]),ast.dump(funcs[1]))
        trees=[ast.parse((ROOT/f'analyze_dream_recurrent_v{v}.py').read_text()) for v in (2,3)]
        funcs=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='primary_met') for t in trees]
        self.assertEqual(ast.dump(funcs[0]),ast.dump(funcs[1]))


if __name__=='__main__':unittest.main()
