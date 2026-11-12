"""Sensitivity-measurement tests with synthetic records, not neural evidence."""
import copy
import unittest

from analyze_dream_recurrent_v3 import paired_probe, reference_sensitivity
from remote.dream_trace_gate_v2 import FINGERPRINT_KEYS
from test_analyze_dream_recurrent_v1 import records


class SensitivityTests(unittest.TestCase):
    def test_flagged_rows_keep_original_denominator_in_worst_case_interval(self):
        rows = records()
        summary = dict(completed=True, examples=4, stage='validity',
                       numerical_validity_passed=True, public_reference_all_equal=False,
                       reference_agreement={'self': {'mismatch_ids': ['0']}})
        report = reference_sensitivity(rows, summary, resamples=20)
        # Three fixed scores are1; the flagged score can range from-1 to1.
        self.assertEqual(report['arbitrary_flagged_score_mean_interval'], [.5, 1])
        self.assertEqual(report['retained_examples'], 3)
        self.assertEqual(len(rows), 4)
        summary['reference_agreement']['self']['mismatch_ids'] = [str(i) for i in range(4)]
        report = reference_sensitivity(rows, summary, resamples=20)
        self.assertEqual(report['arbitrary_flagged_score_mean_interval'], [-1, 1])
        self.assertIsNone(report['retained_subset'])

    def test_paired_probe_requires_matching_tokens_and_single_setting_change(self):
        rows = records()
        full = {key: 'same' for key in (*FINGERPRINT_KEYS, 'legacy_full_summary_sha256', 'diagnostic_summary_sha256', 'model_weight_manifest_sha256')}
        full.update(completed=True, stage='full', examples=4, execution_policy='within_environment_v3_corrected_readout',
                    numerical_validity_passed=True, public_reference_all_equal=False,
                    numerical_flags={'bf16_reduced_precision_reduction': True})
        probe = dict(full, stage='sensitivity', numerical_validity_passed=None,
                     original_numerical_flags=full['numerical_flags'],
                     numerical_flags={'bf16_reduced_precision_reduction': False})
        tokens = {str(i): [[i], [i+1]] for i in range(4)}
        plan = {'sensitivity': {'block_starts': [0], 'block_size': 4, 'examples': 4}}
        report = paired_probe(rows, full, copy.deepcopy(rows), probe, tokens, tokens, plan, resamples=20)
        self.assertEqual(report['paired_primary_change'], 0)
        self.assertTrue(report['captured_predictions_and_logits_identical'])
        self.assertFalse(report['fragile_on_tested_setting'])
        bad = copy.deepcopy(tokens)
        bad['0'][0][0] = 100
        with self.assertRaises(AssertionError):
            paired_probe(rows, full, rows, probe, tokens, bad, plan, resamples=20)


if __name__ == '__main__':
    unittest.main()
