"""Measurement tests, not fabricated experimental observations."""
import copy
import unittest
from analyze_late_binding_c2c_v1 import analyze, validate_records


def records(stale=False):
    rows = []
    def score(index):
        return dict(pred=index, probs=[int(i==index) for i in range(4)], option_mass=.9, top_text=' ABCD'[index+1])
    for i in range(4):
        for shift in range(4):
            order = [(j+shift)%4 for j in range(4)]
            target = order.index(i)
            rows.append(dict(id=str(i), shift=shift, order=order, original_correct=i,
                correct=target, prefix_sha256=str(i), original={m:score(i) for m in ('native','early')},
                scores={m:score(i if stale and m=='late' else target) for m in ('native','late','early')}))
    return rows


class MeasurementTests(unittest.TestCase):
    def test_perfect_equivariance_is_null(self):
        a = analyze(records(), resamples=30)
        self.assertEqual(a['estimates']['late_native_did'], 0)
        self.assertEqual(a['estimates']['early_late_did'], 0)
        self.assertEqual(a['development_decision'], 'no_predefined_candidate_signal_do_not_automatically_scale_this_branch')

    def test_stale_letter_trap_has_expected_signed_effect(self):
        a = analyze(records(stale=True), resamples=30)
        self.assertEqual(a['estimates']['late_native_did'], -1)
        self.assertEqual(a['estimates']['early_late_did'], 1)
        self.assertEqual(a['estimates']['late_native_stale_label_gap'], 1)
        self.assertEqual(a['conditional_native_and_identity_correct']['n'], 12)
        self.assertEqual(a['question_bootstrap_descriptive_ci95']['late_native_did'], [-1,-1])

    def test_incomplete_or_duplicate_question_is_rejected(self):
        for rows in (records()[:-1], records()+records()[:1]):
            with self.assertRaises(AssertionError):
                validate_records(rows)

    def test_changed_sender_prefix_is_rejected(self):
        rows = copy.deepcopy(records())
        rows[1]['prefix_sha256'] = 'changed'
        with self.assertRaises(AssertionError):
            validate_records(rows)


if __name__ == '__main__':
    unittest.main()
