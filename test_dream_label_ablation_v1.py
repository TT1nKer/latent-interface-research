import copy
import unittest

from remote.dream_label_plan_v1 import donor_indices, build_plan, digest, ARMS
from analyze_dream_label_ablation_v1 import analyze, validate_provenance
from analyze_dream_label_tracking_v1 import tracking


class PlanTests(unittest.TestCase):
    def test_derangement_preserves_or_changes_class_and_marginal(self):
        labels = [0]*17 + [1]*11 + [2]*12
        ids = [str(i) for i in range(len(labels))]
        for same in (True, False):
            for seed in (0, 1):
                d = donor_indices(ids, labels, seed, same_class=same)
                self.assertEqual(sorted(d), list(range(40)))
                self.assertTrue(all(i != j and ((labels[i] == labels[j]) == same) for i, j in enumerate(d)))
                self.assertEqual(d, donor_indices(ids, labels, seed, same_class=same))

    def test_impossible_plans_fail(self):
        with self.assertRaises(AssertionError):
            donor_indices(['a', 'b', 'c'], [0, 0, 1], 0, same_class=True)
        with self.assertRaises(AssertionError):
            donor_indices(['a', 'b', 'c'], [0, 0, 1], 0, same_class=False)

    def test_role_direction_and_relabel_invariance(self):
        ids = [str(i) for i in range(32)]
        messages = [[[int(j == i % 8) for j in range(8)],
                     [int(j == (i+3) % 8) for j in range(8)]] for i in range(32)]
        a = build_plan(ids, messages)
        b = build_plan(ids, [x[::-1] for x in messages])
        self.assertEqual(a['receivers'][0]['sender_predicted_labels'], b['receivers'][1]['sender_predicted_labels'])
        self.assertEqual(a['receivers'][0]['donors'], b['receivers'][1]['donors'])
        self.assertEqual(digest(a), digest(copy.deepcopy(a)))


class MeasurementTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(1000):
            gold = i % 8
            conditions = {}
            for arm in ARMS:
                pred = (gold+1) % 8 if arm == 'different_class' else gold
                conditions[arm] = [dict(prediction=pred, incoming_class=pred,
                    label_logits=[float(j == pred) for j in range(8)]) for _ in (0, 1)]
            rows.append(dict(example_id=str(i), gold=gold, conditions=conditions))
        return rows

    def test_strong_control_and_exact_reconstruction(self):
        report = analyze(self.rows(), dict(completed=True, implementation_validity_passed=True, examples=1000), 10)
        self.assertEqual(report['decision'], 'low_dimensional_label_relay_candidate_needs_heldout_confirmation')
        self.assertEqual(report['by_arm_receiver'][-2]['accuracy_change'], 0)

    def test_invalid_run_cannot_be_analyzed(self):
        with self.assertRaises(AssertionError):
            analyze(self.rows(), dict(completed=True, implementation_validity_passed=False, examples=1000), 10)

    def test_accuracy_cancellation_does_not_hide_changed_answers(self):
        rows = self.rows()
        for i, row in enumerate(rows[:100]):
            # All these changes are between two incorrect classes: accuracy is
            # unchanged but the intervention changes more than5% of answers.
            for arm, offset in [('matched', 1), ('within_class_0', 2)]:
                cell = row['conditions'][arm][0]
                cell['prediction'] = (row['gold']+offset) % 8
                cell['label_logits'] = [float(j == cell['prediction']) for j in range(8)]
        report = analyze(rows, dict(completed=True, implementation_validity_passed=True, examples=1000), 10)
        cell = next(c for c in report['by_arm_receiver'] if c['arm'] == 'within_class_0' and c['receiver'] == 0)
        self.assertEqual(cell['accuracy_change'], 0)
        self.assertFalse(cell['operational_tolerance_met'])

    def test_weak_positive_control_blocks_candidate(self):
        rows = self.rows()
        for row in rows:
            row['conditions']['different_class'] = copy.deepcopy(row['conditions']['matched'])
        report = analyze(rows, dict(completed=True, implementation_validity_passed=True, examples=1000), 10)
        self.assertEqual(report['decision'], 'different_class_positive_control_weak_no_bottleneck_inference')

    def test_tracking_uses_sender_and_previous_round(self):
        logits = lambda c: [float(j == c) for j in range(8)]
        native = dict(message_logits=[logits(0), logits(1)])
        cross = [dict(predictions=[1, 0] if d % 2 == 0 else [0, 1],
                      message_logits=[logits(1), logits(0)] if d % 2 == 0 else [logits(0), logits(1)])
                 for d in range(6)]
        own = [dict(predictions=[0, 1], message_logits=[logits(0), logits(1)]) for _ in range(6)]
        result = tracking([dict(gold=1, native=native, cross=cross, self=own)])
        for r in result['receivers']:
            for route in ('cross', 'self'):
                self.assertTrue(all(d['readout_equals_incoming_class'] == 1 and
                                    d['outgoing_retains_incoming_class'] == 1 and
                                    d['readout_equals_initial_ancestral_class'] == 1
                                    for d in r['routes'][route]))

    def test_provenance_checks_actual_donors_not_just_pass_flag(self):
        rows = self.rows()
        old = [dict(example_id=r['example_id'], gold=r['gold'],
                    native={'message_logits': [[float(j == r['gold']) for j in range(8)]]*2},
                    cross=[{'predictions': [r['gold']]*2,
                            'label_logits': [[float(j == r['gold']) for j in range(8)]]*2}]) for r in rows]
        plan = build_plan([r['example_id'] for r in old], [r['native']['message_logits'] for r in old])
        for i, row in enumerate(rows):
            for arm in ARMS:
                for receiver in (0, 1):
                    rp = plan['receivers'][receiver]
                    donor = i if arm == 'label_prefix' else rp['donors'][arm][i]
                    row['conditions'][arm][receiver].update(donor_index=donor,
                        incoming_class=rp['sender_predicted_labels'][donor])
        p = dict(baseline_summary_sha256='s', baseline_per_item_sha256='i', donor_plan_sha256=digest(plan),
                 implementation_sha256={'dream_label_ablation_v1.py':'c','dream_label_plan_v1.py':'h'})
        summary = dict(baseline_summary_sha256='s', baseline_per_item_sha256='i', donor_plan_sha256=digest(plan),
                       code_sha256='c',helper_sha256='h',matched_replay_checks=[
                           dict(receiver=i, predictions_equal=True, logits_exact=True) for i in (0,1)])
        validate_provenance(rows, summary, p, old)
        rows[0]['conditions']['within_class_0'][0]['donor_index'] = 0
        with self.assertRaises(AssertionError):
            validate_provenance(rows, summary, p, old)


if __name__ == '__main__':
    unittest.main()
