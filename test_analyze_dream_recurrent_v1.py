"""Measurement tests only; synthetic records are not neural evidence."""
import copy
import math
import unittest
from analyze_dream_recurrent_v1 import analyze, context_ownership_analysis, PARITY_WEIGHTS


def state(a,b):
    logits=lambda label:[2.0 if i==label else -1.0 for i in range(8)]
    return dict(predictions=[a,b],label_logits=[logits(a),logits(b)],
                message_logits=[logits(a),logits(b)])


def records():
    rows=[]
    for i in range(4):
        native=state(0,1)
        cross=[state(1,0) if r%2 else state(0,1) for r in range(1,7)]
        cells=[dict(source=s,relative_message_rms=0.0,message_cosine=1.0,
                    message_classifier_tv=0.0,incoming_message_label=s,
                    outgoing_message_labels=[s,s],common_receiver_logits=state(s,s)['label_logits'],
                    common_receiver_tv=0.0) for s in (0,1)]
        rows.append(dict(example_id=str(i),gold=1,native=native,cross=cross,
                         self=[copy.deepcopy(native) for _ in range(6)],common_input=cells))
    return rows


class MeasurementTests(unittest.TestCase):
    def test_linear_trend_is_not_parity(self):
        for intercept,slope in [(0,0),(0.1,0.1),(1,-0.1)]:
            estimate=sum(w*(intercept+slope*(r+1)) for r,w in enumerate(PARITY_WEIGHTS))
            self.assertTrue(math.isclose(estimate,0,abs_tol=1e-12))

    def test_known_alternation_and_sample_gate(self):
        report=analyze(records(),dict(completed=True,examples=4,stage='validity',
                       numerical_validity_passed=True,public_reference_all_equal=True),resamples=20)
        self.assertAlmostEqual(report['contrasts']['linear_depth_adjusted_parity'],1)
        self.assertEqual(report['question_bootstrap_descriptive_ci95']['linear_depth_adjusted_parity'],[1,1])
        self.assertTrue(report['repeated_even_peaks_in_mean_curve'])
        self.assertEqual(report['decision'],'validity_sample_only_no_mechanism_decision')
        self.assertEqual(report['observed_one_to_four_losses'],4)
        self.assertEqual(report['of_these_agree_with_self_four'],4)

    def test_corrupt_predictions_rejected(self):
        rows=records()
        rows[0]['cross'][0]['predictions'][0]=2
        with self.assertRaises(AssertionError):
            analyze(rows,dict(completed=True,examples=4,stage='validity'),resamples=20)

    def test_literal_side_measurement_is_invariant_to_agent_renaming(self):
        rows=records()
        data=[dict(example_id=str(i),answer_label='B',
                   answer_options=[('A','alpha'),('B','beta')],
                   private_contexts=['alpha','beta']) for i in range(4)]
        first=context_ownership_analysis(rows,data)
        swapped_rows=copy.deepcopy(rows)
        swapped_data=copy.deepcopy(data)
        for row in swapped_rows:
            for s in [row['native'],*row['cross'],*row['self']]:
                for key in ('predictions','label_logits','message_logits'):
                    s[key].reverse()
        for source in swapped_data:
            source['private_contexts'].reverse()
        second=context_ownership_analysis(swapped_rows,swapped_data)
        self.assertEqual(first['within_question_literal_side_curves'],
                         second['within_question_literal_side_curves'])
        self.assertEqual(first['linear_depth_adjusted_literal_side_contrast'],
                         second['linear_depth_adjusted_literal_side_contrast'])
        self.assertEqual(first['extra_model_forwards'],0)


if __name__=='__main__':
    unittest.main()
