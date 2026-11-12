import unittest
import copy
from remote.dream_label_confidence_plan_v1 import pair_plan, softmax, distance, build_plan, digest, ARMS
from analyze_dream_label_confidence_v1 import analyze


class ConfidencePlanTests(unittest.TestCase):
    def test_even_odd_ties_are_bijective_and_same_class(self):
        for scalar in (True,False):
            ids=[str(i) for i in range(19)]
            labels=[0]*8+[1]*11
            values=[[float(i%4)] for i in range(19)]
            d=pair_plan(ids,labels,values,scalar=scalar)
            self.assertEqual(sorted(d),list(range(19)))
            self.assertTrue(all(i!=j and labels[i]==labels[j] for i,j in enumerate(d)))

    def test_nearest_pairs_when_unambiguous(self):
        for scalar in (True,False):
            d=pair_plan(list('abcdef'),[0]*6,[[x] for x in [0,.01,3,3.01,9,9.01]],scalar=scalar)
            self.assertEqual(d,[1,0,3,2,5,4])

    def test_softmax_shift_and_large_logits(self):
        self.assertLess(distance(softmax([10000,9999,9998]),softmax([0,-1,-2])),1e-12)

    def test_plan_ignores_outcomes_and_preserves_sender_roles(self):
        ids=[str(i) for i in range(32)]
        logits=[[[float(j==i%8) for j in range(8)],[float(j==(i+3)%8) for j in range(8)]] for i in range(32)]
        a=build_plan(ids,logits);b=build_plan(ids,[x[::-1] for x in logits])
        self.assertEqual(a['receivers'][0]['donors'],b['receivers'][1]['donors'])
        self.assertEqual(digest(a),digest(build_plan(ids,logits)))
        other=copy.deepcopy(a)
        other['receivers'][0]['matching_quality']['posterior_matched']['posterior']['mean']+=1e-16
        self.assertEqual(digest(a),digest(other))
        other['receivers'][0]['donors']['posterior_matched'][0]=0
        self.assertNotEqual(digest(a),digest(other))

    def test_fixed_candidate_rule_requires_both_receivers(self):
        rows=[]
        for i in range(1000):
            conditions={}
            for arm in ARMS:
                cells=[]
                for r in (0,1):
                    pred=1 if arm=='margin_matched' and r==1 and i<100 else 0
                    cells.append(dict(prediction=pred,incoming_class=0,label_logits=[float(j==pred) for j in range(8)]))
                conditions[arm]=cells
            rows.append(dict(example_id=str(i),gold=0,conditions=conditions))
        previous={'by_arm_receiver':[dict(arm=a,receiver=r,changed_label_rate=.08) for a in ['within_class_0','within_class_1'] for r in (0,1)]}
        result=analyze(rows,dict(completed=True,implementation_validity_passed=True,examples=1000,matching_quality=[]),previous,10)
        self.assertEqual(result['decision'],'eight_score_matching_retains_candidate_behavior')
        self.assertFalse(result['arm_passes_both_receivers']['margin_matched'])


if __name__=='__main__':unittest.main()
