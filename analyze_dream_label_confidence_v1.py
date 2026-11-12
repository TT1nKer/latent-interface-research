"""Fixed development comparison against prior random within-class donors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from remote.dream_label_confidence_plan_v1 import ARMS, build_plan, digest
from analyze_dream_label_ablation_v1 import effect


def validate_provenance(rows, summary, protocol, baseline_rows):
    for key in ('baseline_summary_sha256', 'baseline_per_item_sha256', 'donor_plan_sha256'):
        assert summary[key] == protocol[key], key
    assert summary['code_sha256'] == protocol['implementation_sha256']['dream_label_confidence_v1.py']
    assert summary['helper_sha256'] == protocol['implementation_sha256']['dream_label_confidence_plan_v1.py']
    checks = summary['matched_replay_checks']
    assert len(checks) == 2 and {c['receiver'] for c in checks} == {0, 1}
    assert all(c['predictions_equal'] and c['logits_exact'] for c in checks)
    ids = [r['example_id'] for r in baseline_rows]
    assert [r['example_id'] for r in rows] == ids
    plan = build_plan(ids, [r['native']['message_logits'] for r in baseline_rows])
    assert digest(plan) == protocol['donor_plan_sha256']
    for i, (row, old) in enumerate(zip(rows, baseline_rows, strict=True)):
        assert row['gold'] == old['gold']
        for receiver in (0, 1):
            matched = row['conditions']['matched'][receiver]
            assert matched['prediction'] == old['cross'][0]['predictions'][receiver]
            assert matched['label_logits'] == old['cross'][0]['label_logits'][receiver]
            rp = plan['receivers'][receiver]
            for arm in ARMS:
                expected = i if arm == 'label_prefix' else rp['donors'][arm][i]
                cell = row['conditions'][arm][receiver]
                assert cell['donor_index'] == expected
                assert cell['incoming_class'] == rp['sender_predicted_labels'][expected]


def analyze(rows, summary, previous, resamples=2000):
    assert summary['completed'] and summary['implementation_validity_passed']
    assert len(rows)==summary['examples']==1000
    assert all(set(r['conditions'])==set(ARMS) for r in rows)
    for row in rows:
        for values in row['conditions'].values():
            assert len(values)==2
            for cell in values:
                assert len(cell['label_logits'])==8 and all(math.isfinite(x) for x in cell['label_logits'])
                assert cell['prediction']==max(range(8),key=cell['label_logits'].__getitem__)
    cells=[effect(rows,a,r,resamples) for a in ARMS for r in (0,1)]
    comparisons=[]
    for arm in ARMS[1:]:
        for receiver in (0,1):
            cell=next(c for c in cells if c['arm']==arm and c['receiver']==receiver)
            reference=[c for c in previous['by_arm_receiver'] if c['arm'].startswith('within_class') and c['receiver']==receiver]
            assert len(reference)==2
            old_change=sum(c['changed_label_rate'] for c in reference)/2
            comparisons.append(dict(arm=arm,receiver=receiver,
                changed_label_rate=cell['changed_label_rate'],
                old_random_mean_changed_label_rate=old_change,
                relative_reduction_in_changed_predictions=1-cell['changed_label_rate']/old_change,
                accuracy_change_from_matched=cell['accuracy_change'],
                candidate_rule_passes=cell['changed_label_rate']<=.5*old_change and abs(cell['accuracy_change'])<=.02))
    passes={a:all(c['candidate_rule_passes'] for c in comparisons if c['arm']==a) for a in ARMS[1:]}
    decision=('scalar_margin_matching_retains_candidate_behavior' if passes['margin_matched'] else
              'eight_score_matching_retains_candidate_behavior' if passes['posterior_matched'] else
              'planned_uncertainty_matching_does_not_meet_candidate_rule')
    return dict(questions=1000,decision=decision,by_arm_receiver=cells,comparisons=comparisons,
                arm_passes_both_receivers=passes,matching_quality=summary['matching_quality'],
                limitations=[
                    'Development follow-up motivated by post-result uncertainty strata, not independent confirmation.',
                    'Original class-only sufficiency hypothesis still failed; this is a different intervention.',
                    'Matching is approximate, especially for rare score patterns; no semantic-information or confidence-only proof.',
                    'Softmax scores are not calibrated probabilities. Nearest donors may also share unmeasured properties.',
                    '50% fewer changes and at most2pp accuracy difference are fixed operational criteria, not population equivalence.',
                    'Question-bootstrap intervals are descriptive conditional on these donor plans; donor dependence is omitted.',
                    'A passing intervention retains full donor packets, so it is not a demonstrated scalar/8-score codec or bandwidth reduction.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    root=Path(__file__).resolve().parent
    proto_path=root/'remote/dream_label_confidence_protocol_v1.json';protocol=json.loads(proto_path.read_text())
    summary=json.loads((a.run/'summary.json').read_text())
    assert summary['protocol_sha256']==hashlib.sha256(proto_path.read_bytes()).hexdigest()
    source=root/'results/dream_readout_full_v3_12454951/per_item.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['baseline_per_item_sha256']
    rows=[json.loads(l) for l in (a.run/'per_item.jsonl').read_text().splitlines()]
    baseline=[json.loads(l) for l in source.read_text().splitlines()]
    validate_provenance(rows,summary,protocol,baseline)
    prior_path=root/'results/dream_label_ablation_v1.analysis.json'
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest()==protocol['prior_analysis_sha256']
    report=analyze(rows,summary,json.loads(prior_path.read_text()))
    report['source_sha256']={n:hashlib.sha256((a.run/n).read_bytes()).hexdigest() for n in ('summary.json','per_item.jsonl')}
    report['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
