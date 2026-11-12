"""Paired descriptive analysis of label-preserving substitutions; stdlib only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from remote.dream_label_plan_v1 import ARMS, build_plan, digest


def validate_provenance(rows, summary, protocol, baseline_rows):
    for key in ('baseline_summary_sha256', 'baseline_per_item_sha256', 'donor_plan_sha256'):
        assert summary[key] == protocol[key], key
    assert summary['code_sha256'] == protocol['implementation_sha256']['dream_label_ablation_v1.py']
    assert summary['helper_sha256'] == protocol['implementation_sha256']['dream_label_plan_v1.py']
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


def effect(rows, arm, receiver, resamples=2000):
    delta, changed, donor_follow, score_change = [], [], [], []
    for row in rows:
        a = row['conditions']['matched'][receiver]
        b = row['conditions'][arm][receiver]
        delta.append(int(b['prediction'] == row['gold']) - int(a['prediction'] == row['gold']))
        changed.append(int(a['prediction'] != b['prediction']))
        donor_follow.append(int(b['prediction'] == b['incoming_class']))
        score_change.append(max(abs(x-y) for x, y in zip(a['label_logits'], b['label_logits'], strict=True)))
    rng = random.Random(20261112 + receiver)
    samples = sorted(sum(rng.choices(delta, k=len(delta)))/len(delta) for _ in range(resamples))
    n = len(rows)
    ci = [samples[int(.025*(resamples-1))], samples[int(.975*(resamples-1))]]
    return {'arm': arm, 'receiver': receiver, 'n': n,
            'accuracy': statistics.mean(row['conditions'][arm][receiver]['prediction'] == row['gold'] for row in rows),
            'accuracy_change': statistics.mean(delta), 'changed_labels': sum(changed),
            'changed_label_rate': statistics.mean(changed), 'incoming_class_follow_rate': statistics.mean(donor_follow),
            'accuracy_change_descriptive_ci95': ci,
            'max_abs_label_logit_change': max(score_change),
            'operational_tolerance_met': abs(statistics.mean(delta)) <= .02 and statistics.mean(changed) <= .05}


def analyze(rows, summary, resamples=2000):
    assert summary['completed'] and summary['implementation_validity_passed'] is True
    assert len(rows) == summary['examples'] == 1000
    assert len({r['example_id'] for r in rows}) == len(rows)
    for row in rows:
        assert set(row['conditions']) == set(ARMS)
        for cells in row['conditions'].values():
            assert len(cells) == 2
            for cell in cells:
                assert len(cell['label_logits']) == 8
                assert all(math.isfinite(v) for v in cell['label_logits'])
                assert cell['prediction'] == max(range(8), key=cell['label_logits'].__getitem__)
    cells = [effect(rows, arm, receiver, resamples) for arm in ARMS for receiver in (0, 1)]
    same = [c for c in cells if c['arm'].startswith('within_class')]
    hard = [c for c in cells if c['arm'] == 'label_prefix']
    different = [c for c in cells if c['arm'] == 'different_class']
    live = all(c['changed_label_rate'] >= .20 for c in different)
    if not live:
        decision = 'different_class_positive_control_weak_no_bottleneck_inference'
    elif all(c['operational_tolerance_met'] for c in same + hard):
        decision = 'low_dimensional_label_relay_candidate_needs_heldout_confirmation'
    elif all(c['operational_tolerance_met'] for c in same):
        decision = 'within_class_exchange_stable_but_hard_label_prefix_insufficient'
    else:
        decision = 'within_class_message_details_matter_inspect_confidence_and_geometry_not_semantics_by_default'
    return {'questions': len(rows), 'decision': decision, 'by_arm_receiver': cells,
            'positive_control_live': live, 'operational_accuracy_margin': .02,
            'operational_changed_label_margin': .05, 'resamples': resamples,
            'limitations': [
                'Development intervention; operational tolerance is not a population equivalence test or paper-level confirmation.',
                'Bootstrap resamples target questions conditional on fixed donor banks/permutations; donor dependence and donor-plan uncertainty are not captured.',
                'Predicted labels come from the frozen message classifier, never gold labels. Label_prefix uses the existing training teacher-prefix dictionary.',
                'Hard-prefix failure may reflect confidence, calibration or projection geometry rather than rich semantic content.',
                'A class index requires at most three transmitted bits with a shared codebook, but sender inference and classifier costs remain; no computational speedup is established.',
                'Two within-class permutations are sensitivity checks, not independent replications. One checkpoint, task and depth.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    root = Path(__file__).resolve().parent
    protocol_path = root/'remote/dream_label_ablation_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    baseline_path = root/'results/dream_readout_full_v3_12454951/per_item.jsonl'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == protocol['baseline_per_item_sha256']
    summary = json.loads((args.run/'summary.json').read_text())
    assert summary['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (args.run/'per_item.jsonl').read_text().splitlines()]
    validate_provenance(rows, summary, protocol,
                        [json.loads(line) for line in baseline_path.read_text().splitlines()])
    report = analyze(rows, summary)
    report['source_sha256'] = {name: hashlib.sha256((args.run/name).read_bytes()).hexdigest()
                               for name in ('summary.json', 'per_item.jsonl')}
    report['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
