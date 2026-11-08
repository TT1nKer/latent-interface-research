"""Paired question-level analysis for the preregistered development experiment.

Uses only the standard library. Rotations are correlated observations, so all
contrasts and descriptive bootstrap intervals resample questions, not rows.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

MODES = ('native', 'late', 'early')


def validate_records(rows):
    groups = {}
    for row in rows:
        assert row['shift'] in range(4)
        group = groups.setdefault(row['id'], {})
        assert row['shift'] not in group, 'Duplicate item/rotation'
        group[row['shift']] = row
        assert row['order'] == [(i+row['shift']) % 4 for i in range(4)]
        assert row['order'][row['correct']] == row['original_correct']
        for mode in MODES:
            scores = row['scores'][mode]
            assert scores['pred'] in range(4)
            assert all(math.isfinite(v) for v in scores['probs'])
            assert abs(sum(scores['probs']) - 1) < 1e-4
            assert scores['pred'] == max(range(4), key=lambda i: scores['probs'][i])
    for key, group in groups.items():
        assert set(group) == set(range(4)), f'Incomplete four-rotation item: {key}'
        assert len({r['prefix_sha256'] for r in group.values()}) == 1
        assert len({r['original_correct'] for r in group.values()}) == 1
        assert len({json.dumps(r['original'], sort_keys=True) for r in group.values()}) == 1
    assert groups
    return list(groups.values())


def correct(row, mode):
    return int(row['scores'][mode]['pred'] == row['correct'])


def per_question(group):
    identity = {m: correct(group[0], m) for m in MODES}
    changed = {m: statistics.mean(correct(group[s], m) for s in (1,2,3)) for m in MODES}
    original = {m:int(group[0]['original'][m]['pred'] == group[0]['original_correct'])
                for m in ('native','early')}
    return dict(
        native_changed=changed['native'], late_changed=changed['late'], early_changed=changed['early'],
        native_identity=identity['native'], late_identity=identity['late'], early_identity=identity['early'],
        native_original=original['native'], early_original=original['early'],
        late_native_did=(changed['late']-changed['native'])-(identity['late']-identity['native']),
        early_late_did=(changed['early']-changed['late'])-(identity['early']-identity['late']),
        late_native_stale_label_gap=statistics.mean(
            int(group[s]['scores']['late']['pred'] == group[s]['original_correct'])
            - int(group[s]['scores']['native']['pred'] == group[s]['original_correct']) for s in (1,2,3)))


def mean(values):
    return sum(values)/len(values) if values else None


def analyze(rows, *, resamples=2000):
    groups = validate_records(rows)
    q = [per_question(g) for g in groups]
    estimates = {k:mean([v[k] for v in q]) for k in q[0]}
    rng = random.Random(20261107)
    contrasts = ('late_native_did','early_late_did','late_native_stale_label_gap')
    boot = {k:[] for k in contrasts}
    for _ in range(resamples):
        sampled = rng.choices(q, k=len(q))
        for k in contrasts:
            boot[k].append(mean([v[k] for v in sampled]))
    intervals = {}
    for k, v in boot.items():
        v.sort()
        intervals[k] = [v[int(.025*(len(v)-1))], v[int(.975*(len(v)-1))]]
    eligible = [g[s] for g in groups for s in (1,2,3)
                if correct(g[s], 'native') and all(correct(g[0], m) for m in MODES)]
    conditional = dict(n=len(eligible), questions=len({r['id'] for r in eligible}),
        late_accuracy=mean([correct(r,'late') for r in eligible]),
        early_accuracy=mean([correct(r,'early') for r in eligible]))
    masses = {m:dict(mean=mean([r['scores'][m]['option_mass'] for r in rows]),
                    min=min(r['scores'][m]['option_mass'] for r in rows),
                    top_text_counts=dict(Counter(r['scores'][m]['top_text'] for r in rows))) for m in MODES}
    # This gate governs only whether to spend more on this candidate. It cannot
    # certify novelty/significance, and no resampling fixes the development sample.
    if estimates['native_changed'] < .60:
        decision = 'native_limited_do_not_attribute_failure_to_interface'
    elif estimates['early_original'] < estimates['native_original']-.15:
        decision = 'original_interface_degradation_resolve_specific_validity_issue_before_interpretation'
    elif estimates['late_native_did'] <= -.10 and estimates['early_late_did'] >= .10:
        decision = 'candidate_signal_requires_fresh_confirmation_and_mechanism_controls'
    else:
        decision = 'no_predefined_candidate_signal_do_not_automatically_scale_this_branch'
    return dict(questions=len(groups), correlated_rotation_rows=len(rows), estimates=estimates,
        question_bootstrap_descriptive_ci95=intervals, bootstrap_resamples=resamples,
        bootstrap_seed=20261107, conditional_native_and_identity_correct=conditional,
        answer_format_diagnostics=masses, development_decision=decision,
        effect_units='proportions; multiply by100 for percentage points',
        interpretation={'late_native_did':'Negative means a greater change-vs-identity loss for late fusion than for native.',
            'early_late_did':'Positive means early fusion improves changed encodings more than identity relative to late fusion.',
            'late_native_stale_label_gap':'Positive means more old-answer letter selection after recoding under late fusion than native.'},
        limitations=['No publication, novelty or scaling conclusion from this development sample.',
            'Bootstrap intervals describe sampled questions only; selection from first100 validation rows is not unbiased benchmark sampling.',
            'Conditioning on correct native/identity predictions is a diagnostic, not the primary causal estimand.',
            'This test uses query-conditioned shared input and cannot establish private-information transfer.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('predictions', type=Path)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads(args.metadata.read_text())
    assert meta.get('completed'), 'Do not analyze an incomplete run as complete evidence'
    assert len(meta['checks']) == 2
    assert all(c['chunked_max_probability_error'] < .02 and c['chunked_prediction_matches']
               and c['repeated_fusion_max_probability_error'] < 1e-6 for c in meta['checks'])
    raw = args.predictions.read_bytes()
    rows = [json.loads(r) for r in raw.decode().splitlines()]
    report = analyze(rows)
    assert len(rows) == meta['summary']['all']['n']
    report.update(job_id=meta['job_id'], predictions_sha256=hashlib.sha256(raw).hexdigest(),
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert not args.out.exists(), 'Refusing to overwrite a prior analysis'
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('questions','estimates','development_decision')}, indent=2))


if __name__ == '__main__':
    main()
