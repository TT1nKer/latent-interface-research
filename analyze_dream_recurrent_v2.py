"""Amended interpretation and sensitivity checks; original v1 statistics intact."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from analyze_dream_recurrent_v1 import analyze, per_question, context_ownership_analysis
from remote.dream_trace_gate_v2 import FINGERPRINT_KEYS, implementation_checks_pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(path):
    return ([json.loads(line) for line in (path/'per_item.jsonl').read_text().splitlines()],
            json.loads((path/'summary.json').read_text()))


def primary_met(report):
    return (report['repeated_even_peaks_in_mean_curve'] and
            report['question_bootstrap_descriptive_ci95']['linear_depth_adjusted_parity'][0] > 0)


def primary_extract(report):
    return {key: report[key] for key in ('questions', 'depth_curves', 'contrasts',
            'question_bootstrap_descriptive_ci95', 'repeated_even_peaks_in_mean_curve')}


def reference_sensitivity(rows, summary, resamples=2000):
    ids = {r['example_id'] for r in rows}
    flagged = set().union(*(set(c['mismatch_ids']) for c in summary['reference_agreement'].values()))
    assert flagged <= ids
    retained = [r for r in rows if r['example_id'] not in flagged]
    n, k = len(rows), len(flagged)
    fixed_sum = sum(per_question(r)['linear_depth_adjusted_parity'] for r in retained)
    # The six fixed weights have positive sum1 and negative sum-1, hence each
    # binary-agreement score lies in[-1,1]. This interval is not a confidence CI.
    worst_case = [(fixed_sum-k)/n, (fixed_sum+k)/n]
    trimmed = None
    if retained:
        subset_summary = dict(summary, examples=len(retained))
        trimmed = primary_extract(analyze(retained, subset_summary, resamples=resamples))
    return dict(flagged_examples=k, flagged_ids=sorted(flagged), retained_examples=len(retained),
                retained_subset=trimmed, arbitrary_flagged_score_mean_interval=worst_case,
                interval_definition='Keep allN in denominator, hold unflagged scores fixed, allow each flagged score anywhere in[-1,1]. Not a probability or confidence interval, and not a bound on unknown unflagged errors.',
                limitation='Post-outcome exclusion diagnostic; primary result retains every question. It does not identify the mismatch cause or remove selection bias.')


def paired_probe(full_rows, full_summary, probe_rows, probe_summary, full_tokens,
                 probe_tokens, amendment, resamples=2000):
    assert probe_summary['completed'] and probe_summary['stage'] == 'sensitivity'
    assert probe_summary['execution_policy'] == 'within_environment_v2'
    for key in FINGERPRINT_KEYS:
        assert full_summary[key] == probe_summary[key], key
    assert probe_summary['original_numerical_flags'] == full_summary['numerical_flags']
    expected_flags = dict(full_summary['numerical_flags'], bf16_reduced_precision_reduction=False)
    assert full_summary['numerical_flags']['bf16_reduced_precision_reduction'] is True
    assert probe_summary['numerical_flags'] == expected_flags
    plan = amendment['sensitivity']
    expected_rows = [r for start in plan['block_starts']
                     for r in full_rows[start:start+plan['block_size']]]
    assert [r['example_id'] for r in probe_rows] == [r['example_id'] for r in expected_rows]
    assert len(probe_rows) == plan['examples'] == probe_summary['examples']
    assert len(probe_tokens) == len(probe_rows)
    for row in probe_rows:
        key = row['example_id']
        assert full_tokens[key] == probe_tokens[key], f'Token rows differ: {key}'
    n = len(probe_rows)
    baseline_report = analyze(expected_rows, dict(full_summary, examples=n), resamples=resamples)
    probe_report = analyze(probe_rows, probe_summary, resamples=resamples)
    deltas = [per_question(b)['linear_depth_adjusted_parity'] -
              per_question(a)['linear_depth_adjusted_parity']
              for a, b in zip(expected_rows, probe_rows, strict=True)]
    rng = random.Random(20261110)
    samples = sorted(statistics.mean(rng.choices(deltas, k=n)) for _ in range(resamples))
    comparison = []
    for route, depths in [('native', [0]), ('cross', range(1, 7)), ('self', range(1, 7))]:
        for depth in depths:
            get = lambda r: r['native'] if route == 'native' else r[route][depth-1]
            for agent in (0, 1):
                changed, max_label_delta, max_message_delta = [], 0.0, 0.0
                for a, b in zip(expected_rows, probe_rows, strict=True):
                    sa, sb = get(a), get(b)
                    if sa['predictions'][agent] != sb['predictions'][agent]:
                        changed.append(a['example_id'])
                    max_label_delta = max(max_label_delta, max(abs(x-y) for x,y in zip(
                        sa['label_logits'][agent], sb['label_logits'][agent], strict=True)))
                    max_message_delta = max(max_message_delta, max(abs(x-y) for x,y in zip(
                        sa['message_logits'][agent], sb['message_logits'][agent], strict=True)))
                comparison.append(dict(route=route, depth=depth, agent=agent,
                    label_changes=len(changed), changed_ids=changed,
                    max_abs_label_logit_change=max_label_delta,
                    max_abs_message_classifier_logit_change=max_message_delta))
    sign = lambda x: (x > 0) - (x < 0)
    before, after = baseline_report['contrasts'], probe_report['contrasts']
    reversals = {key: sign(before[key]) != sign(after[key]) for key in
                 ('linear_depth_adjusted_parity', 'agreement_change_2_to_3', 'agreement_change_4_to_5')}
    return dict(examples=n, token_rows_equal=True,
        baseline=primary_extract(baseline_report), perturbation=primary_extract(probe_report),
        paired_primary_change=statistics.mean(deltas),
        paired_descriptive_ci95=[samples[int(.025*(resamples-1))], samples[int(.975*(resamples-1))]],
        direction_changes=reversals, fragile_on_tested_setting=any(reversals.values()),
        captured_predictions_and_logits_identical=all(c['label_changes']==0 and
            c['max_abs_label_logit_change']==0 and c['max_abs_message_classifier_logit_change']==0 for c in comparison),
        by_route_depth_agent=comparison,
        limitation='192 preselected, partly failure-targeted questions; one numerical setting only. Does not establish full-dataset robustness, FP32 equivalence, hardware causality, or independent replication. Equal outputs can mean the selected kernels did not use the disabled option.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    assert not args.out.exists(), 'Preserve earlier analyses'
    root=Path(__file__).resolve().parent
    amendment_path=root/'remote/dream_trace_amendment_v2.json'
    amendment=json.loads(amendment_path.read_text())
    rows, summary=load_run(args.baseline)
    assert summary['execution_policy']=='within_environment_v2'
    assert summary['stage']=='full' and summary['completed'] and len(rows)==1000
    assert summary['amendment_sha256']==sha(amendment_path)
    assert summary['numerical_validity_passed'] is True
    assert implementation_checks_pass(summary['independent_author_checks'])
    report=analyze(rows, summary)
    report['v1_strict_replication_decision']=report.pop('decision')
    report['analysis_policy']='within_environment_v2'
    report['reference_replication_complete']=summary['public_reference_all_equal']
    report['reference_agreement']=summary['reference_agreement']
    report['reference_sensitivity']=reference_sensitivity(rows,summary)
    development_path=root/'external/hidden_messages/data/development.jsonl'
    assert sha(development_path)==summary['data_sha256']
    development=[json.loads(line) for line in development_path.read_text().splitlines()]
    report['secondary_context_ownership']=context_ownership_analysis(rows,development)
    report['primary_pattern_criteria_met']=primary_met(report)
    report['numerical_probe']=None
    if args.probe:
        probe_rows, probe_summary=load_run(args.probe)
        assert probe_summary['prior_validity_receipt_sha256']==sha(args.baseline/'summary.json')
        def tokens(path):
            values=[json.loads(line) for line in (path/'token_rows.jsonl').read_text().splitlines()]
            result={r['example_id']:r['agent_token_ids'] for r in values}
            assert len(result)==len(values)
            return result
        report['numerical_probe']=paired_probe(rows,summary,probe_rows,probe_summary,
            tokens(args.baseline),tokens(args.probe),amendment)
    trimmed=report['reference_sensitivity']['retained_subset']
    if not report['primary_pattern_criteria_met']:
        decision='no_predefined_parity_pattern_in_baseline_do_not_expand_parity_branch'
    elif trimmed is None or not primary_met(trimmed):
        decision='primary_pattern_observed_but_reference_exclusion_sensitive_no_robust_mechanism_claim'
    elif report['numerical_probe'] is None:
        decision='within_environment_pattern_provisional_numerical_probe_pending'
    elif report['numerical_probe']['fragile_on_tested_setting']:
        decision='within_environment_pattern_numerically_sensitive_no_robust_mechanism_claim'
    else:
        decision='within_environment_candidate_survives_planned_diagnostics_needs_independent_confirmation'
    report['decision']=decision
    report['limitations'].extend([
        'Execution policy was explicitly amended after small-cohort reference mismatches; original core statistic and all-question inclusion unchanged.',
        'Within-environment evidence does not imply exact reproduction or a cross-hardware mechanism.',
        'Exclusion and finite numerical probes cannot prove immunity to numerical effects.'])
    report['evidence_sha256']=dict(analysis=sha(__file__),base_analysis=sha(root/'analyze_dream_recurrent_v1.py'),
        amendment=sha(amendment_path),per_item=sha(args.baseline/'per_item.jsonl'),summary=sha(args.baseline/'summary.json'))
    if args.probe:
        report['evidence_sha256'].update(probe_per_item=sha(args.probe/'per_item.jsonl'),
            probe_summary=sha(args.probe/'summary.json'))
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('questions','contrasts','decision')},indent=2))


if __name__=='__main__':
    main()
