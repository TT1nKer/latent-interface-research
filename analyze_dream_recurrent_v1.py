"""Question-paired analysis for frozen QASC recurrence; standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

# Regression contrast for an even-depth indicator with intercept + linear depth.
# Plain even-minus-odd would confound parity with increasing depth.
PARITY_WEIGHTS = (-5/24,11/24,-8/24,8/24,-11/24,5/24)


def argmax(xs):
    return max(range(len(xs)),key=xs.__getitem__)


def distribution(xs):
    ordered=sorted(xs)
    return dict(mean=statistics.mean(xs),median=statistics.median(xs),
                p10=ordered[int(.10*(len(ordered)-1))],p90=ordered[int(.90*(len(ordered)-1))])


def per_question(row):
    agreement=[int(a['predictions'][0]==b['predictions'][0]) for a,b in zip(row['cross'],row['self'],strict=True)]
    result={'linear_depth_adjusted_parity':sum(w*x for w,x in zip(PARITY_WEIGHTS,agreement)),
            'raw_even_minus_odd_agreement':statistics.mean(agreement[1::2])-statistics.mean(agreement[::2])}
    for i in range(5):
        result[f'agreement_change_{i+1}_to_{i+2}']=agreement[i+1]-agreement[i]
    for i in range(1,5):
        value=agreement[i]-(agreement[i-1]+agreement[i+1])/2
        result[f'alternating_interior_contrast_round{i+1}']=value if i%2 else -value
    return result


def analyze(rows,summary,resamples=2000):
    assert rows and len({r['example_id'] for r in rows})==len(rows)
    assert summary['completed'] and summary['examples']==len(rows)
    for row in rows:
        assert len(row['cross'])==len(row['self'])==6
        assert row['gold'] in range(8)
        for state in [row['native'],*row['cross'],*row['self']]:
            assert len(state['predictions'])==len(state['label_logits'])==len(state['message_logits'])==2
            for prediction,logits in zip(state['predictions'],state['label_logits'],strict=True):
                assert len(logits)==8 and all(math.isfinite(x) for x in logits)
                assert prediction==argmax(logits)
    curves=[]
    for d in range(6):
        curves.append(dict(round=d+1,
            cross_accuracy=statistics.mean(int(r['cross'][d]['predictions'][0]==r['gold']) for r in rows),
            self_accuracy=statistics.mean(int(r['self'][d]['predictions'][0]==r['gold']) for r in rows),
            cross_self_agreement=statistics.mean(int(r['cross'][d]['predictions'][0]==r['self'][d]['predictions'][0]) for r in rows)))
    q=[per_question(r) for r in rows]
    estimates={key:statistics.mean(v[key] for v in q) for key in q[0]}
    boot={key:[] for key in estimates}
    rng=random.Random(20261107)
    for _ in range(resamples):
        selected=rng.choices(q,k=len(q))
        for key in estimates:
            boot[key].append(sum(x[key] for x in selected)/len(selected))
    ci={}
    for key,samples in boot.items():
        samples.sort()
        ci[key]=[samples[int(.025*(resamples-1))],samples[int(.975*(resamples-1))]]
    common=[]
    for source in (0,1):
        cells=[r['common_input'][source] for r in rows]
        assert all(c['source']==source for c in cells)
        common.append(dict(source=source,
            relative_message_rms=distribution([c['relative_message_rms'] for c in cells]),
            message_classifier_tv=distribution([c['message_classifier_tv'] for c in cells]),
            common_receiver_tv=distribution([c['common_receiver_tv'] for c in cells]),
            message_label_context_agreement=statistics.mean(c['outgoing_message_labels'][0]==c['outgoing_message_labels'][1] for c in cells),
            common_receiver_label_context_agreement=statistics.mean(argmax(c['common_receiver_logits'][0])==argmax(c['common_receiver_logits'][1]) for c in cells),
            incoming_class_retention_by_context=[statistics.mean(c['outgoing_message_labels'][i]==c['incoming_message_label'] for c in cells) for i in (0,1)]))
    replication=summary['numerical_validity_passed'] and summary['public_reference_all_equal']
    agreement=[r['cross_self_agreement'] for r in curves]
    repeated_peaks=(agreement[1]>agreement[0] and agreement[1]>agreement[2]
                    and agreement[3]>agreement[2] and agreement[3]>agreement[4]
                    and agreement[5]>agreement[4])
    if len(rows)!=1000 or summary['stage']!='full':
        decision='validity_sample_only_no_mechanism_decision'
    elif not replication:
        decision='resolve_numerical_or_reproduction_difference_before_mechanism_claim'
    elif repeated_peaks and ci['linear_depth_adjusted_parity'][0]>0:
        decision='development_parity_pattern_observed_inspect_common_input_mechanism_and_confirm_elsewhere'
    else:
        decision='no_predefined_repeated_parity_pattern_do_not_expand_parity_branch_automatically'
    lost=[r for r in rows if r['cross'][0]['predictions'][0]==r['gold'] and r['cross'][3]['predictions'][0]!=r['gold']]
    return dict(questions=len(rows),rounds=list(range(1,7)),depth_curves=curves,
        contrasts=estimates,question_bootstrap_descriptive_ci95=ci,bootstrap_resamples=resamples,
        common_input_by_source=common,repeated_even_peaks_in_mean_curve=repeated_peaks,
        observed_one_to_four_losses=len(lost),
        of_these_agree_with_self_four=sum(r['cross'][3]['predictions'][0]==r['self'][3]['predictions'][0] for r in lost),
        decision=decision,
        limitations=['One released checkpoint and previously inspected development set; not held-out confirmation or scaling evidence.',
          'All depth/source conditions on one question are correlated; bootstrap unit is question.',
          'Linear detrending removes a linear depth trend only; report full curves and direction reversals.',
          'Common-input initial messages probe two points per question; cannot prove global equality or stability bounds.',
          'A pattern of even-round resemblance does not by itself establish the causal explanation for errors.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert not args.out.exists(),'Do not overwrite prior analysis'
    raw=(args.run/'per_item.jsonl').read_bytes()
    rows=[json.loads(x) for x in raw.decode().splitlines()]
    summary=json.loads((args.run/'summary.json').read_text())
    report=analyze(rows,summary)
    report.update(per_item_sha256=hashlib.sha256(raw).hexdigest(),
                  analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=summary)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('questions','depth_curves','decision')},indent=2))


if __name__=='__main__':
    main()
