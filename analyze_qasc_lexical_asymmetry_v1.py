"""Exploratory public-endpoint diagnostic; stdlib only, no model execution.

Question: is the one-to-four-round change concentrated where the correct option
is literally present in one agent's private evidence? A label-blind baseline
selects an option only when exactly one option occurs as a contiguous token
sequence. Ambiguous/absent matches abstain. All contexts include distractors.
This is not semantic sufficiency, a causal intervention, or held-out evidence.
"""
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'external/hidden_messages/data/development.jsonl'
PRED = ROOT / 'external/hidden_messages/qasc_predictions.jsonl'
LABELS = 'ABCDEFGH'


def tokens(text):
    return tuple(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def contains(sequence, phrase):
    return bool(phrase) and any(sequence[i:i+len(phrase)] == phrase
                               for i in range(len(sequence)-len(phrase)+1))


def lexical_matches(options, context):
    sequence = tokens(context)
    return [label for label, text in options if contains(sequence, tokens(text))]


def entropy(counts):
    n = sum(counts.values())
    return -sum((k/n)*math.log2(k/n) for k in counts.values()) if n else None


def paired_summary(rows):
    n = len(rows)
    losses = sum(r['correct']['final_only'] and not r['correct']['matched'] for r in rows)
    gains = sum(not r['correct']['final_only'] and r['correct']['matched'] for r in rows)
    return dict(n=n, correct={c:sum(r['correct'][c] for r in rows)
                             for c in ('no_message','final_only','matched','self')},
                losses=losses, gains=gains, net_four_minus_one=gains-losses,
                net_fraction=(gains-losses)/n if n else None,
                loss_rate_among_one_correct=(losses/sum(r['correct']['final_only'] for r in rows)
                    if any(r['correct']['final_only'] for r in rows) else None),
                four_self_agreement=sum(r['labels']['matched']==r['labels']['self'] for r in rows))


def baseline_summary(rows, agent):
    answered = [r for r in rows if r['lexical_predictions'][agent] is not None]
    correct = sum(r['lexical_predictions'][agent] == r['gold'] for r in answered)
    agreement = {c:sum(r['lexical_predictions'][agent] == r['labels'][c] for r in answered)
                 for c in ('no_message','final_only','matched','self')}
    return dict(n=len(rows), answered=len(answered), correct=correct,
                coverage=len(answered)/len(rows),
                selective_accuracy=correct/len(answered) if answered else None,
                accuracy_counting_abstentions_wrong=correct/len(rows),
                published_endpoint_agreement_on_answered=agreement,
                ambiguous=sum(len(r['lexical_matches'][agent])>1 for r in rows),
                no_match=sum(not r['lexical_matches'][agent] for r in rows))


def standardized_comparison(rows, resamples=2000):
    """Descriptive stratified question bootstrap; no causal interpretation."""
    groups = [[r for r in rows if r['gold_present']==pair]
              for pair in ([False,True],[True,False])]
    cells = [[list(int(r['correct']['matched'])-int(r['correct']['final_only'])
                   for r in group if r['gold']==label) for label in LABELS]
             for group in groups]
    assert all(cell for group in cells for cell in group)
    effects = [sum(sum(cell)/len(cell) for cell in group)/8 for group in cells]
    rng = random.Random(20261108)
    samples = []
    for _ in range(resamples):
        draws = [sum(sum(rng.choices(cell,k=len(cell)))/len(cell) for cell in group)/8
                 for group in cells]
        samples.append(draws[0]-draws[1])
    samples.sort()
    return dict(estimand='Equal-weight A-H mean of paired four-minus-one accuracy; sender-only minus receiver-only',
        sender_only_standardized_delta=effects[0],receiver_only_standardized_delta=effects[1],
        standardized_difference=effects[0]-effects[1],
        descriptive_bootstrap_ci95=[samples[int(.025*(resamples-1))],samples[int(.975*(resamples-1))]],
        resamples=resamples,seed=20261108,
        bootstrap_unit='Unique source question, independently within each literal-presence/gold-label cell',
        native_correctness_strata={
            name:{str(native):paired_summary([r for r in group if r['correct']['no_message']==native])
                  for native in (False,True)}
            for name,group in zip(('sender_only','receiver_only'),groups)})


def main():
    raw = [json.loads(line) for line in DATA.read_text().splitlines()]
    predictions = [json.loads(line) for line in PRED.read_text().splitlines()]
    lookup = {r['example_id']:r for r in predictions}
    assert len(raw)==len(predictions)==len(lookup)==1000
    assert len({r['source_example_id'] for r in raw})==1000
    assert {r['example_id'] for r in raw}==set(lookup)
    rows = []
    for r in raw:
        p = lookup[r['example_id']]
        assert r['answer_label']==p['answer_label']
        assert r['source_example_id']==p['source_task_id']
        assert len(r['private_contexts'])==2
        assert [x[0] for x in r['answer_options']]==list(LABELS)
        matches = [lexical_matches(r['answer_options'], context) for context in r['private_contexts']]
        gold = r['answer_label']
        labels = {c:v['label'] for c,v in p['conditions'].items()}
        assert all(v['correct']==(v['label']==gold) for v in p['conditions'].values())
        rows.append(dict(example_id=r['example_id'],gold=gold,labels=labels,
            correct={c:v==gold for c,v in labels.items()},lexical_matches=matches,
            lexical_predictions=[m[0] if len(m)==1 else None for m in matches],
            gold_present=[gold in m for m in matches]))

    groups = {}
    for present0 in (False,True):
        for present1 in (False,True):
            key=f'agent0_{int(present0)}_agent1_{int(present1)}'
            selected = [r for r in rows if r['gold_present']==[present0,present1]]
            groups[key] = paired_summary(selected)
    by_label = {label:paired_summary([r for r in rows if r['gold']==label]) for label in LABELS}
    distributions = {}
    for condition in predictions[0]['conditions']:
        counts = Counter(r['labels'][condition] for r in rows)
        errors = Counter(r['labels'][condition] for r in rows if not r['correct'][condition])
        distributions[condition] = dict(counts={x:counts[x] for x in LABELS},
            entropy_bits=entropy(counts),error_counts={x:errors[x] for x in LABELS},
            error_entropy_bits=entropy(errors))

    report = dict(role='Exploratory secondary analysis of author outputs; no new neural run',
        question='Whether literal answer availability is asymmetric and associated with one-to-four loss',
        predictions_sha256=hashlib.sha256(PRED.read_bytes()).hexdigest(),
        development_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline='NFKC + casefold + Unicode word tokens; contiguous exact option phrase; select only a unique matching option; otherwise abstain. Selection never accesses gold labels. Uses complete private contexts including distractors; no semantic equivalence or synonym matching.',
        all=paired_summary(rows),baselines={f'agent{i}':baseline_summary(rows,i) for i in (0,1)},
        gold_literal_presence_groups=groups,by_gold_label=by_label,
        standardized_comparison=standardized_comparison(rows),
        gold_distribution=dict(Counter(r['gold'] for r in rows)),
        prediction_distributions=distributions,
        limitations=[
            'Public development data and previously observed outcomes; exploratory, not preregistered or confirmatory.',
            'Literal answer presence does not establish that a context suffices to answer; absence does not imply missing semantic information.',
            'Selective baseline accuracy must be reported with coverage and abstention-as-error accuracy.',
            'Prediction entropy cannot exclude item-specific attractors or sensitivity to option reordering.',
            'Natural context groups are observational, unbalanced, and differ in difficulty; do not infer causality from subgroup differences.',
            'No context swap, option permutation, neural trajectory, or new model run is performed here.'
        ])
    out = ROOT/'results/qasc_lexical_asymmetry_v1.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('baselines','gold_literal_presence_groups','standardized_comparison')},indent=2))


if __name__=='__main__':
    main()
