"""Post-result, exploratory uncertainty strata; no new model evaluation."""
import json
import hashlib
from pathlib import Path


def analyze(rows, baseline):
    assert [r['example_id'] for r in rows] == [r['example_id'] for r in baseline]
    output = []
    arms = ['matched', 'within_class_0', 'within_class_1', 'label_prefix']
    for receiver in (0, 1):
        def counts(indices):
            return {a: dict(correct=sum(rows[i]['conditions'][a][receiver]['prediction'] == rows[i]['gold'] for i in indices),
                            changed=sum(rows[i]['conditions'][a][receiver]['prediction'] != rows[i]['conditions']['matched'][receiver]['prediction'] for i in indices)) for a in arms}
        margins = []
        for row in baseline:
            logits = sorted(row['native']['message_logits'][1-receiver], reverse=True)
            margins.append(logits[0]-logits[1])
        ordered = sorted(range(len(rows)), key=lambda i: (margins[i], i))
        quintiles = []
        for q in range(5):
            indices = ordered[q*200:(q+1)*200]
            quintiles.append(dict(quintile=q, n=200, margin_min=min(margins[i] for i in indices),
                                  margin_max=max(margins[i] for i in indices), counts=counts(indices)))
        accuracy_strata = []
        for correct in (False, True):
            indices = [i for i, r in enumerate(rows) if (r['conditions']['matched'][receiver]['incoming_class'] == r['gold']) == correct]
            accuracy_strata.append(dict(sender_class_correct=correct, n=len(indices), counts=counts(indices)))
        output.append(dict(receiver=receiver, margin_quintiles=quintiles, sender_class_correctness=accuracy_strata))
    return dict(status='Exploratory strata chosen after the intervention results, not confirmatory.',
                n=len(rows), receivers=output,
                limits=['Raw top-two logit margin is an uncertainty proxy, not calibrated confidence.',
                        'Stratum membership uses original frozen sender scores; gold used only for descriptive correctness grouping.',
                        'Concentration does not prove uncertainty causes the changes. A new matched intervention is needed.'])


if __name__ == '__main__':
    paths=[Path('results/dream_label_ablation_v1_12457578/per_item.jsonl'),
           Path('results/dream_readout_full_v3_12454951/per_item.jsonl')]
    expected=['45ac6fa8ff991ea2321f1eca72890232fc890a9e107580fb94c6f01ac6aa55a4',
              'a8a411cad532fdedd2b3cf01785b273928b7f8d725411df6779c22abe8e4fc85']
    for p,h in zip(paths,expected): assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    result=analyze(*[[json.loads(s) for s in p.read_text().splitlines()] for p in paths])
    result['source_sha256']=dict(zip(['intervention_per_item','baseline_per_item'],expected))
    out=Path('results/dream_label_uncertainty_v1.json');assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
