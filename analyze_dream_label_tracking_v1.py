"""Descriptive decoded-label tracking on already validated trajectories; stdlib."""
import argparse
import hashlib
import json
from pathlib import Path

from remote.dream_label_plan_v1 import argmax


SUMMARY_SHA = '9f43bb1d8ddfb0ec580fdca690bdee8ae0f81e76406446dce99d938b7f8cb9e5'
ITEM_SHA = 'a8a411cad532fdedd2b3cf01785b273928b7f8d725411df6779c22abe8e4fc85'


def tracking(rows):
    result = []
    for receiver in (0, 1):
        sender = 1-receiver
        native_sender = [argmax(r['native']['message_logits'][sender]) for r in rows]
        routes = {}
        for route in ('cross', 'self'):
            depths = []
            for d in range(6):
                incoming_agent = sender if route == 'cross' else receiver
                incoming = [argmax((r['native'] if d == 0 else r[route][d-1])['message_logits'][incoming_agent])
                            for r in rows]
                predictions = [r[route][d]['predictions'][receiver] for r in rows]
                outgoing = [argmax(r[route][d]['message_logits'][receiver]) for r in rows]
                seed_agent = sender if route == 'cross' and d % 2 == 0 else receiver
                seed = [argmax(r['native']['message_logits'][seed_agent]) for r in rows]
                depths.append(dict(depth=d+1,
                    readout_equals_incoming_class=sum(x == y for x, y in zip(predictions, incoming)),
                    outgoing_retains_incoming_class=sum(x == y for x, y in zip(outgoing, incoming)),
                    readout_equals_initial_ancestral_class=sum(x == y for x, y in zip(predictions, seed)),
                    outgoing_equals_initial_ancestral_class=sum(x == y for x, y in zip(outgoing, seed)),
                    readout_correct=sum(p == r['gold'] for p, r in zip(predictions, rows)),
                    initial_ancestral_class_correct=sum(p == r['gold'] for p, r in zip(seed, rows))))
            routes[route] = depths
        result.append(dict(receiver=receiver, sender=sender,
            native_sender_class_counts={str(c): native_sender.count(c) for c in range(8)},
            native_sender_class_correct=sum(p == r['gold'] for p, r in zip(native_sender, rows)),
            routes=routes))
    return dict(n=len(rows), receivers=result, interpretation='Exploratory decoded-label associations, not a causal or semantic sufficiency test.',
        limitations=[
            'Every depth was already inspected on this development set; this is not independent confirmation.',
            'Classifier argmax hides confidence and within-class information; agreement does not prove the receiver uses only the class.',
            'Initial ancestry alternation is a structural identity of memoryless cross updates; empirical label retention is an additional observation, not proof of convergence.',
            'No new model evaluation, training, donor intervention, or compute-efficiency result.'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    hashes = {n: hashlib.sha256((a.run/n).read_bytes()).hexdigest() for n in ('summary.json', 'per_item.jsonl')}
    assert hashes == {'summary.json': SUMMARY_SHA, 'per_item.jsonl': ITEM_SHA}
    summary = json.loads((a.run/'summary.json').read_text())
    assert summary['completed'] and summary['numerical_validity_passed'] is True
    rows = [json.loads(line) for line in (a.run/'per_item.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['example_id'] for r in rows}) == 1000
    result = tracking(rows)
    result['source_sha256'] = hashes
    result['analysis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
