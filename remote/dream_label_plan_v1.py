"""Fixed donor plans using frozen predicted message labels, never gold answers."""
import hashlib
import json


ARMS = ('matched', 'within_class_0', 'within_class_1', 'different_class', 'label_prefix')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def rank(ids, indices, salt):
    return sorted(indices, key=lambda i: hashlib.sha256(f'{salt}\0{ids[i]}'.encode()).digest())


def donor_indices(ids, labels, seed, *, same_class):
    assert len(ids) == len(labels) and len(set(ids)) == len(ids)
    assert labels and all(x in range(8) for x in labels)
    groups = {c: rank(ids, [i for i, y in enumerate(labels) if y == c],
                      f'label-ablation-v1-{seed}-{c}') for c in sorted(set(labels))}
    result = [-1] * len(ids)
    if same_class:
        assert min(map(len, groups.values())) >= 2, 'No within-class derangement for singleton'
        for group in groups.values():
            for a, b in zip(group, group[1:] + group[:1]):
                result[a] = b
    else:
        largest = max(map(len, groups.values()))
        assert 2 * largest <= len(ids), 'No different-class bijection for majority class'
        ordered = [i for group in groups.values() for i in group]
        rotated = ordered[largest:] + ordered[:largest]
        for a, b in zip(ordered, rotated):
            result[a] = b
    assert sorted(result) == list(range(len(ids)))
    assert all(i != j and ((labels[i] == labels[j]) == same_class)
               for i, j in enumerate(result))
    return result


def build_plan(ids, message_logits):
    assert len(ids) == len(message_logits)
    result = {'version': 1, 'example_ids': list(ids), 'receivers': []}
    for receiver in (0, 1):
        labels = [argmax(pair[1-receiver]) for pair in message_logits]
        result['receivers'].append({
            'receiver': receiver, 'sender': 1-receiver, 'sender_predicted_labels': labels,
            'class_counts': {str(c): labels.count(c) for c in range(8)},
            'donors': {
                'matched': list(range(len(ids))),
                'within_class_0': donor_indices(ids, labels, 0, same_class=True),
                'within_class_1': donor_indices(ids, labels, 1, same_class=True),
                'different_class': donor_indices(ids, labels, 0, same_class=False),
            }})
    return result


def resource_projection(baseline):
    assert baseline['completed'] and baseline['numerical_validity_passed'] is True
    assert baseline['examples'] == 1000 and len(baseline['terminal_seconds']) == 26
    estimate = 1.5 * (baseline['setup_seconds'] + 10 * max(baseline['terminal_seconds'])) + 180
    assert estimate <= 1200, f'{estimate:.1f}s exceeds fixed 20-minute cap'
    assert baseline['torch_peak_reserved_bytes'] < 22 * 1024**3
    return estimate
