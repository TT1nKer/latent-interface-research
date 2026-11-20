"""Reduce frozen Karel rollouts to paired, equal-weight, exact-program estimates."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SIMPLE = (
    'suffix_live', 'suffix_initial', 'suffix_masked', 'modal_plan', 'index_vote',
    'dtw_uniform', 'dtw_actions_only', 'dtw_support_weighted',
)
SEEDS = (17, 29, 43)
RETRIEVAL = tuple(f'retrieval_{seed}' for seed in SEEDS)
CENTROID = tuple(f'centroid_{seed}' for seed in SEEDS)
SCRAMBLED = 'retrieval_scrambled_17'
METHODS = SIMPLE + RETRIEVAL + CENTROID + ('initial_17', SCRAMBLED)
RETRIEVAL_MEAN = len(METHODS)
CENTROID_MEAN = RETRIEVAL_MEAN + 1
LABELS = METHODS + ('retrieval_seed_average', 'centroid_seed_average')
DOMAINS = ('development_holdout', 'external')
METRICS = ('endpoint_success', 'crash', 'cap', 'retrieval_only', 'comparator_only')
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20261120


def load_rows(path):
    """Validate identities, paired predictions and complete six-world panels."""
    groups = {domain: defaultdict(dict) for domain in DOMAINS}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                code, record, target = row['code_id'], row['program_id'], row['target_id']
                if not isinstance(code, str) or not code or not isinstance(record, (str, int)) or isinstance(record, bool):
                    raise ValueError('invalid code or program id')
                if type(target) is not int or target not in range(6):
                    raise ValueError('invalid target id')
                panel = groups[row['domain']][record]
                if target in panel:
                    raise ValueError('duplicate program/target')
                if type(row['reference_action_count']) is not int or row['reference_action_count'] < 0:
                    raise ValueError('invalid reference length')
                if type(row['support_unique_traces']) is not int or not 1 <= row['support_unique_traces'] <= 5:
                    raise ValueError('invalid number of distinct traces among five supports')
                if set(row['methods']) != set(METHODS):
                    raise ValueError(f'wrong method set: {set(row["methods"]) ^ set(METHODS)}')
                values = np.zeros((len(LABELS), len(METRICS)))
                for i, name in enumerate(METHODS):
                    result = row['methods'][name]
                    success, termination = result['success'], result['termination']
                    if type(success) is not bool or termination not in ('stop', 'crash', 'cap'):
                        raise ValueError(f'invalid rollout outcome: {name}')
                    if success and termination != 'stop':
                        raise ValueError(f'success without STOP: {name}')
                    values[i, :3] = (success, termination == 'crash', termination == 'cap')
                for output, seeds in [(RETRIEVAL_MEAN, RETRIEVAL), (CENTROID_MEAN, CENTROID)]:
                    values[output] = values[[METHODS.index(name) for name in seeds]].mean(axis=0)
                rseed = values[[METHODS.index(name) for name in RETRIEVAL], 0]
                cseed = values[[METHODS.index(name) for name in CENTROID], 0]
                values[:len(SIMPLE), 3] = values[RETRIEVAL_MEAN, 0] * (1 - values[:len(SIMPLE), 0])
                values[:len(SIMPLE), 4] = (1 - values[RETRIEVAL_MEAN, 0]) * values[:len(SIMPLE), 0]
                values[CENTROID_MEAN, 3:] = (np.mean(rseed * (1 - cseed)), np.mean((1 - rseed) * cseed))
                j = METHODS.index(SCRAMBLED)
                values[j, 3:] = (rseed[0] * (1 - values[j, 0]), (1 - rseed[0]) * values[j, 0])
                panel[target] = (code, row['support_unique_traces'], row['reference_action_count'], values)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f'{path}:{line_no}: {e}') from e
    for domain, records in groups.items():
        if not records:
            raise ValueError(f'{domain}: expected a complete panel')
        for record, panel in records.items():
            if set(panel) != set(range(6)) or len({item[0] for item in panel.values()}) != 1:
                raise ValueError(f'{domain}/{record}: incomplete targets or inconsistent code ids')
    return groups


def reduce_programs(records, subset):
    by_code = defaultdict(list)
    n_targets = n_records = 0
    for panel in records.values():
        target_rows = [panel[i] for i in range(6)]
        if subset == 'variable_support':
            target_rows = [x for x in target_rows if x[1] >= 2]
        elif subset == 'zero_reference_actions':
            target_rows = [x for x in target_rows if x[2] == 0]
        elif subset != 'all':
            raise ValueError(f'unknown subset {subset}')
        if target_rows:
            by_code[target_rows[0][0]].append(np.mean([x[3] for x in target_rows], axis=0))
            n_targets += len(target_rows)
            n_records += 1
    counts = dict(targets=n_targets, program_records=n_records, distinct_codes=len(by_code))
    if not by_code:
        return counts, None
    # Sorting fixes the mapping between bootstrap indices and exact code identities.
    return counts, np.stack([np.mean(by_code[code], axis=0) for code in sorted(by_code)])


def bootstrap_means(program_success):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_codes, n_methods = program_success.shape
    out = np.empty((BOOTSTRAPS, n_methods))
    for first in range(0, BOOTSTRAPS, 64):
        count = min(64, BOOTSTRAPS - first)
        sampled = rng.integers(n_codes, size=(count, n_codes))
        out[first:first + count] = program_success[sampled].mean(axis=1)
    return out


def estimate(point, draws):
    return {'mean': float(point), 'percentile_95': np.quantile(draws, [0.025, 0.975]).tolist()}


def summarize(records, subset):
    counts, programs = reduce_programs(records, subset)
    if programs is None:
        return {'counts': counts, 'estimates': None}
    means = programs.mean(axis=0)
    scores = means[:, 0]
    draws = bootstrap_means(programs[:, :, 0])
    estimates = {}
    for i, method in enumerate(LABELS):
        estimates[method] = dict(endpoint_success=estimate(scores[i], draws[:, i]),
                                 crash=float(means[i, 1]), cap=float(means[i, 2]))
    # The winner may differ across bootstrap resamples. Keep comparisons paired by code.
    simple_scores = scores[:len(SIMPLE)]
    strongest_score = simple_scores.max()
    strongest = [method for method, value in zip(SIMPLE, simple_scores) if value == strongest_score]
    delta = scores[RETRIEVAL_MEAN] - strongest_score
    delta_draws = draws[:, RETRIEVAL_MEAN] - draws[:, :len(SIMPLE)].max(axis=1)
    contrasts = {'retrieval_minus_strongest_simple': estimate(delta, delta_draws),
                 'highest_scoring_simple_methods': strongest,
                 'retrieval_minus_each_simple': {method: estimate(scores[RETRIEVAL_MEAN] - scores[i],
                                                                   draws[:, RETRIEVAL_MEAN] - draws[:, i])
                                                for i, method in enumerate(SIMPLE)},
                 'each_retrieval_seed_minus_strongest_simple': {name: float(scores[METHODS.index(name)] - strongest_score)
                                                               for name in RETRIEVAL}}
    for name, i, j in [('retrieval_minus_centroid', RETRIEVAL_MEAN, CENTROID_MEAN),
                       ('retrieval_minus_scrambled_seed17', METHODS.index(RETRIEVAL[0]), METHODS.index(SCRAMBLED))]:
        contrasts[name] = estimate(scores[i] - scores[j], draws[:, i] - draws[:, j])
    discordance = {LABELS[i]: {'retrieval_only_success': float(means[i, 3]),
                               'comparator_only_success': float(means[i, 4])}
                   for i in [*range(len(SIMPLE)), CENTROID_MEAN, METHODS.index(SCRAMBLED)]}
    return dict(counts=counts, estimates=estimates, contrasts=contrasts, paired_discordance=discordance)


def analyze(path):
    groups = load_rows(path)
    domains = {domain: {subset: summarize(records, subset) for subset in ('all', 'variable_support', 'zero_reference_actions')}
               for domain, records in groups.items()}
    c = domains['external']['all']['contrasts']
    main = c['retrieval_minus_strongest_simple']
    practical = main['mean'] >= 0.02 and main['percentile_95'][0] > 0 and all(
        value > 0 for value in c['each_retrieval_seed_minus_strongest_simple'].values())
    control_support = c['retrieval_minus_centroid']['mean'] > 0 and c['retrieval_minus_scrambled_seed17']['mean'] > 0
    return dict(unit='Targets averaged within each source record, eligible records averaged within identical code, codes equally weighted. Each subgroup recomputes these denominators. Neural family estimates average individual seed performances.',
                metric='Exact reference endpoint on explicit STOP. Crash and cap rates are separate. Rates and differences are fractions.',
                uncertainty=dict(method='Paired exact-program percentile bootstrap; population unit is code. The maximum over all fixed simple methods is reselected in every draw. Intervals describe test-program variation conditional on fitted policies.', resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED),
                domains=domains, gate=dict(domain='external', subset='all', practical_advantage=bool(practical),
                positive_control_contrasts=bool(control_support), matching_candidate=bool(practical and control_support)))


def check_receipt(path, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            hasher.update(block)
    if receipt.get('completed') is not True or receipt.get('per_target_sha256') != hasher.hexdigest():
        raise ValueError('uncompleted evaluation or per-target digest differs from the evaluation receipt')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('per_target', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--receipt', type=Path, help='Evaluation summary (defaults to summary.json beside per_target).')
    args = parser.parse_args()
    receipt = check_receipt(args.per_target, args.receipt or args.per_target.with_name('summary.json'))
    result = analyze(args.per_target)
    for domain in DOMAINS:
        counts = result['domains'][domain]['all']['counts']
        denominator = dict(programs=counts['program_records'], distinct_codes=counts['distinct_codes'], targets=counts['targets'])
        if denominator != receipt['domains'][domain]:
            raise ValueError(f'{domain}: scoring receipt and analyzed denominators disagree')
    result['evidence'] = {key: receipt[key] for key in ('per_target_sha256', 'protocol_sha256', 'checkpoint_sha256')}
    result['evidence']['reducer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.out.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
