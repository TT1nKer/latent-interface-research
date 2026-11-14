"""Stdlib timing feasibility analysis. This never estimates robot task success."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def quantile(xs, q):
    xs = sorted(xs)
    t = (len(xs) - 1) * q
    a = math.floor(t)
    return xs[a] + (xs[min(a + 1, len(xs) - 1)] - xs[a]) * (t - a)


def correlation(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    den = math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        assert math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0
        groups[(r['episode'], r['frame'])].append(r)
    assert groups and all(sorted(r['repeat'] for r in g) == [0, 1, 2] for g in groups.values())
    assert all(len({r['retained_tokens'] for r in g}) == 1 for g in groups.values()), 'controller replay differs'
    assert all(len({json.dumps(r['controller_history_before']) for r in g}) == 1 for g in groups.values())
    medians = [st.median(r['wall_seconds'] for r in g) for g in groups.values()]
    jitter = [max(r['wall_seconds'] for r in g) - min(r['wall_seconds'] for r in g) for g in groups.values()]
    tokens = [g[0]['retained_tokens'] for g in groups.values()]
    center = st.median(medians)
    fixed_steps = math.ceil(center * 20)
    steps = [math.ceil(m * 20) for m in medians]
    stable_mismatch = [all(math.ceil(r['wall_seconds'] * 20) != fixed_steps for r in g)
                       and len({math.ceil(r['wall_seconds'] * 20) for r in g}) == 1 for g in groups.values()]
    per_episode = defaultdict(list)
    for key, med, tok in zip(groups, medians, tokens): per_episode[key[0]].append((med, tok))
    centered_times, centered_tokens = [], []
    for pairs in per_episode.values():
        mt, mk = st.mean(p[0] for p in pairs), st.mean(p[1] for p in pairs)
        centered_times.extend(p[0] - mt for p in pairs)
        centered_tokens.extend(p[1] - mk for p in pairs)
    spread = quantile(centered_times, .9) - quantile(centered_times, .1)
    return {'observations': len(groups), 'median_wall_seconds': center,
            'p10_wall_seconds': quantile(medians, .1), 'p90_wall_seconds': quantile(medians, .9),
            'median_within_observation_range_seconds': st.median(jitter),
            'within_episode_p90_minus_p10_seconds': spread,
            'token_min': min(tokens), 'token_max': max(tokens),
            'within_episode_token_time_correlation': correlation(centered_tokens, centered_times),
            'fixed_median_delivery_steps_20hz': fixed_steps,
            'delivery_step_disagreement_fraction': st.mean(s != fixed_steps for s in steps),
            'stable_three_replay_delivery_disagreement_fraction': st.mean(stable_mismatch),
            'median_absolute_fixed_time_error_seconds': st.median(abs(m - center) for m in medians),
            'p90_absolute_fixed_time_error_seconds': quantile([abs(m - center) for m in medians], .9),
            'max_reserved_bytes': max(r['max_reserved_bytes'] for r in rows)}


def analyze(folder):
    folder = Path(folder)
    native = json.loads((folder / 'native_summary.json').read_text())
    fast = json.loads((folder / 'fastvla_summary.json').read_text())
    assert native['completed'] and fast['completed'] and fast['native_check_all'] is True
    assert native['cohort_sha256'] == fast['cohort_sha256']
    assert fast['rows_sha256'] == digest(folder / 'fastvla.jsonl')
    assert native['rows_sha256'] == digest(folder / 'native.jsonl')
    rows = [json.loads(s) for s in (folder / 'fastvla.jsonl').read_text().splitlines()]
    assert len(rows) == fast['rows']
    by_mode = {mode: [r for r in rows if r['mode'] == mode] for mode in ('disabled', 'fixed', 'adaptive')}
    disabled = by_mode['disabled']
    assert len(disabled) == fast['native_check_count'] == 3 * native['rows']
    assert all(r['native_tolerance_passed'] is True for r in disabled)
    cohorts = [{(r['episode'], r['frame'], r['repeat']) for r in rs} for rs in by_mode.values()]
    assert all(c == cohorts[0] for c in cohorts)
    metrics = {mode: summarize(rs) for mode, rs in by_mode.items()}
    a, f = metrics['adaptive'], metrics['fixed']
    gates = {
        'nonconstant_adaptive_token_count': a['token_max'] > a['token_min'],
        'stable_20hz_difference_at_least_ten_percent': a['stable_three_replay_delivery_disagreement_fraction'] >= .1,
        'state_spread_exceeds_twice_replay_jitter': a['within_episode_p90_minus_p10_seconds'] > 2 * a['median_within_observation_range_seconds'],
        'adaptive_spread_exceeds_fixed_plus_jitter': a['within_episode_p90_minus_p10_seconds'] > f['within_episode_p90_minus_p10_seconds'] + a['median_within_observation_range_seconds'],
        'below_22gib': max(native['max_reserved_bytes'], fast['max_reserved_bytes']) < 22 * 1024**3,
    }
    return {'schema': 'vla_latency_feasibility_v1', 'metrics': metrics, 'triage_gates': gates,
            'candidate_for_closed_loop_design': all(gates.values()),
            'decision': 'Eligible to design an independent closed-loop experiment; not a mechanism discovery.' if all(gates.values())
                        else 'Do not expand this first deployment branch on current evidence; preserve limits and failures.',
            'native_max_abs_error': max(r['native_max_abs_error'] for r in disabled),
            'native_exact_action_chunks': sum(r['native_exact'] for r in disabled),
            'native_check_count': len(disabled), 'cohort_sha256': fast['cohort_sha256'],
            'limits': ['Development replay of three public training demonstrations, not closed-loop task success.',
                       'Three replays do not establish a universal latency distribution or hardware robustness.',
                       '20 Hz buckets are a simulator measurement convention, not a physical efficacy result.',
                       'Within-episode centering reduces task prompt confounding; does not establish causality.',
                       'Upstream synchronization/bookkeeping retained; measured release runtime, not optimized deployment.',
                       'Failure rejects this bounded evidence for expansion, not every adaptive policy or hardware regime.']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    result = analyze(args.folder)
    with out.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__': main()
