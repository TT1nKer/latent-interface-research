"""Check whether full rollout behavior distinguishes within-action support alignments."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean

import torch

from karel_trace_control import sha
from karel_unaligned_evaluate import code, load_banks, rollout
from karel_unaligned_model import ACTIONS, BOS, FEATURES, NeuralPolicy, load_trained, read_checkpoint
from karel_unaligned_sites import label_record, tied_score


class MassPolicy(NeuralPolicy):
    def __init__(self, model, supports):
        super().__init__(model, supports)
        _, prior, votes = self.memory
        self.real_votes = votes * torch.isfinite(prior).unsqueeze(-1)
        self.slot_counts = self.real_votes.sum(dim=1, keepdim=True).clamp_min(1)
        self.slot_to_action = self.real_votes.transpose(1, 2)
        self.use_uniform = False
        self.queries = self.changed = self.nonidentical = 0
        self.max_abs = 0.0

    def read(self, obs):
        x = torch.zeros(1, 1, FEATURES)
        x[0, 0, :4] = torch.tensor(obs, dtype=torch.float32)
        x[0, 0, 4 + self.previous] = 1
        q, self.pending_state = self.model.encode(x, self.state)
        keys, prior, votes = self.memory
        position = self.model.predict(q, (keys, prior, None))
        scores = position.bmm(votes)
        uniform = (scores / self.slot_counts).bmm(self.slot_to_action)
        new_scores = uniform.bmm(votes)
        if self.queries == 0:
            if not torch.equal(scores, self.model.predict(q, self.memory)):
                raise ValueError("Original action scores differ from frozen predict")
            if torch.count_nonzero(uniform.masked_select(~torch.isfinite(prior).unsqueeze(1))):
                raise ValueError("Redistribution included padding")
        return scores, new_scores, position, uniform

    def act(self, obs):
        scores, new_scores, _, _ = self.read(obs)
        if not torch.isfinite(scores).all() or not torch.isfinite(new_scores).all():
            raise ValueError("Nonfinite action score")
        delta = (scores - new_scores).abs().max().item()
        self.max_abs = max(self.max_abs, delta)
        self.nonidentical += delta > 0
        orig = scores.argmax(-1).item()
        new = new_scores.argmax(-1).item()
        self.queries += 1
        self.changed += orig != new
        return ACTIONS[new if self.use_uniform else orig]


def teacher_diagnostic(policy, demos_sites, target):
    """A scorer with private target labels evaluates already-computed position weights."""
    query, gold = target
    padded = max(len(demo) for demo, _ in demos_sites)
    candidates = defaultdict(list)
    for d, (demo, sites) in enumerate(demos_sites):
        for j, site in enumerate(sites):
            candidates[demo[j][1]].append((d * padded + j, j, site))
    count = 0
    learned_hits = uniform_hits = 0.0
    for (obs, action), correct in zip(query[:-1], gold, strict=True):
        _, _, positions, redistributed = policy.read(obs)
        matched = candidates[action]
        good = sum(site == correct for _, _, site in matched)
        if 0 < good < len(matched):
            count += 1
            learned_hits += tied_score(positions[0, 0].tolist(), matched, correct)
            expected = tied_score(redistributed[0, 0].tolist(), matched, correct)
            if abs(expected - good / len(matched)) > 1e-15:
                raise ValueError("Intervened weights do not tie all real same-primitive slots")
            uniform_hits += expected
        policy.advance(obs, action)
    return count, learned_hits, uniform_hits


def external_prior(path, individual):
    result = {}
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['domain'] != 'external':
                continue
            key = (r['program_id'], r['target_id']) if individual else r['program_id']
            if key in result:
                raise ValueError("Duplicate prior identity")
            result[key] = r
    return result


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    job_id = os.environ['SLURM_JOB_ID']
    protocol_path = Path('karel_equal_action_mass_20261120.json')
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['pins'].items():
        if digest != sha(Path(name)):
            raise ValueError(f'Input changed: {name}')
    checkpoint = Path(protocol['checkpoint'])
    train_codes = set(read_checkpoint(checkpoint)['training_codes'])
    domains = load_banks(args, train_codes)
    world_class, records = next((cls, rows) for d, cls, rows in domains if d == 'external')
    model = load_trained(checkpoint)['retrieval', 17]
    archived = external_prior(Path(protocol['per_target']), True)
    archived_sites = external_prior(Path(protocol['per_program_sites']), False)
    out = Path(f'results/karel_equal_action_mass_{job_id}')
    out.mkdir(exist_ok=False)
    details = out/'per_target.jsonl'
    aggregates = defaultdict(list)
    with torch.inference_mode(), details.open('x') as stream:
        for record in records:
            guid, digest = record['guid'], code(record)
            old_sites = archived_sites.pop(guid)
            if digest != old_sites['code']:
                raise ValueError("Site record code mismatch")
            trajectories = label_record(record, world_class)
            for j, target in enumerate(trajectories):
                old = archived.pop((guid, j))
                others = trajectories[:j] + trajectories[j+1:]
                demos = [demo for demo, sites in others]
                example = record['examples'][j]
                start = world_class.parse(example['inpgrid_json'])
                goal = world_class.parse(example['outgrid_json']).state()
                cap = min(512, max(32, 2*max(len(d)-1 for d in demos) + 8))
                if old['rollout_cap'] != cap or old['code_id'] != digest:
                    raise ValueError('Original cap or code differs')
                policy = MassPolicy(model, demos)
                origin = rollout(policy, start, goal, cap)
                if origin != old['methods']['retrieval_17']:
                    raise ValueError('Original outcome failed to reproduce')
                # With identical choices on the original path, deterministic execution
                # and recurrent state must stay identical through the terminal boundary.
                intervened = origin
                if policy.changed:
                    alternative = MassPolicy(model, demos)
                    alternative.use_uniform = True
                    intervened = rollout(alternative, start, goal, cap)
                scorer = MassPolicy(model, demos)
                steps, learned_hits, uniform_hits = teacher_diagnostic(scorer, others, target)
                previous = old_sites['targets'][j]['groups']['hard']
                if steps != previous['steps'] or abs(learned_hits - previous['scores']['retrieval_17']) > 1e-10:
                    raise ValueError('Original hard-position diagnostic differs')
                row = dict(code=digest, program_id=guid, target=j, original=origin, intervened=intervened,
                           original_path_queries=policy.queries, action_flips=policy.changed,
                           nonidentical_scores=policy.nonidentical, max_abs=policy.max_abs,
                           hard_steps=steps, learned_hits=learned_hits, uniform_hits=uniform_hits)
                aggregates[digest].append(row)
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    if archived or archived_sites or len(aggregates) != 764 or any(len(v) != 6 for v in aggregates.values()):
        raise ValueError("Incomplete external cohort")
    all_rows = [r for group in aggregates.values() for r in group]
    eligible = [group for group in aggregates.values() if sum(r['hard_steps'] for r in group)]
    summed = lambda rows, field: sum(r[field] for r in rows)
    hard_steps = summed(all_rows, 'hard_steps')
    if len(eligible) != 524 or hard_steps != 10369:
        raise ValueError("Hard-position denominator changed")
    site = {}
    for metric in ('learned_hits', 'uniform_hits'):
        site[metric] = dict(code_macro=mean(summed(g, metric)/summed(g, 'hard_steps') for g in eligible),
                            step_micro=summed(all_rows, metric)/hard_steps)
    result = dict(completed=True, job=job_id, posthoc=True, programs=len(aggregates), worlds=len(all_rows),
                  original_records_reproduced=len(all_rows), original_sites_reproduced=len(all_rows),
                  hard_codes=len(eligible), hard_steps=hard_steps, sites=site,
                  original_path_queries=summed(all_rows, 'original_path_queries'),
                  nonidentical_scores=summed(all_rows, 'nonidentical_scores'),
                  max_abs=max(r['max_abs'] for r in all_rows),
                  original_path_action_flips=summed(all_rows, 'action_flips'),
                  alternative_paths_executed=sum(bool(r['action_flips']) for r in all_rows),
                  changed_outcome_records=sum(r['original'] != r['intervened'] for r in all_rows),
                  original_successes=sum(r['original']['success'] for r in all_rows),
                  intervened_successes=sum(r['intervened']['success'] for r in all_rows),
                  protocol_sha256=sha(protocol_path), worker_sha256=sha(Path(__file__)),
                  per_target_sha256=sha(details))
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('dev-zip', 'external', 'source-receipt', 'external-source-protocol'):
        parser.add_argument('--'+arg, type=Path, required=True)
    main(parser.parse_args())
