"""Test whether a frozen action matcher can carry a second output on ambiguous primitives."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

from karel_predicates import observe
from karel_unaligned_baselines import build, obs_code
from karel_unaligned_sites import PositionReader, label_record
from leaps_karel_source_format import LeapsWorld, parse_program

METHODS = ('neural', 'uniform', 'clock', 'occurrence', 'edit')
SOURCE = '4fb37ac92f658b4769ad7543d7f613f7373aff4be29b33bb46ac0f6ea5035d90'
CHECKPOINT = '670119eb898578c2d85268fe5fd2a681243c94c68d48100408b497c22b60ea35'


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def choose_marks(code, supports):
    by_primitive = defaultdict(set)
    for demo, labels in supports:
        for (_, action), site in zip(demo[:-1], labels, strict=True):
            by_primitive[action].add(site)
    pairs = [pair for sites in by_primitive.values()
             for pair in combinations(sorted(sites, key=json.dumps), 2)]
    if not pairs:
        return None
    def rank(pair):
        return hashlib.sha256(json.dumps([code, *pair]).encode()).digest()
    return min(pairs, key=rank)


def labeled_example(row, program):
    def window(examples):
        return label_record(dict(row, program_json=program, examples=examples), LeapsWorld)
    first = window(row['examples'][:6])
    second = window(row['examples'][4:])
    if first[-2:] != second[:2]:
        raise ValueError('Repeated label windows disagree')
    return first + second[2:]


def support_slots(support, tags):
    groups = defaultdict(list)
    stride = max(map(len, support))
    for d, (trace, signals) in enumerate(zip(support, tags, strict=True)):
        rank = Counter()
        for j, ((sensors, action), pair) in enumerate(zip(trace[:-1], signals, strict=True)):
            groups[action].append((d, j, d*stride+j, obs_code(sensors), rank[action], pair))
            rank[action] += 1
    return groups


def outputs(weights, tags):
    total = sum(weights)
    return tuple(2*sum(w for w, pair in zip(weights,tags,strict=True) if pair[i]) > total
                 for i in range(2))


def vote_slots(reader, motor, obs, t, nth, slots):
    probs = reader.weights(obs)
    cost = motor.cost
    q = obs_code(obs)
    weights, clock, occurrence, edit, tags = [], [], [], [], []
    for d, j, k, key, rank, pair in slots:
        weights.append(probs[k])
        clock.append(abs(t-j))
        occurrence.append(abs(nth-rank))
        edit.append(motor.prefix[d][j]+cost[q^key])
        tags.append(pair)
    votes = dict(neural=outputs(weights,tags), uniform=outputs([1]*len(tags),tags))
    for label, values in (('clock',clock),('occurrence',occurrence),('edit',edit)):
        best = min(values)
        votes[label] = outputs([val == best for val in values],tags)
    return votes


def run_world(model, support, slots, world):
    motor = build('dtw_support_weighted',support)
    reader = PositionReader(model,support)
    cap = min(512,max(32,2*max(len(d)-1 for d in support)+8))
    emissions = {k:[[],[]] for k in METHODS}
    ranks = Counter()
    for t in range(cap+1):
        obs = observe(world)
        action = motor.act(obs)
        if action == 'STOP':
            return 'stop',world.state(),emissions
        if t == cap:
            return 'cap',world.state(),emissions
        if not slots[action]:
            raise ValueError('Motor chose unseen primitive')
        votes = vote_slots(reader,motor,obs,t,ranks[action],slots[action])
        if not world.step(action):
            return 'crash',world.state(),emissions
        for label, pair in votes.items():
            for i, pulse in enumerate(pair):
                if pulse:
                    emissions[label][i].append(t)
        ranks[action] += 1
        motor.advance(obs,action)
        reader.advance(obs,action)
    raise RuntimeError('Unreachable step cap')


def evaluate(row, tokens, model):
    code = row['code']
    if hashlib.sha256(' '.join(tokens).encode()).hexdigest() != code:
        raise ValueError('Different native program')
    traces = labeled_example(row,parse_program(tokens))
    order = sorted(range(10),key=lambda i:hashlib.sha256(f'event-transfer-v1|{code}|{i}'.encode()).digest())
    donors = [traces[i] for i in order[:5]]
    marks = choose_marks(code,donors)
    if marks is None:
        return None
    support = [d for d,_ in donors]
    tags = [[(s==marks[0],s==marks[1]) for s in labels] for _,labels in donors]
    slots = support_slots(support,tags)
    results = []
    for i in order[5:]:
        example = row['examples'][i]
        status,state,emitted = run_world(model,support,slots,LeapsWorld.parse(example['inpgrid_json']))
        expected = [[j for j,s in enumerate(traces[i][1]) if s==mark] for mark in marks]
        grid = status=='stop' and state == LeapsWorld.parse(example['outgrid_json']).state()
        correct = {k:[grid and a==b for a,b in zip(pair,expected,strict=True)] for k,pair in emitted.items()}
        results.append(dict(world=i, motor=grid, distinguished=expected[0]!=expected[1],
                            missions=correct, both={k:all(v) for k,v in correct.items()}))
    return dict(code=code,tests=results)


def summarize(rows):
    import numpy as np
    from statistics import mean
    bank = np.array([[mean(t['both'][k] for t in r['tests']) for k in METHODS] for r in rows])
    averages = bank.mean(axis=0)
    rng = np.random.default_rng(902107)
    draws = []
    for i in range(0,5000,100):
        sampled = bank[rng.integers(len(bank),size=(100,len(bank)))].mean(axis=1)
        draws.extend(sampled[:,0]-sampled[:,1:].max(axis=1))
    lo,hi = map(float,np.quantile(draws,[.025,.975]))
    gain=float(averages[0]-max(averages[1:]))
    tests = [t for row in rows for t in row['tests']]
    distinguished = [t for t in tests if t['distinguished']]
    return dict(programs=len(rows),targets=len(tests),differing_reference_streams=len(distinguished),
                both=dict(zip(METHODS,map(float,averages))),motor_ceiling=mean(t['motor'] for t in tests),
                per_mission={k:mean(x for t in tests for x in t['missions'][k]) for k in METHODS},
                both_on_differing_worlds_pooled={k:mean(t['both'][k] for t in distinguished) for k in METHODS} if distinguished else {},
                neural_minus_strongest_simple=gain,paired_program_95=[lo,hi],retained=len(rows)>=100 and gain>=.02 and lo>0)


def main(a):
    import torch
    from karel_unaligned_model import load_trained
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.is_available():
        raise RuntimeError('School CPU allocation required')
    if digest(a.cohort)!=SOURCE or digest(a.checkpoint)!=CHECKPOINT:
        raise ValueError('Unexpected source or model')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_trained(a.checkpoint)['retrieval',17]
    programs = json.loads(a.tokens.read_text())
    seen, rows = set(), []
    with torch.inference_mode(), a.cohort.open() as f:
        for line in f:
            row = json.loads(line)
            code = row['code']
            if code in seen or len(row['examples'])!=10:
                raise ValueError('Duplicated or incomplete source')
            seen.add(code)
            outcome = evaluate(row, programs[code], model)
            if outcome is not None:
                rows.append(outcome)
    if len(seen)!=766 or seen!=set(programs):
        raise ValueError('Program registry changed')
    a.out.mkdir(parents=True,exist_ok=False)
    detail=a.out/'per_program.jsonl'
    detail.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    result=summarize(rows)
    result.update(completed=True,job=os.environ['SLURM_JOB_ID'],all_screened=len(seen),worker_sha256=digest(Path(__file__)),
                  cohort_sha256=SOURCE,checkpoint_sha256=CHECKPOINT,protocol_sha256=digest(a.protocol),tokens_sha256=digest(a.tokens),
                  per_program_sha256=digest(detail))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','tokens','protocol','checkpoint','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
