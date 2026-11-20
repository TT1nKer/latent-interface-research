"""Paired routing and memory interventions with one paid demonstration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from karel_teaching_transfer import LeapsWorld, NAMES, agent, demonstration, digest, execute, load_trained


def route(correct):
    return 'retrieval' if correct['retrieval'] and not correct['edit'] else 'edit'


def factorial(before, after, chosen):
    fixed_before = before['edit']
    routed_before = before[chosen]
    fixed_after = after['edit']
    routed_after = after[chosen]
    route_before = routed_before - fixed_before
    route_after = routed_after - fixed_after
    routed_insert = routed_after - routed_before
    fixed_insert = fixed_after - fixed_before
    values = dict(fixed_before=fixed_before, routed_before=routed_before, fixed_after=fixed_after,
                  routed_after=routed_after, route_gain_before=route_before, route_gain_after=route_after,
                  interaction=route_after-route_before, fixed_insertion_gain=fixed_insert,
                  routed_insertion_gain=routed_insert, switch=int(chosen == 'retrieval'),
                  route_help_insert_harm=int(route_before > 0 and routed_insert < 0))
    for label, change in [('routed', routed_insert), ('fixed', fixed_insert)]:
        values[label+'_insert_help'] = int(change > 0)
        values[label+'_insert_harm'] = int(change < 0)
        values[label+'_insert_tie'] = int(change == 0)
    return values


def program(record, original, model):
    cases = record['examples']
    if [row['example_index'] for row in cases] != list(range(10)) or record['code'] != original['code']:
        raise ValueError('Cohort does not align with the main trial')
    permutation = sorted(range(10), key=lambda i: hashlib.sha256(
        f"karel_teaching_v1|{record['guid']}|{i}".encode()).digest())
    if permutation != original['world_order']:
        raise ValueError('Role assignment changed')
    cases = [cases[i] for i in permutation]
    starts = [LeapsWorld.parse(row['inpgrid_json']) for row in cases]
    base = [demonstration(starts[i], cases[i]['actions'], LeapsWorld.parse(cases[i]['outgrid_json']).state())
            for i in range(4)]
    cap = min(512, max(32, 2*max(len(demo)-1 for demo in base)+8))

    def four_correct(which, world):
        goal = LeapsWorld.parse(cases[world]['outgrid_json']).state()
        return execute(agent(which, model, base), starts[world], cap) == ('stop', goal)

    base_success = {which: mean(four_correct(which, idx) for idx in (8,9)) for which in NAMES}
    if base_success != original['base']:
        raise ValueError('Four-support future replay differs from native main trial')

    candidates = []
    for j in range(2):
        observed = {which: four_correct(which, 4+j) for which in NAMES}
        chosen = route(observed)
        success = {}
        for which in NAMES:
            endpoints = original['candidate_by_calibration_then_final'][which][j][2:]
            if len(endpoints) != 2 or any(value not in (0,1) for value in endpoints):
                raise ValueError('Missing five-support final outcomes')
            success[which] = mean(endpoints)
        cells = factorial(base_success, success, chosen)
        candidates.append(dict(exact_on_paid_candidate=observed, chosen=chosen,
                               five_support_final_mean=success, comparisons=cells))
    averages = {key: mean(c['comparisons'][key] for c in candidates) for key in candidates[0]['comparisons']}
    return dict(code=record['code'], world_order=permutation, four_support_final_mean=base_success,
                candidates=candidates, uniform_candidate=averages)


def summarize(rows):
    keys = list(rows[0]['uniform_candidate'])
    values = np.asarray([[row['uniform_candidate'][k] for k in keys] for row in rows])
    rng = np.random.default_rng(20261120)
    means = []
    for i in range(0, 5000, 64):
        indices = rng.integers(len(rows), size=(min(64,5000-i), len(rows)))
        means.append(values[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(means), [.025,.975], axis=0)
    result = {k:dict(mean=float(m),paired_program_95=[float(l),float(h)])
              for k,m,l,h in zip(keys,values.mean(axis=0),low,high)}
    gate = result['route_gain_after']
    return dict(eligible_programs=len(rows), estimates=result,
                retained=(gate['mean'] >= .02 and gate['paired_program_95'][0] > 0))


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.is_available():
        raise RuntimeError('School CPU allocation required')
    spec = json.loads(args.protocol.read_text())
    main_summary = json.loads((args.main/'summary.json').read_text())
    source = json.loads(args.receipt.read_text())
    original_rows = args.main/'per_program.jsonl'
    helpers = {name:digest(Path(__file__).with_name(name)) for name in spec['helpers_sha256']}
    if helpers != spec['helpers_sha256'] or not main_summary['completed'] or not source['completed']:
        raise ValueError('Incomplete source, main study, or changed policy code')
    for k, val in main_summary['helpers_sha256'].items():
        if helpers.get(k) != val:
            raise ValueError('Main study used different helpers')
    checks = {
        'job': (str(main_summary['job']), spec['source_job']),
        'main_worker': (main_summary['worker_sha256'], helpers['karel_teaching_transfer.py']),
        'checkpoint': (digest(args.checkpoint), main_summary['checkpoint_sha256'],spec['model_sha256']),
        'cohort': (digest(args.cohort),main_summary['source_sha256'],source['artifact_sha256'],spec['source_cohort_sha256']),
        'receipt': (digest(args.receipt),main_summary['source_receipt_sha256'],spec['source_receipt_sha256']),
        'source_protocol': (source['protocol_sha256'],main_summary['protocol_sha256'],spec['source_protocol_sha256']),
        'per_program': (digest(original_rows),main_summary['per_program_sha256']),
        'n': (source['eligible_programs'],main_summary['programs']),
    }
    for name, vals in checks.items():
        if len(set(vals)) != 1:
            raise ValueError('Source differs at '+name)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_trained(args.checkpoint)['retrieval',17]
    with torch.inference_mode(), args.cohort.open() as fs, original_rows.open() as fo:
        rows = [program(json.loads(rec),json.loads(orig),model) for rec,orig in zip(fs,fo,strict=True)]
    if len(rows) != source['eligible_programs'] or len(rows) != len({r['code'] for r in rows}):
        raise ValueError('Incomplete panel')
    args.out.mkdir(parents=True,exist_ok=False)
    path = args.out/'per_program.jsonl'
    path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    summary = summarize(rows)
    summary.update(completed=True, job=os.environ['SLURM_JOB_ID'],source_receipt_sha256=digest(args.receipt),
                   main_summary_sha256=digest(args.main/'summary.json'),main_rows_sha256=digest(original_rows),
                   source_cohort_sha256=digest(args.cohort),worker_sha256=digest(Path(__file__)),
                   protocol_sha256=digest(args.protocol),helpers_sha256=helpers,per_program_sha256=digest(path))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for opt in ('main','receipt','cohort','checkpoint','protocol','out'):
        parser.add_argument('--'+opt,required=True,type=Path)
    main(parser.parse_args())
