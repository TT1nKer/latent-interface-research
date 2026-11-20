"""Evaluate the frozen demonstration policies on complete held-out worlds in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from karel_online_control import STOP, demonstration
from karel_predicates import observe
from karel_trace_control import ACTIONS, World, sha
from karel_unaligned_baselines import METHODS, build
from karel_unaligned_model import SEEDS, NeuralPolicy, load_trained, read_checkpoint
from leaps_karel_source_format import LeapsWorld


DEV_ZIP_SHA = '967dc1c557595e7c4a5ff08a636ec3fe2b5a5320fd6548be50295cf36c5ab4cd'
SOURCE_GATE_SHA = '968746d7d875fb425cab6fb0e210bf1e8e923fa569313fcd7ef74a950b4c864a'
SOURCE_PROTOCOL_SHA = 'c62afad2df99a7a9dd5f5e7baf0b0948b382dab64f65d4e19fbdd9ecdbe36dbe'


def fingerprint(path):
    return sha(path)


def write(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2)+'\n')
    temp.replace(path)


def code(row):
    return hashlib.sha256(' '.join(row['program_tokens']).encode()).hexdigest()


def dev_holdout(digest):
    return int(digest[:8],16) % 5 == 0


def load_banks(args, training_codes):
    gate = json.loads(args.source_receipt.read_text())
    if (fingerprint(args.dev_zip) != DEV_ZIP_SHA or fingerprint(args.source_receipt) != SOURCE_GATE_SHA or
            fingerprint(args.external_source_protocol) != SOURCE_PROTOCOL_SHA or
            gate['protocol_sha256'] != SOURCE_PROTOCOL_SHA or not gate['completed'] or
            not gate['native_validity_passed'] or fingerprint(args.external) != gate['artifact_sha256'] or
            gate['native_steps'] + gate['validated_stop'] != gate['native_observations'] or
            gate['native_states'] != gate['native_observations'] or gate['selected_worlds'] != gate['validated_stop'] or
            gate['eligible_exact_overlap_prior_leaps'] or gate['eligible_exact_overlap_development']):
        raise ValueError('Source provenance, native validation, or disjointness check failed')
    for name in ('karel_predicates.py','karel_trace_control.py','leaps_karel_source_format.py'):
        if fingerprint(Path(__file__).with_name(name)) != gate['code_sha256'][name]:
            raise ValueError(f'Native simulator source changed: {name}')

    holdout, all_dev, fitted = [], set(), set()
    dev_record_count = 0
    with zipfile.ZipFile(args.dev_zip) as archive, archive.open('val.json') as data:
        for line in data:
            row = json.loads(line)
            digest = code(row)
            dev_record_count += 1
            if len(row['examples']) < 6:
                raise ValueError('Development record missing an original world')
            all_dev.add(digest)
            if dev_holdout(digest):
                if digest in training_codes:
                    raise ValueError('A held-out code was used for fitting')
                holdout.append(row)
            else:
                fitted.add(digest)
    if dev_record_count != 2500 or len(all_dev) != 2045 or fitted != training_codes:
        raise ValueError('Checkpoint training codes differ from the frozen development partition')
    external, ext_codes = [], set()
    with args.external.open() as data:
        for line in data:
            row = json.loads(line)
            digest = code(row)
            if (not row['eligible'] or row['exclusion'] is not None or row['code'] != digest or
                    digest in all_dev or digest in ext_codes or len(row['examples']) != 6):
                raise ValueError('External eligibility or code disjointness failed')
            ext_codes.add(digest)
            external.append(row)
    if len(external) != gate['eligible_programs'] or len(ext_codes)*6 != gate['eligible_worlds']:
        raise ValueError('External denominator mismatch')
    return [('development_holdout',World,holdout),('external',LeapsWorld,external)]


def rollout(policy, original, target_state, cap):
    current = original.clone()
    steps = 0
    while True:
        observation = observe(current)
        act = policy.act(observation)
        if act == STOP:
            return {'success':current.state() == target_state, 'termination':'stop', 'steps':steps}
        if act not in ACTIONS:
            raise ValueError(f'Policy returned an unknown primitive {act!r}')
        if steps == cap:
            return {'success':False, 'termination':'cap', 'steps':steps}
        if not current.step(act):
            return {'success':False, 'termination':'crash', 'steps':steps}
        policy.advance(observation, act)
        steps += 1


def policies(models, supports):
    for method in METHODS:
        yield method,build(method,supports)
    for variant in ('retrieval','centroid'):
        for seed in SEEDS:
            yield f'{variant}_{seed}',NeuralPolicy(models[variant,seed],supports)
    yield 'initial_17',NeuralPolicy(models['initial',17],supports)
    yield 'retrieval_scrambled_17',NeuralPolicy(models['retrieval',17],supports,scrambled=True)


def per_target(domain, cls, record, models):
    cases = record['examples'][:6]
    starts = [cls.parse(case['inpgrid_json']) for case in cases]
    goals = [cls.parse(case['outgrid_json']).state() for case in cases]
    demos = [demonstration(world,case['actions'],goal) for world,case,goal in zip(starts,cases,goals,strict=True)]
    for target in range(6):
        # No target action, length, endpoint or absolute map is supplied to the policy.
        support = tuple(tuple(d) for j,d in enumerate(demos) if j != target)
        traces = {tuple(action for _,action in d[:-1]) for d in support}
        cap = min(512,max(32,2*max(map(len,traces))+8))
        outcomes = {name:rollout(p,starts[target],goals[target],cap) for name,p in policies(models,support)}
        yield {'domain':domain,'code_id':code(record),'program_id':record['guid'],'target_id':target,
               'reference_action_count':len(cases[target]['actions']),'support_unique_traces':len(traces),
               'rollout_cap':cap,'methods':outcomes}


def main(args):
    import torch
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This scoring worker requires a Slurm allocation')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if tuple(SEEDS) != (17,29,43):
        raise ValueError('Unexpected learner seeds')
    timer = time.monotonic()
    ckpt = fingerprint(args.checkpoint)
    protocol = json.loads(args.protocol.read_text())
    if tuple(protocol['comparators']) != tuple(METHODS):
        raise ValueError('Simple policies differ from preregistration')
    tr = args.checkpoint.with_name('training_summary.json')
    training = json.loads(tr.read_text())
    model_hash = fingerprint(Path(__file__).with_name('karel_unaligned_model.py'))
    if (training['checkpoint_sha256'] != ckpt or training['model_sha256'] != model_hash or
            training['protocol_sha256'] != fingerprint(args.protocol) or training['source_sha256'] != DEV_ZIP_SHA):
        raise ValueError('Training checkpoint, protocol or worker fingerprint mismatch')
    payload = read_checkpoint(args.checkpoint)
    trainer_hash = fingerprint(Path(__file__).with_name('karel_unaligned_train.py'))
    if not training['completed'] or training['trainer_sha256'] != trainer_hash:
        raise ValueError('Unverified training source or completion')
    for name in ('source_sha256','protocol_sha256','model_sha256','trainer_sha256','seeds'):
        if payload[name] != training[name]:
            raise ValueError(f'Checkpoint metadata differs from training receipt: {name}')
    training_codes = set(payload['training_codes'])
    ordered_hash = hashlib.sha256(('\n'.join(sorted(training_codes))+'\n').encode()).hexdigest()
    if ordered_hash != training['training_code_sha256'] or tuple(training['seeds']) != tuple(SEEDS):
        raise ValueError('Invalid fitting code registry or seed set')
    if len(training_codes) != len(payload['training_codes']) or len(training_codes) != training['training_code_count']:
        raise ValueError('Training code registry contains duplicates')
    banks = load_banks(args,training_codes)
    models = load_trained(args.checkpoint)
    expected = {(v,s) for v in ('retrieval','centroid') for s in SEEDS} | {('initial',17)}
    if set(models) != expected:
        raise ValueError('Checkpoint learner set mismatch')
    for m in models.values():
        m.eval()
    args.out.mkdir(parents=True,exist_ok=False)
    workers = ('karel_unaligned_evaluate.py','karel_unaligned_model.py','karel_unaligned_baselines.py',
               'karel_online_control.py','karel_predicates.py','karel_trace_control.py','leaps_karel_source_format.py')
    receipt = {'completed':False,'slurm_job_id':os.environ['SLURM_JOB_ID'],'checkpoint_sha256':ckpt,
               'training_summary_sha256':fingerprint(tr),'protocol_sha256':fingerprint(args.protocol),
               'training_codes':len(training_codes),'dev_source_sha256':DEV_ZIP_SHA,
               'external_sha256':fingerprint(args.external),'external_source_receipt_sha256':SOURCE_GATE_SHA,
               'workers_sha256':{f:fingerprint(Path(__file__).with_name(f)) for f in workers},
               'horizon':'successful primitives before STOP, illegal action, or cap; STOP allowed at cap'}
    write(args.out/'started.json',receipt)
    tmp = args.out/'per_target.jsonl.partial'
    with torch.inference_mode(), tmp.open('x') as out:
        sizes = {}
        for domain,cls,records in banks:
            guids = [r['guid'] for r in records]
            if len(set(guids)) != len(guids):
                raise ValueError('Duplicate record GUID in source')
            count = 0
            for row in records:
                for item in per_target(domain,cls,row,models):
                    out.write(json.dumps(item,separators=(',',':'))+'\n')
                    count += 1
            if count != 6*len(records):
                raise ValueError('Incomplete paired evaluations')
            sizes[domain] = {'programs':len(records),'distinct_codes':len({code(row) for row in records}),
                             'targets':count}
    path = args.out/'per_target.jsonl'; tmp.replace(path)
    receipt.update({'completed':True,'domains':sizes,'per_target_sha256':fingerprint(path),
                    'wall_s':round(time.monotonic()-timer,3)})
    write(args.out/'summary.json',receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint','protocol','dev-zip','external','source-receipt','external-source-protocol','out'):
        parser.add_argument('--'+arg,required=True,type=Path)
    main(parser.parse_args())
