"""Counterfactual value of a fifth demonstration for two frozen executors."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from karel_online_control import STOP, demonstration
from karel_predicates import observe
from karel_unaligned_baselines import PrefixAlignment, build, support_weights
from karel_unaligned_model import NeuralPolicy, load_trained
from leaps_karel_source_format import LeapsWorld


NAMES = ('retrieval','edit')
CKPT = '670119eb898578c2d85268fe5fd2a681243c94c68d48100408b497c22b60ea35'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def agent(which, model, support):
    if which == 'retrieval':
        return NeuralPolicy(model,support)
    if len(support) == 5:
        return build('dtw_support_weighted',support)
    if len(support) != 4:
        raise ValueError('Unexpected support size')
    return PrefixAlignment(support,support_weights(support))


def execute(policy, initial, cap):
    world = initial.clone()
    for i in range(cap+1):
        obs = observe(world)
        action = policy.act(obs)
        if action == STOP:
            return 'stop',world.state()
        if i == cap:
            return 'cap',world.state()
        if not world.step(action):
            return 'crash',world.state()
        policy.advance(obs,action)
    raise RuntimeError('Unreachable')


def pick_one(scores):
    a,b = scores
    return 1. if b>a else 0. if a>b else .5


def utility(outcomes, p_one):
    return (1-p_one)*outcomes[0] + p_one*outcomes[1]


def label_free_candidates(base, candidate_starts, starts_of_base, model, cap):
    disagree, novelty = [], []
    for world in candidate_starts:
        guesses = [execute(agent(k,model,base),world,cap) for k in NAMES]
        disagree.append(int(guesses[0] != guesses[1]))
        bits = observe(world)
        novelty.append(min(sum(x != y for x,y in zip(bits,observe(s))) for s in starts_of_base))
    return {'disagreement':pick_one(disagree),'novelty':pick_one(novelty),'uniform':.5}


def run_record(record, model):
    cases = record['examples']
    if len(cases)!=10 or [x['example_index'] for x in cases] != list(range(10)):
        raise ValueError('Incomplete or reordered ten-world panel')
    order = sorted(range(10),key=lambda i:hashlib.sha256(f"karel_teaching_v1|{record['guid']}|{i}".encode()).digest())
    worlds = [LeapsWorld.parse(cases[i]['inpgrid_json']) for i in order]
    goals = [LeapsWorld.parse(cases[i]['outgrid_json']).state() for i in order]
    def demo(i):
        return demonstration(worlds[i],cases[order[i]]['actions'],goals[i])
    base = [demo(i) for i in range(4)]
    cap = min(512,max(32,2*max(len(t)-1 for t in base)+8))
    choices = label_free_candidates(base,worlds[4:6],worlds[:4],model,cap)
    all_out, baseline = {}, {}
    for k in NAMES:
        plain = [execute(agent(k,model,base),w,cap)==('stop',goal) for w,goal in zip(worlds[8:],goals[8:])]
        baseline[k]=mean(plain)
        all_out[k]=[]
        for fifth in (4,5):
            supports=base+[demo(fifth)]
            all_out[k].append([int(execute(agent(k,model,supports),worlds[i],cap)==('stop',goals[i]))
                               for i in range(6,10)])
    final = {k:[mean(row[2:]) for row in rows] for k,rows in all_out.items()}
    teach = {k:pick_one([sum(row[:2]) for row in rows]) for k,rows in all_out.items()}
    metrics = {}
    diff = {}
    for k,other in zip(NAMES,reversed(NAMES)):
        v = final[k]
        metrics[k+'_base']=baseline[k]
        metrics[k+'_oracle']=max(v)
        for label,p in dict(choices,own=teach[k],other=teach[other]).items():
            metrics[k+'_'+label]=utility(v,p)
        metrics[k+'_own_minus_other']=metrics[k+'_own']-metrics[k+'_other']
        for label in ('oracle','disagreement','novelty','own'):
            metrics[k+'_'+label+'_minus_uniform']=metrics[k+'_'+label]-metrics[k+'_uniform']
        metrics[k+'_random_demo_gain']=metrics[k+'_uniform']-baseline[k]
        diff[k]=v[1]-v[0]
    a,b=diff.values()
    metrics['strictly_opposite']=int(a*b<0)
    metrics['personalization_opportunity']=(abs(a)+abs(b)-abs(a+b))/4
    shared=max(mean((final[NAMES[0]][j],final[NAMES[1]][j])) for j in range(2))
    assert abs(metrics['personalization_opportunity']-(mean(max(v) for v in final.values())-shared))<1e-9
    metrics['interaction']=mean(metrics[k+'_own_minus_other'] for k in NAMES)
    metrics['calibration_choices_differ']=int(teach[NAMES[0]]!=teach[NAMES[1]])
    return {'code':record['code'],'world_order':order,'label_free_p_candidate_1':choices,'calibration_p_candidate_1':teach,
            'base':baseline,'candidate_by_calibration_then_final':all_out,'metrics':metrics}


def summarize(rows):
    keys=list(rows[0]['metrics'])
    matrix=np.array([[r['metrics'][k] for k in keys] for r in rows],dtype=float)
    rng=np.random.default_rng(20261120)
    blocks=[]
    for i in range(0,5000,64):
        idx=rng.integers(len(rows),size=(min(64,5000-i),len(rows)))
        blocks.append(matrix[idx].mean(axis=1))
    low,high=np.quantile(np.concatenate(blocks),[.025,.975],axis=0)
    summary={k:{'mean':float(m),'paired_program_95':[float(l),float(h)]} for k,m,l,h in zip(keys,matrix.mean(0),low,high)}
    discord=[r for r in rows if r['metrics']['calibration_choices_differ']]
    return {'programs':len(rows),'future_worlds':2*len(rows),'estimates':summary,
            'strictly_opposite_programs':sum(r['metrics']['strictly_opposite'] for r in rows),
            'calibration_choices_differ_programs':len(discord),
            'discordant_choices_interaction':mean(r['metrics']['interaction'] for r in discord) if discord else None,
            'retained':bool(summary['interaction']['mean']>=.02 and summary['interaction']['paired_program_95'][0]>0
                            and all(summary[k+'_own_minus_other']['mean']>0 for k in NAMES))}


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.is_available():
        raise RuntimeError('School CPU allocation required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt=json.loads(args.receipt.read_text())
    if not receipt['completed'] or digest(args.cohort)!=receipt['artifact_sha256'] or digest(args.protocol)!=receipt['protocol_sha256'] or digest(args.checkpoint)!=CKPT:
        raise ValueError('Inputs differ from fixed source or model')
    model=load_trained(args.checkpoint)['retrieval',17]
    with torch.inference_mode(),args.cohort.open() as f:
        rows=[run_record(json.loads(line),model) for line in f]
    if len(rows)!=receipt['eligible_programs'] or len(rows)!=len({r['code'] for r in rows}):
        raise ValueError('Panel missing or duplicated')
    args.out.mkdir(exist_ok=False,parents=True)
    detail=args.out/'per_program.jsonl'
    detail.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    result=summarize(rows)
    helpers=['karel_online_control.py','karel_unaligned_baselines.py','karel_unaligned_model.py','karel_predicates.py','leaps_karel_source_format.py']
    result.update(job=os.environ['SLURM_JOB_ID'],completed=True,source_receipt_sha256=digest(args.receipt),source_sha256=digest(args.cohort),
                  checkpoint_sha256=CKPT,worker_sha256=digest(Path(__file__)),protocol_sha256=digest(args.protocol),
                  notes_sha256=digest(args.notes),per_program_sha256=digest(detail),
                  helpers_sha256={k:digest(Path(__file__).with_name(k)) for k in helpers})
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ('protocol','notes','cohort','receipt','checkpoint','out'):
        p.add_argument('--'+opt,type=Path,required=True)
    main(p.parse_args())
