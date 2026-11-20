"""Held-out composition of the released BiVLC caption- and image-ranking decisions."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    return {'True': True, 'False': False}[value]


def cycle(rows, cols):
    """Input two choices (indices 0 or 1) each for image and caption queries."""
    return (rows[:, 0] != rows[:, 1]) & (cols[:, 0] != cols[:, 1]) & (rows[:, 0] != cols[:, 0])


def certify_permutations():
    for bits in itertools.product(range(2), repeat=4):
        rows, cols = np.array([bits[:2]]), np.array([bits[2:]])
        for i, j in itertools.product(range(2), repeat=2):
            rr = rows[:, [i, 1-i]] ^ j
            cc = cols[:, [j, 1-j]] ^ i
            assert np.array_equal(cycle(rows, cols), cycle(rr, cc))


def agreement_summary(pred, baseline):
    gain = int((pred & ~baseline).sum())
    loss = int((baseline & ~pred).sum())
    return dict(n=len(pred), correct=int(pred.sum()), baseline_correct=int(baseline.sum()),
                gain=gain, loss=loss, difference=(gain-loss)/len(pred))


def paired_ci(pred, baseline, source_ids):
    groups, idx = np.unique(source_ids, return_inverse=True)
    sizes = np.bincount(idx)
    changes = np.bincount(idx, weights=pred.astype(int)-baseline.astype(int))
    rng = np.random.default_rng(20261120)
    deltas = []
    for begin in range(0, 5000, 50):
        draws = rng.integers(0, len(groups), size=(min(50, 5000-begin), len(groups)))
        deltas.extend(changes[draws].sum(1)/sizes[draws].sum(1))
    return np.quantile(deltas, [.025, .975]).tolist()


def run(directory, protocol):
    certify_permutations()
    files = protocol['source']['files']
    meta = read_csv(directory/'source_photo_ids.csv')
    n = len(meta)
    cols = ['caption', 'negative_caption', 'type', 'subtype']
    keys = [tuple(row[k].casefold() for k in cols) for row in meta]
    source = np.array([row['filename'] for row in meta])
    edit = np.array([row['type'] for row in meta])
    folds = np.array([int.from_bytes(hashlib.sha256(('bivlc-direction-20261120:'+p).encode()).digest(), 'big')%5 for p in source])
    row_choices, col_choices, official = [], [], []
    lookup = np.arange(2)
    for name in files:
        data = read_csv(directory/name)
        assert [tuple(row[k].casefold() for k in cols) for row in data] == keys
        bits = np.array([[decode(r[key]) for key in ['text_score_i0','text_score_i1','image_score_c0','image_score_c1']] for r in data])
        r_ok, c_ok = bits[:,:2].all(1), bits[:,2:].all(1)
        for check, values in [('text_score', r_ok), ('image_score', c_ok), ('group_score', r_ok & c_ok)]:
            assert np.array_equal(values, [decode(row[check]) for row in data])
        row_choices.append(np.where(bits[:,:2],lookup,1-lookup))
        col_choices.append(np.where(bits[:,2:],lookup,1-lookup))
        official.append(r_ok & c_ok)
    pairs = list(itertools.product(range(len(files)), repeat=2))
    row_ok = [(x==lookup).all(1) for x in row_choices]
    col_ok = [(x==lookup).all(1) for x in col_choices]
    hits = {(a,b):row_ok[a] & col_ok[b] for a,b in pairs}
    veto = {(a,b):cycle(row_choices[a],col_choices[b]) for a,b in pairs}
    assert all(not veto[a,a].any() for a in range(len(files)))
    assert all(not (veto[k] & hits[k]).any() for k in pairs)
    outputs = {name:np.zeros(n,dtype=bool) for name in ['baseline','direction_raw','direction_veto','direction_cycle','primary','primary_raw','primary_cycle']}
    fold_records = []
    for fold in range(5):
        tr = folds!=fold
        te = ~tr
        def best_single(scores):
            return max(range(len(files)), key=lambda a:np.count_nonzero(scores[a][tr]))
        single = best_single(official)
        directions = best_single(row_ok), best_single(col_ok)
        def with_veto(k):
            return np.where(veto[k],official[single],hits[k])
        primary = max(pairs, key=lambda k:(with_veto(k)[tr].sum(),k[0]==k[1],-k[0],-k[1]))
        for name, p in [('baseline',official[single]),('direction_raw',hits[directions]),('direction_veto',with_veto(directions)),('direction_cycle',veto[directions]),('primary',with_veto(primary)),('primary_raw',hits[primary]),('primary_cycle',veto[primary])]:
            outputs[name][te] = p[te]
        fold_records.append(dict(fold=fold,n=int(te.sum()),sources=int(len(set(source[te]))),single=single,direction=list(directions),primary=list(primary)))
    base = outputs['baseline']
    primary = outputs['primary']
    ci = paired_ci(primary,base,source)
    perf = {k:agreement_summary(p,base) for k,p in outputs.items() if not k.endswith('cycle') and k!='baseline'}
    families = {k:agreement_summary(primary[edit==k],base[edit==k]) for k in sorted(set(edit))}
    all_singles = np.logical_or.reduce(official)
    all_pairs = np.logical_or.reduce(list(hits.values()))
    prevalence = {f'{a}_{b}':int(veto[a,b].sum()) for a,b in pairs if a!=b}
    return dict(models=files,input_sha256={name:digest(directory/name) for name in files + ['source_photo_ids.csv']},
                items=n,unique_source_images=int(len(set(source))),folds=fold_records,
                single_model_correct=[int(p.sum()) for p in official], baseline_correct=int(base.sum()),
                primary_paired_cluster_ci=ci,heldout_comparisons=perf,primary_by_edit=families,
                retained=bool(perf['primary']['difference']>=.02 and ci[0]>0 and sum(x['difference']>0 for x in families.values())>=2),
                veto_outcomes={k:dict(rejected=int(outputs[k+'_cycle'].sum()),rescued=int((outputs[k+'_cycle'] & base).sum())) for k in ['direction','primary']},
                pooled_ordered_off_diagonal_pairs=dict(events=sum(prevalence.values()),opportunities=len(prevalence)*n,by_pair=prevalence),
                oracle=dict(any_single=int(all_singles.sum()),any_directional_pair=int(all_pairs.sum()),extra_recombinations_only=int((all_pairs & ~all_singles).sum())),
                syntax_checks=['published four individual comparisons reconstruct all released group / direction values',
                               'four textual metadata fields and row order agree after casefolding in each file versus the pinned benchmark',
                               'all four label permutations preserve each of the sixteen possible cycle outputs',
                               'no within-model cycle and no cycle coexists with a correct group'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    protocol_path=Path(__file__).with_name('bivlc_cycle_protocol_20261120.json')
    result=run(args.scores,json.loads(protocol_path.read_text()))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
