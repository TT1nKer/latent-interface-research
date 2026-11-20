"""Apply the fixed part and edit matchers to new toys and new participants."""
import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import random
from statistics import mean, quantiles

from analyze_assembly101_forced_alignment import predict_pair
from analyze_assembly101_part_context import credit_and_chance, noun_vectors
from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs

METHODS = ('parts', 'edit', 'uniform')


def rates(scores):
    return {m:float(mean(r[m] for r in scores)) for m in METHODS} if scores else {m:None for m in METHODS}


def primary_outcome(by_toy, physical_pairs):
    toy_means = {toy:{m:mean(r[m] for r in rows) for m in METHODS} for toy,rows in sorted(by_toy.items())}
    if not toy_means:
        return dict(toys=0, recording_pairs=0, meets_gate=False, status='No eligible disjoint reversal')
    vals = list(toy_means.values())
    gains = [float(row['parts']-row['edit']) for row in vals]
    rng = random.Random(20261120)
    draws = [mean(rng.choices(gains,k=len(gains))) for _ in range(10000)]
    q = quantiles(draws,n=40,method='inclusive')
    distinct_unique_parts = sum(any(r['parts']==1 for r in rows) for rows in by_toy.values())
    return dict(toys=len(vals),recording_pairs=len(physical_pairs),equal_toy_mean=rates(vals),
                parts_minus_edit=mean(gains),paired_toy_95=[q[0],q[-1]],
                win_tie_loss=[sum(g>0 for g in gains),sum(g==0 for g in gains),sum(g<0 for g in gains)],
                unique_dual_success_toys=distinct_unique_parts,
                meets_gate=len(vals)>=8 and len(physical_pairs)>=20 and mean(gains)>=.1 and q[0]>0)


def score(new_dir, discovery_dir):
    fine,meta,_,coarse = load_inputs(new_dir)
    _,earlier,_,discovery = load_inputs(discovery_dir)
    old_toys={earlier[s][0] for s in discovery}
    old_people={earlier[s][1] for s in discovery}
    if len(discovery)!=60 or len(coarse)!=44:
        raise ValueError('Fixed cohort differs from pre-score inventory')
    if any(s in discovery or meta[s][0] in old_toys or meta[s][1] in old_people for s in coarse):
        raise ValueError('Discovery ID leakage')
    eligible,_=identify_marked_occurrences(fine,coarse)
    seqs={s:sorted(event for event in fine[s] if event[0]<event[1]) for s in coarse}
    tokens={s:[event[2:] for event in seq] for s,seq in seqs.items()}
    indices={s:{ev:i for i,ev in enumerate(seq)} for s,seq in seqs.items()}
    vectors={s:noun_vectors(xs) for s,xs in tokens.items()}
    singles, reverse, preserved = [], [], []
    by_toy=defaultdict(list)
    single_toys=defaultdict(list)
    pairs, reversal_pairs, same_toy_pairs = set(), set(), 0
    overlap_count=0
    for a,b in combinations(sorted(coarse),2):
        if meta[a][0]!=meta[b][0] or meta[a][1]==meta[b][1]:
            continue
        same_toy_pairs+=1
        common = sorted(eligible[a].keys() & eligible[b].keys())
        if not common:
            continue
        pairs.add((a,b))
        swaps=[(x,y) for x,y in combinations(common,2) if x[:2]==y[:2]]
        for s,t in ((a,b),(b,a)):
            marks={k:indices[s][eligible[s][k][2]] for k in common}
            gold={k:indices[t][eligible[t][k][2]] for k in common}
            edit=predict_pair(tokens[s],tokens[t],marks.values())
            credits={}
            for key,i in marks.items():
                j=gold[key]
                assert tokens[s][i]==tokens[t][j]==key[:2]
                part,chance=credit_and_chance(vectors[s][i],vectors[t],tokens[t],key[:2],j)
                selected=edit[i]['forced_edit']
                credits[key]=dict(parts=part,edit=Fraction(int(j in selected),len(selected)),uniform=chance)
                singles.append(credits[key])
                single_toys[meta[s][0]].append(credits[key])
            for x,y in swaps:
                sa,sb=sorted((eligible[s][x][2],eligible[s][y][2]))
                ta,tb=sorted((eligible[t][x][2],eligible[t][y][2]))
                reverse_order=(marks[x]<marks[y])!=(gold[x]<gold[y])
                r={m:credits[x][m]*credits[y][m] for m in METHODS}
                if reverse_order:
                    if sa[1]>=sb[0] or ta[1]>=tb[0]:
                        overlap_count+=1
                        continue
                    by_toy[meta[s][0]].append(r)
                    reversal_pairs.add((a,b))
                    reverse.append(r)
                else:
                    preserved.append(r)
    if same_toy_pairs!=21:
        raise ValueError('Different set of same-toy, different-participant pairs')
    return dict(recordings=len(coarse),toys=len({meta[s][0] for s in coarse}),participants=len({meta[s][1] for s in coarse}),
                discovery_recording_overlap=0,discovery_toy_overlap=0,discovery_participant_overlap=0,
                distinct_person_same_toy_pairs=same_toy_pairs,pairs_with_unique_shared_role=len(pairs),
                primary=primary_outcome(by_toy,reversal_pairs),single=dict(n=len(singles),rates=rates(singles)),
                secondary_single_by_toy={toy:dict(n=len(rows),rates=rates(rows)) for toy,rows in sorted(single_toys.items())},
                reversed=dict(directed_n=len(reverse),rates=rates(reverse)),same_order=dict(directed_n=len(preserved),rates=rates(preserved)),
                reversed_overlap_excluded_directed=overlap_count,
                per_reversal_toy={toy:dict(directed_n=len(rows),rates=rates(rows)) for toy,rows in sorted(by_toy.items())},
                protocol_sha256=sha256(Path(__file__).with_name('assembly101_confirmation_20261120.json').read_bytes()).hexdigest(),
                evaluation='No fitting or window changes. Annotated full action streams, including future observations. Same dataset family; different official split, toy codes and participant IDs.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new',type=Path)
    parser.add_argument('discovery',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--single-by-toy',type=Path,help='Optional post-hoc single-cue breakdown')
    a=parser.parse_args()
    result=score(a.new,a.discovery)
    strata=result.pop('secondary_single_by_toy')
    with a.out.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write('\n')
    if a.single_by_toy:
        gains=[r['rates']['parts']-r['rates']['edit'] for r in strata.values()]
        detail=dict(posthoc=True,by_toy=strata,toys=len(gains),
                    positive_tied_negative=[sum(x>0 for x in gains),sum(x==0 for x in gains),sum(x<0 for x in gains)],
                    equal_toy_advantage=mean(gains) if gains else None)
        with a.single_by_toy.open('x') as output:
            json.dump(detail,output,indent=2,allow_nan=False)
            output.write('\n')
