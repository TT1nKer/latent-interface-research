"""Check how much earlier and later target actions contribute to offline event ranking."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs
from analyze_assembly101_part_context import credit_and_chance, noun_vectors


def target_vectors(tokens, side):
    if side=='both':
        return noun_vectors(tokens)
    vectors=[]
    for i,(_,anchor) in enumerate(tokens):
        span=tokens[max(0,i-8):i] if side=='past' else tokens[i+1:i+9]
        counts=Counter(noun for _,noun in span if noun!=anchor)
        vectors.append((counts,sum(v*v for v in counts.values())))
    return vectors


def assess(directory):
    fine,meta,_,coarse=load_inputs(directory)
    eligible,_=identify_marked_occurrences(fine,coarse)
    events={s:sorted(x for x in fine[s] if x[0]<x[1]) for s in coarse}
    tokens={s:[e[2:] for e in items] for s,items in events.items()}
    pos={s:{e:i for i,e in enumerate(items)} for s,items in events.items()}
    modes=('both','past','future')
    vectors={m:{s:target_vectors(seq,m) for s,seq in tokens.items()} for m in modes}
    scores=defaultdict(list)
    for a,b in combinations(sorted(coarse),2):
        if meta[a][0]!=meta[b][0] or meta[a][1]==meta[b][1]:
            continue
        keys=sorted(eligible[a].keys() & eligible[b].keys())
        for s,t in ((a,b),(b,a)):
            one={}
            for key in keys:
                i=pos[s][eligible[s][key][2]]
                j=pos[t][eligible[t][key][2]]
                source=vectors['both'][s][i]
                one[key]={m:credit_and_chance(source,vectors[m][t],tokens[t],key[:2],j)[0] for m in modes}
                scores['single'].append(one[key])
            for x,y in combinations(keys,2):
                if x[:2]!=y[:2]:
                    continue
                sx,sy=eligible[s][x][2],eligible[s][y][2]
                tx,ty=eligible[t][x][2],eligible[t][y][2]
                reverse=(sx<sy)!=(tx<ty)
                if reverse and any(first[1]>=second[0] for first,second in (sorted((sx,sy)),sorted((tx,ty)))):
                    continue
                scores['reversed' if reverse else 'same_order'].append({m:one[x][m]*one[y][m] for m in modes})
    return {g:dict(n=len(rows),scores={m:float(mean(r[m] for r in rows)) for m in modes})
            for g,rows in scores.items()}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('discovery',type=Path)
    cli.add_argument('heldout',type=Path)
    a=cli.parse_args()
    result=dict(design='Post-hoc time-direction diagnostic on both previously scored panels. The source always uses the fixed eight events on either side. For the same target candidates compare both sides, the eight earlier actions only and the eight later actions only. An earlier action here means earlier in the start-time ordered transcript, not necessarily already completed (bimanual actions can overlap). Includes only gold action transcripts, ranks over all candidates offline, and does not detect when to issue a cue online. No new predictor fitted.',
                discovery=assess(a.discovery),heldout=assess(a.heldout))
    print(json.dumps(result,indent=2,allow_nan=False))
