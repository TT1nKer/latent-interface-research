"""Compare fixed single-mark rules at equal candidate multiplicity across datasets."""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from analyze_assembly101_forced_alignment import predict_pair
from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs
from analyze_assembly101_part_context import credit_and_chance, noun_vectors
from analyze_holoassist_occurrence_transfer import panel


def assembly(directory):
    fine,meta,_,coarse=load_inputs(directory)
    roles,_=identify_marked_occurrences(fine,coarse)
    events={s:sorted(e for e in fine[s] if e[0]<e[1]) for s in coarse}
    pairs=[(meta[a][0],a,b) for a,b in combinations(sorted(coarse),2)
           if meta[a][0]==meta[b][0] and meta[a][1]!=meta[b][1]]
    return pairs,events,roles


def cells(data):
    pairs,events,roles=data
    outcomes=defaultdict(list)
    for _,a,b in pairs:
        keys=roles[a].keys() & roles[b].keys()
        if not keys:
            continue
        for s,t in ((a,b),(b,a)):
            src,tgt=([e[2:] for e in events[n]] for n in (s,t))
            vs,vt=noun_vectors(src),noun_vectors(tgt)
            si={ev:i for i,ev in enumerate(events[s])}
            ti={ev:i for i,ev in enumerate(events[t])}
            anchors={k:si[roles[s][k][2]] for k in keys}
            predicted=predict_pair(src,tgt,anchors.values())
            for k,i in anchors.items():
                j=ti[roles[t][k][2]]
                part,chance=credit_and_chance(vs[i],vt,tgt,k[:2],j)
                best=predicted[i]['forced_edit']
                edit=Fraction(int(j in best),len(best))
                outcomes[chance.denominator].append((part,edit))
    return {n:dict(observations=len(rows),parts=float(mean(x for x,_ in rows)),
                   edit=float(mean(x for _,x in rows)),advantage=float(mean(x-y for x,y in rows)))
            for n,rows in sorted(outcomes.items())}


def compare(assembly_dir,holo_file):
    a=cells(assembly(assembly_dir))
    _,pairs,events,roles=panel(holo_file)
    h=cells((pairs,events,roles))
    common=sorted(set(a)&set(h))
    mass={k:min(a[k]['observations'],h[k]['observations']) for k in common}
    weight=sum(mass.values())
    estimates={d:sum(x[n]['advantage']*mass[n] for n in common)/weight for d,x in [('assembly',a),('holoassist',h)]}
    assert sum(x['observations'] for x in a.values())==146
    assert sum(x['observations'] for x in h.values())==710
    return dict(status='Post-hoc descriptive comparison of the already-seen external results. Compare exact equal numbers of candidate occurrences. Common weights are proportional to the smaller of the two observation counts per multiplicity; use all observations available in each cell to estimate its mean. No assumption of exchangeable tasks or cameras and no causal inference.',
                common_counts=common,mass_overlap=weight,common_weighted_parts_minus_edit=estimates,assembly=a,holoassist=h)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('assembly',type=Path)
    p.add_argument('holoassist',type=Path)
    a=p.parse_args()
    print(json.dumps(compare(a.assembly,a.holoassist),indent=2))
