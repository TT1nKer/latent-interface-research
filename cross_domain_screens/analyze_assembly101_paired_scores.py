"""Score both swapped source cues using the already-fixed full-stream predictors."""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from analyze_assembly101_forced_alignment import predict_pair
from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs
from analyze_assembly101_mark_swaps import count_swaps


def score(directory):
    fine,meta,_,coarse = load_inputs(directory)
    eligible,_ = identify_marked_occurrences(fine,coarse)
    events = {s:sorted(e for e in fine[s] if e[0]<e[1]) for s in coarse}
    tokens = {s:[e[2:] for e in rows] for s,rows in events.items()}
    indices = {s:{e:i for i,e in enumerate(rows)} for s,rows in events.items()}
    groups = defaultdict(list)
    n_pairs = 0
    for a,b in combinations(sorted(coarse),2):
        if meta[a][0]!=meta[b][0] or meta[a][1]==meta[b][1]:
            continue
        keys = eligible[a].keys() & eligible[b].keys()
        swaps = [(x,y) for x,y in combinations(sorted(keys),2) if x[:2]==y[:2]]
        n_pairs += bool(swaps)
        if not swaps:
            continue
        for src,tgt in ((a,b),(b,a)):
            marks = {k:indices[src][eligible[src][k][2]] for k in keys}
            goals = {k:indices[tgt][eligible[tgt][k][2]] for k in keys}
            predictions = predict_pair(tokens[src],tokens[tgt],sorted(marks.values()))
            for x,y in swaps:
                reversed_order = (marks[x]<marks[y])!=(goals[x]<goals[y])
                result={}
                for method in predictions[marks[x]]:
                    credit = Fraction(1)
                    for k in (x,y):
                        cand = predictions[marks[k]][method]
                        credit *= Fraction(1,len(cand)) if goals[k] in cand else 0
                    result[method]=credit
                groups['all'].append(result)
                groups['reversed' if reversed_order else 'same_order'].append(result)
    counts=count_swaps(directory)
    assert n_pairs==counts['physical_recording_pairs']
    assert len(groups['all'])==2*counts['unordered_mission_pairs']
    assert len(groups['reversed'])==2*counts['strict_disjoint_reversals']
    return dict(posthoc=True,unordered_recording_pairs=n_pairs,
                method='For the same fine action and same ordered recording pair, predict each of the two swapped source marks separately; only dual correct counts. Exact expectation for independent uniform tie-breaking per source query. Each unordered mission contributes both directions.',
                strata={key:dict(directed_swaps=len(rows),joint_accuracy={m:float(mean(r[m] for r in rows)) for m in rows[0]})
                        for key,rows in groups.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(score(parser.parse_args().directory),indent=2))
