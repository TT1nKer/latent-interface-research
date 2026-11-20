"""Test full-stream deterministic control against the fixed real occurrence labels."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

from analyze_assembly101_occurrences import load_inputs, identify_marked_occurrences


def distance_table(xs,ys):
    table=[list(range(len(ys)+1))]
    for i,x in enumerate(xs,1):
        before=table[-1]
        row=[i]
        for j,y in enumerate(ys,1):
            row.append(min(row[-1]+1,before[j]+1,before[j-1]+int(x!=y)))
        table.append(row)
    return table


def min_indices(candidates,scores):
    assert len(candidates)==len(scores)>0
    best=min(scores)
    return [c for c,s in zip(candidates,scores) if s==best]


def predict_pair(source_tokens,target_tokens,marks):
    """Predict per marked source index using only action tokens and marks, never coarse roles."""
    forward=distance_table(source_tokens,target_tokens)
    backward=distance_table(source_tokens[::-1],target_tokens[::-1])
    indexed_source=defaultdict(list)
    indexed_target=defaultdict(list)
    for i,t in enumerate(source_tokens):
        indexed_source[t].append(i)
    for j,t in enumerate(target_tokens):
        indexed_target[t].append(j)
    predictions={}
    for i in marks:
        label=source_tokens[i]
        source_positions=indexed_source[label]
        js=indexed_target[label]
        ordinal=source_positions.index(i)
        n,m=len(source_positions),len(js)
        assert n>=2 and m>=2
        scores=[forward[i][j]+backward[len(source_tokens)-i-1][len(target_tokens)-j-1] for j in js]
        relative=[abs(k*(n-1)-ordinal*(m-1)) for k in range(m)]
        predictions[i]=dict(forced_edit=min_indices(js,scores),absolute_clamped=[js[min(ordinal,m-1)]],
                            absolute_strict=[js[ordinal]] if ordinal<m else [],
                            normalized=min_indices(js,relative),uniform=js)
    return predictions


def evaluate(directory):
    fine,meta,_,coarse=load_inputs(directory)
    eligible,_=identify_marked_occurrences(fine,coarse)
    events={s:sorted(e for e in fine[s] if e[0]<e[1]) for s in coarse}
    tokenized={s:[e[2:] for e in rows] for s,rows in events.items()}
    indices={s:{e:i for i,e in enumerate(rows)} for s,rows in events.items()}
    units=defaultdict(list)
    toys=set()
    role_examples=[]
    for a,b in combinations(sorted(coarse),2):
        if meta[a][0]!=meta[b][0] or meta[a][1]==meta[b][1]:
            continue
        keys=sorted(eligible[a].keys() & eligible[b].keys())
        if not keys:
            continue
        toys.add(meta[a][0])
        for src,tgt in [(a,b),(b,a)]:
            gold={indices[src][eligible[src][k][2]]:indices[tgt][eligible[tgt][k][2]] for k in keys}
            assert len(gold)==len(keys)
            predictions=predict_pair(tokenized[src],tokenized[tgt],sorted(gold))
            for i,truth in gold.items():
                assert tokenized[src][i]==tokenized[tgt][truth] and truth in predictions[i]['uniform']
                units[a,b].append({method:(truth in choices,len(choices)) for method,choices in predictions[i].items()})
            if not role_examples:
                k=keys[0]
                si=indices[src][eligible[src][k][2]]
                tj=indices[tgt][eligible[tgt][k][2]]
                sc=[idx for idx,v in enumerate(tokenized[src]) if v==k[:2]]
                tc=predictions[si]['uniform']
                role_examples.append(dict(selection='first eligible pair and role in lexical order',toy=meta[src][0],marked_fine=' '.join(k[:2]),coarse_role=k[2],
                    full_source_action_occurrence=[1+sc.index(si),len(sc)],full_target_action_occurrence=[1+tc.index(tj),len(tc)]))
    items=[record for pair_items in units.values() for record in pair_items]
    out={}
    for method in ['forced_edit','absolute_clamped','absolute_strict','normalized','uniform']:
        def credit(r):
            hit,n=r[method]
            return Fraction(1,n) if hit else Fraction()
        credits=sum(map(credit,items),Fraction())
        out[method]=dict(fractional_credit_sum=str(credits),fractional_top1=float(credits/len(items)),unique_correct=sum(hit and n==1 for hit,n in (r[method] for r in items)),
                         correct_in_minimizers=sum(r[method][0] for r in items),tie_events=sum(r[method][1]>1 for r in items),max_tie=max(r[method][1] for r in items),
                         no_prediction=sum(r[method][1]==0 for r in items),unordered_pair_macro_top1=float(mean(mean(credit(r) for r in records) for records in units.values())))
    return dict(unordered_pairs=len(units),toys=len(toys),ordered_items=len(items),full_target_candidate_range=[min(r['uniform'][1] for r in items),max(r['uniform'][1] for r in items)],
                results=out,example=role_examples[0],scope='Annotation-level retrospective control on entire fine action stream with future tokens; all target positions of the same fine label eligible.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory',type=Path)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    args.out.write_text(json.dumps(evaluate(args.directory),indent=2)+'\n')
