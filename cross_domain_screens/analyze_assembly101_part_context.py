"""Evaluate the registered eight-event noun context on frozen Assembly101 occurrences."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs


def noun_vectors(tokens):
    vectors = []
    for i, (_, anchor_noun) in enumerate(tokens):
        window = tokens[max(0, i-8):i] + tokens[i+1:i+9]
        counts = Counter(noun for _, noun in window if noun != anchor_noun)
        vectors.append((counts, sum(count * count for count in counts.values())))
    return vectors


def cosine_squared(left, right):
    a, aa = left
    b, bb = right
    if aa == 0 or bb == 0:
        return Fraction(0)
    dot = sum(count * b[noun] for noun, count in a.items())
    return Fraction(dot * dot, aa * bb)


def credit_and_chance(source, target, target_tokens, action, truth):
    candidates = [i for i, token in enumerate(target_tokens) if token == action]
    assert truth in candidates
    scores = [cosine_squared(source, target[i]) for i in candidates]
    best = max(scores)
    tied = scores.count(best)
    hit = scores[candidates.index(truth)] == best
    return Fraction(int(hit), tied), Fraction(1, len(candidates))


def summarize(rows):
    credit = sum(r[0] for r in rows)
    chance = sum(r[1] for r in rows)
    return dict(n=len(rows), credit_sum=str(credit), accuracy=float(credit/len(rows)),
                unique_correct=sum(r[0]==1 for r in rows), any_credit=sum(r[0]>0 for r in rows),
                uniform_credit_sum=str(chance), uniform_accuracy=float(chance/len(rows)))


def score(directory):
    root = Path(__file__).resolve().parent
    comparison = json.loads((root/'assembly101_alignment_results.json').read_text())
    reference = json.loads((root/'assembly101_coverage.json').read_text())
    inputs=[directory/'validation.csv',directory/'val_coarse_assembly.txt',*sorted((directory/'coarse').glob('assembly_*.txt'))]
    hashes={p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest=sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if manifest!=reference['inputs']['input_manifest_sha256'] or len(hashes)!=reference['inputs']['input_file_count']:
        raise ValueError('Different source annotation bundle')
    protocol = root/'assembly101_part_context_20261120.json'
    fine,meta,_,coarse = load_inputs(directory)
    eligible,_ = identify_marked_occurrences(fine,coarse)
    events = {s:sorted(e for e in fine[s] if e[0]<e[1]) for s in coarse}
    tokens = {s:[e[2:] for e in seq] for s,seq in events.items()}
    vectors = {s:noun_vectors(seq) for s,seq in tokens.items()}
    indices = {s:{e:i for i,e in enumerate(seq)} for s,seq in events.items()}
    singles, swaps = [], defaultdict(list)
    unique_reversal_toys = set()
    eligible_toys,eligible_pairs = set(),set()
    for a,b in combinations(sorted(coarse),2):
        if meta[a][0]!=meta[b][0] or meta[a][1]==meta[b][1]:
            continue
        keys = sorted(eligible[a].keys() & eligible[b].keys())
        for s,t in ((a,b),(b,a)):
            single = {}
            for key in keys:
                i = indices[s][eligible[s][key][2]]
                j = indices[t][eligible[t][key][2]]
                assert tokens[s][i]==tokens[t][j]==key[:2]
                single[key] = credit_and_chance(vectors[s][i],vectors[t],tokens[t],tokens[s][i],j)
                singles.append(single[key])
            for x,y in combinations(keys,2):
                if x[:2]!=y[:2]:
                    continue
                eligible_pairs.add((a,b))
                eligible_toys.add(meta[a][0])
                sx,sy = eligible[s][x][2],eligible[s][y][2]
                tx,ty = eligible[t][x][2],eligible[t][y][2]
                reverse = (sx<sy)!=(tx<ty)
                if reverse:
                    for left,right in [sorted((sx,sy)),sorted((tx,ty))]:
                        assert left[1]<right[0]
                # Each source mark is queried independently, including its uniform tie break.
                row = (single[x][0]*single[y][0], single[x][1]*single[y][1])
                swaps['all'].append(row)
                swaps['reversed' if reverse else 'same_order'].append(row)
                if reverse and row[0]==1:
                    unique_reversal_toys.add(meta[a][0])
    single_result = summarize(singles)
    swap_results = {k:summarize(v) for k,v in swaps.items()}
    assert len(singles)==212 and len(swaps['all'])==60 and len(swaps['reversed'])==26
    assert len(eligible_pairs)==16 and len(eligible_toys)==8
    exact_forced = Fraction(comparison['results']['forced_edit']['fractional_credit_sum'])
    exceeds_edit = Fraction(single_result['credit_sum'])>exact_forced
    return dict(protocol_commit='d3dac4e',protocol_sha256=sha256(protocol.read_bytes()).hexdigest(),
                verified_input_files=len(hashes),
                single=single_result, existing_forced_edit_single_top1=float(exact_forced/len(singles)),
                swaps=swap_results,unordered_swap_recording_pairs=len(eligible_pairs),swap_toys=len(eligible_toys),
                distinct_toys_with_unique_correct_reversed_dual_cues=len(unique_reversal_toys),
                unique_reversal_toys_sorted=sorted(unique_reversal_toys),exceeds_single_edit_comparator=exceeds_edit,
                advance=exceeds_edit and len(unique_reversal_toys)>=2,
                scope='Exploratory retrospective localization on the same panel; complete, labeled fine-action transcript including future events.')


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory',type=Path)
    cli.add_argument('--out',type=Path,required=True)
    args = cli.parse_args()
    args.out.write_text(json.dumps(score(args.directory),indent=2)+'\n')
