"""External symbolic repeated-event test using the released HoloAssist annotations."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from analyze_assembly101_confirmation import primary_outcome, rates
from analyze_assembly101_forced_alignment import predict_pair
from analyze_assembly101_occurrences import identify_marked_occurrences
from analyze_assembly101_part_context import credit_and_chance, noun_vectors

PROTOCOL = Path(__file__).with_name('holoassist_occurrence_transfer_20261120.json')


def records(source):
    """Incrementally decode array elements, retaining at most one incomplete record."""
    decode=json.JSONDecoder().raw_decode
    with source.open(encoding='utf-8-sig') as f:
        buf=f.read(65536).lstrip()
        if not buf.startswith('['):
            raise ValueError('Expected a JSON array')
        buf=buf[1:]
        while True:
            buf=buf.lstrip()
            if buf.startswith(']'):
                if buf[1:].strip() or f.read().strip():
                    raise ValueError('Unexpected trailing content')
                return
            if not buf:
                buf=f.read(65536)
                if not buf:
                    raise ValueError('Unexpected EOF')
                continue
            if buf[0]==',':
                buf=buf[1:]
                continue
            try:
                row,end=decode(buf)
            except json.JSONDecodeError:
                more=f.read(65536)
                if not more or len(buf)>16_000_000:
                    raise
                buf+=more
                continue
            yield row
            buf=buf[end:]


def text_token(value):
    if not isinstance(value,str):
        return ''
    value=' '.join(value.lower().split())
    return '' if value in ('none','n/a','unknown') else value


def compact(row):
    fine,coarse=set(),[]
    for e in row['events']:
        kind=e.get('label')
        if kind not in ('Fine grained action','Coarse grained action'):
            continue
        fields=e.get('attributes') or {}
        verb,noun=text_token(fields.get('Verb')),text_token(fields.get('Noun'))
        if not verb or not noun or e.get('start') is None or e.get('end') is None:
            continue
        start,end=float(e['start']),float(e['end'])
        if not 0<=start<end:
            continue
        if kind=='Fine grained action':
            fine.add((start,end,verb,noun))
        else:
            coarse.append((start,end,(verb,noun)))
    return text_token(row['taskType']),tuple(sorted(fine)),tuple(sorted(coarse))


def panel(source):
    size=source.stat().st_size
    if size!=json.loads(PROTOCOL.read_text())['source_size_bytes']:
        raise ValueError('Different release')
    available,conflicts,duplicates={},set(),Counter()
    raw=0
    for r in records(source):
        raw+=1
        name=r.get('video_name')
        if not name:
            raise ValueError('Missing recording ID')
        if name in conflicts:
            continue
        data=compact(r)
        if name not in available:
            available[name]=data
        elif available[name]==data:
            duplicates[name]+=1
        else:
            conflicts.add(name)
            del available[name]
    groups=defaultdict(list)
    for name,(group,_,_) in available.items():
        if group:
            groups[group].append(name)
    pairs=[]
    for group,names in sorted(groups.items()):
        names.sort(key=lambda name:sha256(('holoassist-event-v1|'+name).encode()).digest())
        chosen=names[:40]
        pairs.extend((group,a,b) for a,b in zip(chosen[::2],chosen[1::2]))
    ids={n for _,a,b in pairs for n in (a,b)}
    if len(ids)!=len(pairs)*2:
        raise ValueError('Physical recording reused')
    streams={n:available[n][1] for n in ids}
    intervals={n:available[n][2] for n in ids}
    roles,_=identify_marked_occurrences(streams,intervals)
    import hashlib
    with source.open('rb') as f:
        file_sha=hashlib.file_digest(f,'sha256').hexdigest()
    diag=dict(raw_rows=raw,identical_repeats=sum(duplicates.values()),conflicting_recording_ids=len(conflicts),
              available_valid_task_recordings=sum(map(len,groups.values())),available_tasks=len(groups),
              selected_pairs=len(pairs),selected_tasks=len({g for g,_,_ in pairs}),
              selected_distinct_recordings=len(ids),fine_events=sum(map(len,streams.values())),coarse_intervals=sum(map(len,intervals.values())),
              source_sha256=file_sha,sampling_sha256=sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest(),
              design_sha256=sha256(PROTOCOL.read_bytes()).hexdigest())
    return diag,pairs,streams,roles


def assess(source):
    diag,pairs,events,roles=panel(source)
    singles,reverse,same=[],[],[]
    reverse_tasks=defaultdict(list)
    single_tasks=defaultdict(list)
    reverse_pairs=set()
    excluded_overlapping=0
    with_unique=with_swaps=0
    for task,a,b in pairs:
        keys=sorted(roles[a].keys() & roles[b].keys())
        if not keys:
            continue
        with_unique+=1
        swaps=[(x,y) for x,y in combinations(keys,2) if x[:2]==y[:2]]
        with_swaps+=bool(swaps)
        tokens={n:[e[2:] for e in events[n]] for n in (a,b)}
        idx={n:{e:i for i,e in enumerate(events[n])} for n in (a,b)}
        vectors={n:noun_vectors(tokens[n]) for n in (a,b)}
        for src,tgt in ((a,b),(b,a)):
            marks={k:idx[src][roles[src][k][2]] for k in keys}
            gold={k:idx[tgt][roles[tgt][k][2]] for k in keys}
            preds=predict_pair(tokens[src],tokens[tgt],marks.values())
            score={}
            for k,i in marks.items():
                j=gold[k]
                assert tokens[src][i]==tokens[tgt][j]==k[:2]
                parts,uniform=credit_and_chance(vectors[src][i],vectors[tgt],tokens[tgt],k[:2],j)
                best=preds[i]['forced_edit']
                score[k]=dict(parts=parts,edit=Fraction(int(j in best),len(best)),uniform=uniform)
                singles.append(score[k])
                single_tasks[task].append(score[k])
            for x,y in swaps:
                val={m:score[x][m]*score[y][m] for m in score[x]}
                if (marks[x]<marks[y])==(gold[x]<gold[y]):
                    same.append(val)
                else:
                    spans=[sorted((roles[n][x][2],roles[n][y][2])) for n in (src,tgt)]
                    if any(early[1]>=late[0] for early,late in spans):
                        excluded_overlapping+=1
                        continue
                    reverse.append(val)
                    reverse_pairs.add((a,b))
                    reverse_tasks[task].append(val)
    primary=primary_outcome(reverse_tasks,reverse_pairs)
    primary={k.replace('toy','activity').replace('activitys','activities'):v for k,v in primary.items()}
    return dict(source=diag,physical_pairs_with_unique_role=with_unique,physical_pairs_with_two_marks=with_swaps,
                primary=primary,strict_reversed=dict(directed_n=len(reverse),rates=rates(reverse)),same_order=dict(directed_n=len(same),rates=rates(same)),
                excluded_overlapping_reversed_directed=excluded_overlapping,single=dict(n=len(singles),rates=rates(singles)),
                single_by_activity={t:dict(n=len(r),rates=rates(r)) for t,r in sorted(single_tasks.items())},
                reversal_by_activity={t:dict(directed_n=len(r),rates=rates(r)) for t,r in sorted(reverse_tasks.items())},
                limits='Separate dataset and unreused physical recordings; performer identity disjointness is unverified. Roles are exact structured verb/noun annotations and some label vocabularies are sparse. Complete human action transcripts including their future are supplied. No pixel inference or assisted task execution.')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('input',type=Path)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    result=assess(args.input)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
