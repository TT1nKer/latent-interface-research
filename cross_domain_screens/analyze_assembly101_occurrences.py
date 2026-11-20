"""Check occurrence annotation feasibility on Assembly101 validation without video."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import mean

COARSE_FILE = 'val_coarse_assembly.txt'


def load_inputs(directory):
    fine_by_seq = defaultdict(set)
    metadata = {}
    raw_counts = Counter()
    with (directory/'validation.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            seq = row['video'].split('/')[0]
            key = (int(row['start_frame']), int(row['end_frame']), row['verb_cls'], row['noun_cls'])
            fine_by_seq[seq].add(key)
            raw_counts[seq] += 1
            person = re.search(r'_(\d+)_user_id_',seq).group(1)
            info = row['toy_id'], person
            if seq in metadata and metadata[seq]!=info:
                raise ValueError('Inconsistent participant/toy mapping')
            metadata[seq] = info
    coarse = {}
    for line in (directory/COARSE_FILE).read_text().splitlines():
        fname = line.split()[0]
        seq = fname.removeprefix('assembly_').removesuffix('.txt')
        if seq not in fine_by_seq or seq in coarse:
            raise ValueError('Expected exactly one coarse assembly file per matched fine sequence')
        intervals = []
        for row in (directory/'coarse'/fname).read_text().splitlines():
            if not row.strip():
                continue
            s,e,label,*_ = row.rstrip().split('\t')
            clean=label.strip()
            if clean and clean.casefold() not in {'idle','background'}:
                intervals.append((int(s),int(e),clean))
        assert seq not in coarse
        coarse[seq] = intervals
    return fine_by_seq,metadata,raw_counts,coarse


def identify_marked_occurrences(fine, coarse):
    tally=Counter()
    eligible={}
    for seq,intervals in coarse.items():
        events=defaultdict(list)
        for s,e,v,n in sorted(fine[seq]):
            if s>=e:
                tally['zero_or_negative_duration']+=1
                continue
            containers=[role for left,right,role in intervals if left<s and e<right]
            if len(containers)==1:
                events[v,n].append((s,e,containers[0]))
                tally['strictly_contained']+=1
            elif not containers:
                tally['no_containing_foreground_role']+=1
            else:
                tally['multiple_containing_intervals']+=1
        eligible[seq]={}
        for action,occurrences in events.items():
            roles=[z[2] for z in occurrences]
            tally['observed_sequence_action_types']+=1
            if len(roles)>=2:
                tally['repeated_sequence_action_types']+=1
            if len(set(roles))<2:
                continue
            tally['repeated_under_different_roles']+=1
            tally['occurrences_of_such_actions']+=len(roles)
            for role,n in Counter(roles).items():
                if n==1:
                    idx=roles.index(role)
                    start,end,_=occurrences[idx]
                    eligible[seq][(*action,role)]=(idx,len(roles),(start,end,*action))
    return eligible,tally


def measure(directory):
    fine,meta,raw,coarse = load_inputs(directory)
    eligible,tally=identify_marked_occurrences(fine,coarse)
    by_pair=[]
    by_toy=defaultdict(list)
    role_counts=Counter()
    target_count=[]
    unit_sets=defaultdict(set)
    candidate_pairs=0
    for src,tgt in itertools.combinations(sorted(coarse),2):
        if meta[src][0]!=meta[tgt][0] or meta[src][1]==meta[tgt][1]:
            continue
        candidate_pairs+=1
        keys=eligible[src].keys() & eligible[tgt].keys()
        if not keys:
            continue
        pair=Counter()
        for key in sorted(keys):
            for a,b in [(src,tgt),(tgt,src)]:
                rank,n,_=eligible[a][key]
                truth,m,_=eligible[b][key]
                # Closed form objective for distances between normalized 0-based ranks.
                pred=min(range(m),key=lambda i:abs(i*(n-1)-rank*(m-1)))
                pair['ordered_item_count']+=1
                pair['same_ordinal_correct']+=rank==truth
                pair['normalized_ordinal_correct']+=pred==truth
                pair['exact_ordinal_out_of_range']+=rank>=m
                pair['chance_sum']+=1/m
                role_counts[key]+=1
                target_count.append(m)
                if pred!=truth:
                    unit_sets['normalized_fail_recordings'].update([src,tgt])
                    unit_sets['normalized_fail_toys'].add(meta[src][0])
                    unit_sets['normalized_fail_pairs'].add((src,tgt))
                if pred!=truth and rank!=truth:
                    pair['both_ordinal_baselines_wrong']+=1
                    unit_sets['both_fail_pairs'].add((src,tgt))
                    unit_sets['both_fail_toys'].add(meta[src][0])
        unit_sets['eligible_recordings'].update([src,tgt])
        unit_sets['eligible_participants'].update([meta[src][1],meta[tgt][1]])
        by_toy[meta[src][0]].append(pair)
        by_pair.append(pair)
    total=sum(by_pair,Counter())
    trials=total['ordered_item_count']
    candidates=list(role_counts)
    summ=dict(total)
    if trials:
        summ.update({k+'_fraction':total[k]/trials for k in ['same_ordinal_correct','normalized_ordinal_correct','chance_sum','both_ordinal_baselines_wrong']})
        summ['macro_over_unordered_pairs']={k:mean(p[k]/p['ordered_item_count'] for p in by_pair) for k in ['same_ordinal_correct','normalized_ordinal_correct','chance_sum']}
    sampled=sum(len(fine[s]) for s in coarse)
    inputs=[directory/'validation.csv',directory/COARSE_FILE,*sorted((directory/'coarse').glob('assembly_*.txt'))]
    return dict(protocol='assembly101_occurrence_screen_20261120.json',inputs=dict(sha256_per_local_file={p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},total_bytes=sum(p.stat().st_size for p in inputs)),
                sample=dict(official_fine_validation_physical_recordings=len(fine),official_coarse_assembly_validation_recordings=len(coarse),
                toy_ids=len({meta[s][0] for s in coarse}),participants=len({meta[s][1] for s in coarse}),
                fine_video_rows_in_matched_recordings=sum(raw[s] for s in coarse),deduplicated_fine_events_whole_recordings=sampled,
                recordings_with_a_cross_role_event=sum(bool(x) for x in eligible.values()),coarse_intervals=sum(map(len,coarse.values()))),
                containment=dict(tally),pairing=dict(same_toy_different_person_unordered_pairs_available=candidate_pairs,
                unordered_pairs_eligible=len(by_pair),toys_eligible=len(by_toy),distinct_eligible_fine_verbs_and_objects=len({k[:2] for k in candidates}),
                distinct_eligible_coarse_roles=len({k[2] for k in candidates}),target_occurrence_count_min_max=[min(target_count,default=0),max(target_count,default=0)],
                **{k:len(v) for k,v in unit_sets.items()}),rank_baselines=summ,
                caveat='Table of held-out annotations; foreground filtering and fine labels require annotation. Ordered items share recordings, toys and roles. Ordinal controls do not evaluate a learned model or online execution. Dataset by Sener et al. (CVPR 2022), annotations CC BY-NC 4.0; original analysis.')


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('input',type=Path)
    args.add_argument('--out',type=Path,required=True)
    config=args.parse_args()
    config.out.write_text(json.dumps(measure(config.input),indent=2)+'\n')
