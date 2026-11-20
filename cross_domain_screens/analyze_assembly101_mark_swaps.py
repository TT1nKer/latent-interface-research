"""Count real paired interventions that change only the marked source occurrence."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from analyze_assembly101_occurrences import identify_marked_occurrences, load_inputs


def count_swaps(directory):
    fine, meta, _, coarse = load_inputs(directory)
    eligible, _ = identify_marked_occurrences(fine, coarse)
    pairs = set()
    toys = set()
    participants = set()
    records = set()
    reversed_pairs = set()
    reversed_toys = set()
    strict_reverse_gaps = []
    first_reversal = None
    sizes = Counter()
    role_pairs = total = change_order = distinct_marked = 0
    for a, b in combinations(sorted(coarse), 2):
        if meta[a][0] != meta[b][0] or meta[a][1] == meta[b][1]:
            continue
        by_action = defaultdict(list)
        for key in eligible[a].keys() & eligible[b].keys():
            by_action[key[:2]].append(key)
        for _,keys in sorted(by_action.items()):
            n = len(keys)
            total += n*2
            sizes[n] += 1
            if n < 2:
                continue
            distinct_marked += n*2
            pairs.add((a,b))
            toys.add(meta[a][0])
            participants.update((meta[a][1],meta[b][1]))
            records.update((a,b))
            for x, y in combinations(sorted(keys), 2):
                ax,ay = eligible[a][x][2],eligible[a][y][2]
                bx,by = eligible[b][x][2],eligible[b][y][2]
                assert ax!=ay and bx!=by and ax[2:]==ay[2:]==bx[2:]==by[2:]
                role_pairs+=1
                if (ax<ay) != (bx<by):
                    change_order+=1
                    first_a, later_a = sorted((ax,ay))
                    first_b, later_b = sorted((bx,by))
                    gap = min(later_a[0]-first_a[1], later_b[0]-first_b[1])
                    if gap>0:
                        strict_reverse_gaps.append(gap)
                        if first_reversal is None:
                            a_roles = [k[2] for _,k in sorted(((ax,x),(ay,y)))]
                            b_roles = [k[2] for _,k in sorted(((bx,x),(by,y)))]
                            first_reversal = dict(toy=meta[a][0],action=list(x[:2]),source_roles=a_roles,
                                                  target_roles=b_roles,minimum_frame_gap=gap)
                        reversed_pairs.add((a,b))
                        reversed_toys.add(meta[a][0])
    assert total==212
    return dict(ordered_marks_original=total,ordered_marks_within_swappable_group=distinct_marked,
                eligible_groups_by_number_of_shared_roles=dict(sorted(sizes.items())),unordered_mission_pairs=role_pairs,
                physical_recording_pairs=len(pairs),physical_recordings=len(records),participants=len(participants),
                toys=len(toys),unordered_mission_pairs_with_reversed_rank=change_order,
                strict_disjoint_reversals=len(strict_reverse_gaps),strict_reversal_recording_pairs=len(reversed_pairs),
                strict_reversal_toys=len(reversed_toys),minimum_frame_gap_of_strict_reversals=min(strict_reverse_gaps,default=None),
                first_disjoint_reversal=first_reversal,
                max_pairs_exact_for_one_monotone_map=role_pairs-len(strict_reverse_gaps),
                definition='Post-hoc availability count. Fix the two recordings and one fine verb/object; two different source marks must require two different target occurrences, each drawn from a coarse role shared and uniquely annotated in both recordings. Each unordered pair of roles gives one swap before reversing the direction of transfer. No method outcomes used.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(count_swaps(parser.parse_args().directory),indent=2))
