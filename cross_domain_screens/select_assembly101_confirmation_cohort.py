"""Select the untouched Assembly101 confirmation cohort using only split metadata."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re


def recording_metadata(split):
    records = {}
    for line in split.read_text().splitlines():
        filename, _, toy, _ = line.split(maxsplit=3)
        seq = filename.removeprefix('assembly_').removesuffix('.txt')
        match = re.fullmatch(r'nusar-2021_action_both_([0-9]+)-([^-_]+)_([0-9]+)_user_id_.+', seq)
        assert match and match[2] == toy and seq not in records
        records[seq] = dict(toy=toy, participant=match[3], recording_prefix_person=match[1])
    return records


def select(discovery, train):
    old = recording_metadata(discovery)
    pool = recording_metadata(train)
    toys = {r['toy'] for r in old.values()}
    people = {r['participant'] for r in old.values()}
    selected = {s:r for s,r in sorted(pool.items()) if r['toy'] not in toys and r['participant'] not in people}
    possible_old = people | {r['recording_prefix_person'] for r in old.values()}
    assert all(not ({r['participant'],r['recording_prefix_person']} & possible_old) for r in selected.values())
    assert not set(selected) & set(old)
    pairs = [(a,b) for a,b in combinations(selected,2)
             if selected[a]['toy']==selected[b]['toy'] and selected[a]['participant']!=selected[b]['participant']]
    return dict(status='Frozen from official coarse train/discovery split metadata before opening training annotations or scores. Physical recording, exact toy IDs and participant IDs are disjoint; toy variants within the same toy category can remain related.',
                sources={'schema':'https://github.com/assembly-101/assembly101-annotations/tree/01651287eec9778bdf1de20e91e4b3162cccd770',
                         'official_drive':'https://drive.google.com/drive/folders/1QoT-hIiKUrSHMxYBKHvWpW9Z9aCznJB7',
                         'val_file_id':'1rXNhSvm87cUxCBybLJAqpfjI-ncvuDRD',
                         'train_file_id':'16zMkWiIyKjpDY19WQOceMGZpWYo-pRSJ',
                         'discovery_split_sha256':sha256(discovery.read_bytes()).hexdigest(),
                         'official_train_split_sha256':sha256(train.read_bytes()).hexdigest()},
                n_discovery=len(old), n_discovery_toys=len(toys),n_discovery_people=len(people), n_train=len(pool),
                n_disjoint=len(selected),n_toys=len({r['toy'] for r in selected.values()}),
                n_people=len({r['participant'] for r in selected.values()}),n_distinct_person_same_toy_pairs=len(pairs),
                n_toys_with_pairs=len({selected[a]['toy'] for a,b in pairs}), recordings=selected)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('discovery',type=Path)
    cli.add_argument('train',type=Path)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    args.out.write_text(json.dumps(select(args.discovery,args.train),indent=2)+'\n')
