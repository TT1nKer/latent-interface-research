"""Recover native token lists from the already downloaded release ranges."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
from leaps_karel_source_format import VOCAB


def extract(folder, out):
    programs = {}
    with h5py.File(folder/'archive.sparse','r') as bank, (folder/'cohort.jsonl').open() as records:
        for line in records:
            row = json.loads(line)
            tokens = [VOCAB[int(t)] for t in bank[row['guid']]['program'][:]]
            code = hashlib.sha256(' '.join(tokens).encode()).hexdigest()
            if code != row['code'] or code in programs:
                raise ValueError('Native tokens must match the unique held-out code')
            programs[code] = tokens
    if len(programs)!=766:
        raise ValueError('Incomplete program registry')
    with out.open('x') as target:
        json.dump(programs,target,separators=(',',':'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    extract(args.folder,args.out)
