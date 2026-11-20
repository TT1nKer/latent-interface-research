"""Recover ten native-checked worlds per program, with held-out program codes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
from leaps_karel_source import RemoteFile, Source, sha_bytes, sha_file, write
from leaps_karel_source_format import convert


def all_ten(group, index, guid):
    # Reuse the frozen six-world converter. The two overlapping windows cross-check
    # the world-index translation. Every stored state, action and STOP is verified.
    counts = Counter()
    first = convert(group, index, guid, counts)
    order = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    shifted = {k: group[k][:][order] for k in ('s_h', 'a_h', 's_h_len', 'a_h_len')}
    shifted['program'] = group['program']
    second = convert(shifted, index, guid, counts)
    for j, case in enumerate(second['examples']):
        case['example_index'] = j + 4
    if first['examples'][4:6] != second['examples'][:2] or counts['validated_stop'] != 12:
        raise ValueError('Source world IDs or native terminations disagree')
    first['examples'].extend(second['examples'][2:])
    return first


def main(a):
    cfg = json.loads(a.protocol.read_text())
    previous, previous_ids = set(), set()
    for path in a.prior:
        for line in path.open():
            record = json.loads(line)
            previous.add(record['code'])
            if 'source_index' in record:
                previous_ids.add(record['guid'])
    a.cache.mkdir(exist_ok=False, parents=True)
    (a.cache/'.gitignore').write_text('*\n')
    source = Source(cfg, a.cache)
    counts = Counter()
    seen = set()
    meta = dict(completed=False, protocol_sha256=sha_file(a.protocol), prior_sha256=[sha_file(p) for p in a.prior],
                worker_sha256=sha_file(Path(__file__)), converter_sha256=sha_file(Path(__file__).with_name('leaps_karel_source_format.py')))
    a.receipt.parent.mkdir(exist_ok=True, parents=True)
    try:
        manifest = source.fetch(cfg['manifest'], 0, cfg['id_bytes'])
        if sha_bytes(manifest) != '1f1f5bf28403079bda7df59a235ca48eaced99746c84d4263313de09c3374eeb':
            raise ValueError('Release IDs changed')
        ids = manifest.decode().splitlines()
        starts, length = cfg['sample']['starts'], cfg['sample']['length']
        path = a.cache/'cohort.jsonl'
        with h5py.File(RemoteFile(source),'r') as bank, path.open('x') as out:
            def address(i):
                return h5py.h5o.get_info(bank[ids[i]].id).addr
            spans = [(address(s)-65536,address(s+length)+65536) for s in starts]
            source.prefetch(spans)
            for start in starts:
                for i in range(start, start + length):
                    guid = ids[i]
                    if guid in previous_ids:
                        raise ValueError('Repeated external generation index')
                    row = all_ten(bank[guid], i, guid)
                    code = hashlib.sha256(' '.join(row['program_tokens']).encode()).hexdigest()
                    counts['native_programs'] += 1
                    counts['native_worlds'] += 10
                    counts['native_steps'] += sum(len(c['actions']) for c in row['examples'])
                    if code in previous or code in seen:
                        counts['excluded_codes'] += 1
                    else:
                        counts['eligible_programs'] += 1
                        out.write(json.dumps(dict(code=code,guid=guid,examples=row['examples']),separators=(',',':'))+'\n')
                    seen.add(code)
        if counts['native_programs'] != len(starts)*length:
            raise ValueError('Incomplete fixed source cohort')
        meta.update(completed=True,artifact_sha256=sha_file(path),native_states=counts['native_steps']+counts['native_worlds'])
    finally:
        os.close(source.fd)
        meta.update(counts, bytes_received=source.bytes_received, requests=source.requests)
        write(a.receipt,meta)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('protocol','cache','receipt'):
        p.add_argument('--'+option, type=Path,required=True)
    p.add_argument('--prior',nargs='+',type=Path,required=True)
    main(p.parse_args())
