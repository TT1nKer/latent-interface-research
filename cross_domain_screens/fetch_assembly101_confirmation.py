"""Acquire the preselected physical recordings with a bounded annotation download."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import requests
from fetch_assembly101_occurrence_inputs import PublicFolder, COARSE_FOLDER, fetch
from select_assembly101_confirmation_cohort import select

TRAIN_FINE='1T2u7gkhoRc6xx5HlN1S7P--0faMFZ7vX'
TRAIN_SIZE=100677017
TRAIN_SPLIT='16zMkWiIyKjpDY19WQOceMGZpWYo-pRSJ'


def main(directory,discovery):
    directory.mkdir(parents=True,exist_ok=True)
    split=directory/'train_coarse_assembly.txt'
    if not split.exists():
        fetch((split,TRAIN_SPLIT,100_000))
    cohort=select(discovery/'val_coarse_assembly.txt',split)
    records=cohort.pop('recordings')
    assert (len(records),cohort['n_toys'],cohort['n_people'],cohort['n_distinct_person_same_toy_pairs'])==(44,24,11,21)
    subset=directory/'val_coarse_assembly.txt'
    names={'assembly_'+s+'.txt' for s in records}
    subset.write_text(''.join(line+'\n' for line in split.read_text().splitlines() if line.split()[0] in names))
    folder=PublicFolder()
    with requests.get('https://drive.google.com/embeddedfolderview',params={'id':COARSE_FOLDER},timeout=30,stream=True) as response:
        response.raise_for_status()
        response.raw.decode_content=True
        listing=response.raw.read(4_000_001)
    assert len(listing)<=4_000_000
    folder.feed(listing.decode())
    coarse=directory/'coarse'
    coarse.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,[(coarse/n,folder.index[n],100_000) for n in sorted(names)]))
    fine=directory/'official_train.csv'
    if not fine.exists():
        tmp=fine.with_suffix('.partial')
        with requests.get('https://drive.usercontent.google.com/download',params=dict(export='download',confirm='t',id=TRAIN_FINE),
                          stream=True,timeout=45) as response,tmp.open('wb') as dest:
            response.raise_for_status()
            if 'html' in response.headers.get('content-type','').lower():
                raise ValueError('Expected published CSV')
            n=0
            for buf in response.iter_content(chunk_size=512*1024):
                n+=len(buf)
                if n>TRAIN_SIZE:
                    raise ValueError('Unexpected train source size')
                dest.write(buf)
        if n!=TRAIN_SIZE:
            raise ValueError('Incomplete train source')
        tmp.replace(fine)
    assert fine.stat().st_size==TRAIN_SIZE
    membership=set()
    filtered=directory/'validation.csv'
    with fine.open(newline='') as inp,filtered.open('w',newline='') as out:
        rows=csv.DictReader(inp)
        writer=csv.DictWriter(out,fieldnames=rows.fieldnames)
        writer.writeheader()
        for r in rows:
            s=r['video'].split('/')[0]
            if s in records:
                if r['toy_id']!=records[s]['toy']:
                    raise ValueError('Fine and coarse toy identities differ')
                writer.writerow(r)
                membership.add(s)
    assert membership==set(records)
    inputs=[subset,filtered,*sorted(coarse.glob('assembly_*.txt'))]
    digests={p.relative_to(directory).as_posix():sha256(p.read_bytes()).hexdigest() for p in inputs}
    with fine.open('rb') as source:
        import hashlib
        cohort['train_fine_sha256']=hashlib.file_digest(source,'sha256').hexdigest()
    cohort.update(input_manifest_sha256=sha256(json.dumps(digests,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                  input_files=len(inputs),train_fine_bytes=TRAIN_SIZE,recording_prefix_ambiguity='Checking both possible participant IDs in inconsistent filename prefixes retains the same 44 recordings as the user-ID-only rule; no eligible ID overlaps either discovery identifier set.')
    # Include only data provenance and aggregate metadata in the public receipt.
    (directory/'acquisition.json').write_text(json.dumps(cohort,indent=2)+'\n')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory',type=Path)
    cli.add_argument('discovery',type=Path)
    a=cli.parse_args()
    main(a.directory,a.discovery)
