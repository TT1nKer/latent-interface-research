"""Bounded LEAPS audit for one frozen cohort. No scoring or parameter fitting."""
import argparse
import bisect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

import h5py
import requests

from leaps_karel_source_format import convert


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


class Source:
    def __init__(self, protocol, folder):
        rev = protocol['dataset_revision']
        self.repo = f'https://huggingface.co/datasets/jesbu1/leaps_karel_dataset/resolve/{rev}/'
        self.size = protocol['hdf5_bytes']
        self.hdf5 = protocol['hdf5']
        budget = protocol['budget']
        self.cap_bytes = budget['source_transfer_bytes']
        self.end = time.monotonic() + budget['source_transfer_wall_s']
        self.start = time.monotonic()
        self.bytes_requested = 0
        self.bytes_received = 0
        self.requests = 0
        self.mutex = threading.Lock()
        self.ranges = []
        self.folder = folder
        self.fd = os.open(folder/'archive.sparse', os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, self.size)
        self.ledger = folder/'ranges.jsonl'

    def fetch(self, name, at, n):
        if n <= 0 or n > 4*1024*1024:
            raise ValueError('Invalid range size')
        with self.mutex:
            if self.bytes_requested + n > self.cap_bytes:
                raise RuntimeError(f'Predeclared byte cap reached at {self.bytes_requested} requested bytes')
            if time.monotonic() >= self.end:
                raise RuntimeError('Predeclared source-transfer time cap reached')
            self.bytes_requested += n
            self.requests += 1
        last = at + n - 1
        # A distinct URL avoids a cached CDN redirect signed for a different byte range.
        u = f'{self.repo}{name}?leapsrange={at}-{last}'
        with requests.get(u, headers={'Range':f'bytes={at}-{last}', 'Accept-Encoding':'identity'},
                          timeout=min(19,max(0.1,self.end - time.monotonic())), stream=True) as r:
            r.raise_for_status()
            expected = f'bytes {at}-{last}/'
            if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(expected):
                raise RuntimeError(f'Unbounded/mismatched response for {at}+{n}')
            data = r.raw.read(n)
            if len(data) != n or time.monotonic() > self.end:
                raise RuntimeError('Truncated or late source response')
        with self.mutex:
            self.bytes_received += len(data)
            with self.ledger.open('a') as log:
                log.write(json.dumps([name,at,n,sha_bytes(data)])+'\n')
        return data

    def save_raw(self, start, end):
        data = self.fetch(self.hdf5, start, end-start)
        if os.pwrite(self.fd, data, start) != len(data):
            raise OSError('Partial write')
        with self.mutex:
            bisect.insort(self.ranges, (start,end))

    def gaps(self, start, end):
        with self.mutex:
            available = self.ranges.copy()
        for a,b in available:
            if b <= start:
                continue
            if a >= end:
                break
            if a > start:
                yield start,min(a,end)
            start = max(start,b)
        if start < end:
            yield start,end

    def prefetch(self, spans):
        chunk = 512*1024
        tasks = []
        for a,b in spans:
            for start,end in self.gaps(a,b):
                tasks.extend((x,min(x+chunk,end)) for x in range(start,end,chunk))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(lambda part:self.save_raw(*part),tasks):
                pass

    def read(self, at, n):
        if not n:
            return b''
        if not (0 <= at <= at+n <= self.size):
            raise ValueError('Read outside release bounds')
        gaps = list(self.gaps(at,at+n))
        if gaps:
            align = 16384
            begin = gaps[0][0] // align * align
            end = min(self.size, (gaps[-1][1]+align-1)//align*align)
            self.prefetch([(begin,end)])
        return os.pread(self.fd,n,at)


class RemoteFile(io.RawIOBase):
    def __init__(self, source):
        self.source, self.pos = source,0

    def seek(self, pos, whence=0):
        self.pos = (0,self.pos,self.source.size)[whence] + pos
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buffer):
        data = self.source.read(self.pos,len(buffer))
        buffer[:] = data
        self.pos += len(data)
        return len(data)


def run(protocol_path, development, folder, report):
    protocol = json.loads(protocol_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'.gitignore').write_text('*\n')
    report.parent.mkdir(parents=True, exist_ok=True)
    info = {'completed':False,'native_validity_passed':False,'protocol_sha256':sha_file(protocol_path),
            'dataset_revision':protocol['dataset_revision'],'author_code_revision':protocol['author_code_revision'],
            'hdf5_release_sha256':'197046171b9002f3116b3174cdd1185d6ecff688cb0ee8f887c21d94389591ad',
            'code_sha256':{f:sha_file(Path(__file__).with_name(f)) for f in
                          (Path(__file__).name,'leaps_karel_source_format.py','karel_return_pairs.py','karel_predicates.py','karel_trace_control.py')},
            'h5py':h5py.__version__, 'development_source_sha256':sha_file(development)}
    source = Source(protocol,folder)
    counts = Counter()
    try:
        manifest = source.fetch(protocol['manifest'],0,protocol['id_bytes'])
        if hashlib.sha1(f'blob {len(manifest)}\0'.encode()+manifest).hexdigest() != '20c2d04169950629e2ced8a33f44078d9e88858e':
            raise ValueError('Manifest does not match pinned Git object')
        info['manifest_sha256'] = sha_bytes(manifest)
        all_ids = manifest.decode().splitlines()
        if len(all_ids) != protocol['source_programs'] or len(set(all_ids)) != len(all_ids):
            raise ValueError('Invalid global ID manifest')
        starts = protocol['sampling']['block_starts']; length = protocol['sampling']['block_length']
        indices = [i for s in starts for i in range(s,s+length)]
        selected = [all_ids[i] for i in indices]
        info['sample_indices_sha256'] = sha_bytes(('\n'.join(selected)+'\n').encode())
        devhash = {json.loads(line)['code'] for line in development.read_text().splitlines()}
        if len(devhash) != 2045:
            raise ValueError('Unexpected development code bank')
        seen = set()
        records = folder/'cohort.jsonl.partial'
        with h5py.File(RemoteFile(source),'r') as bank, records.open('x') as dest:
            # Release serial IDs were allocated in order. Anchor each predeclared
            # window at actual HDF5 group addresses; all nonlocal metadata reads
            # still go through the same range and time budget.
            def addr(i):
                return h5py.h5o.get_info(bank[all_ids[i]].id).addr
            spans = []
            for s in starts:
                a,b = addr(s),addr(s+length)
                if a >= b:
                    raise ValueError('Nonmonotonic cohort anchor addresses')
                spans.append((max(0,a-65536), min(source.size,b+65536)))
            info['prefetch_spans'] = spans
            source.prefetch(spans)
            for i,guid in zip(indices,selected):
                row = convert(bank[guid],i,guid,counts)
                code = sha_bytes(' '.join(row['program_tokens']).encode())
                duplicate, overlap = code in seen,code in devhash
                seen.add(code)
                exclusion = 'dev_overlap' if overlap else ('duplicate' if duplicate else None)
                counts.update({'selected_programs':1,'eligible_programs':int(exclusion is None),
                               'overlap_development':int(overlap),'duplicate_codes_within_sample':int(duplicate)})
                row.update({'eligible':exclusion is None,'exclusion':exclusion,'code':code})
                dest.write(json.dumps(row,separators=(',',':'))+'\n')
        if counts['selected_programs'] != len(indices) or counts['validated_stop'] != len(indices)*6:
            raise ValueError('Partial cohort validation')
        artifact = folder/'cohort.jsonl'; records.replace(artifact)
        info.update({'completed':True, 'native_validity_passed':True,'artifact_sha256':sha_file(artifact),
                     'artifact_bytes':artifact.stat().st_size, 'distinct_codes':len(seen)})
    except Exception as exc:
        info['error'] = repr(exc)
        raise
    finally:
        info.update(dict(counts))
        info.update({'requests':source.requests,'bytes_requested':source.bytes_requested,
                     'bytes_received':source.bytes_received,'wall_s':round(time.monotonic()-source.start,3)})
        if source.ledger.exists():
            info['range_ledger_sha256'] = sha_file(source.ledger)
        os.close(source.fd)
        write(report,info)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ('protocol','development','folder','report'):
        cli.add_argument(field,type=Path)
    args = cli.parse_args()
    run(args.protocol, args.development, args.folder, args.report)
