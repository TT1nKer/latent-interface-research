"""Pinned file acquisition only; runs in a CPU Slurm allocation, never loads tensors."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    from huggingface_hub import snapshot_download
    manifest = json.loads(Path('c2c_acquire_v1_manifest.json').read_text())
    required = sum(f['size'] for m in manifest['models'] for f in m['files'])
    assert shutil.disk_usage('.').free > required * 2
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'models': [], 'scope': 'Acquisition only'}
    start = time.monotonic()
    for m in manifest['models']:
        snapshot = Path(snapshot_download(m['repo'], revision=m['revision'],
            allow_patterns=[f['name'] for f in m['files']], max_workers=1))
        assert snapshot.name == m['revision']
        receipts = []
        for f in m['files']:
            p = snapshot / f['name']
            assert p.stat().st_size == f['size'], p
            with p.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if f['sha256']:
                assert digest == f['sha256'], p
            receipts.append({'name': f['name'], 'bytes': f['size'], 'sha256': digest})
        report['models'].append({'repo': m['repo'], 'revision': m['revision'],
                                  'snapshot': str(snapshot), 'files': receipts})
        print(m['repo'], 'verified', flush=True)
    report['elapsed_seconds'] = time.monotonic() - start
    report['passed'] = True
    Path('results/c2c_acquisition_v1.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
