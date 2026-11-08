"""Pinned Dream acquisition in CPU Slurm; no tensor/model imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
    from huggingface_hub import HfApi, snapshot_download
    repo = 'Dream-org/Dream-v0-Instruct-7B'
    revision = '05334cb9faaf763692dcf9d8737c642be2b2a6ae'
    expected_manifest = 'd97cfa4b2e3a247447039efb60b29e626d7199cc036a1a53d62184a2ef08a2ab'
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    names = [s.rfilename for s in info.siblings
             if s.rfilename.endswith(('.safetensors','.json','.jinja','.model','.txt'))]
    chosen = [s for s in info.siblings if s.rfilename in names]
    required = sum(s.size or 0 for s in chosen)
    assert 10 * 1024**3 < required < 24 * 1024**3, required
    assert shutil.disk_usage('.').free > required * 2
    started = time.monotonic()
    print(json.dumps(dict(stage='download',repo=repo,revision=revision,bytes=required)),flush=True)
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=names,
                cache_dir='hf_cache/hub', max_workers=1))
    records = []
    for source in chosen:
        p = path/source.rfilename
        digest = sha(p)
        assert p.stat().st_size == source.size, p
        if source.lfs:
            expected = source.lfs.get('sha256') if isinstance(source.lfs,dict) else source.lfs.sha256
            assert digest == expected, p
        records.append(dict(path=source.rfilename,bytes=p.stat().st_size,sha256=digest))
    weights = sorted([r for r in records if r['path'].endswith('.safetensors')],key=lambda r:r['path'])
    manifest = hashlib.sha256(json.dumps(weights,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert manifest == expected_manifest, (manifest,expected_manifest)
    report = dict(job_id=os.environ['SLURM_JOB_ID'],passed=True,repo=repo,revision=revision,
        snapshot=str(path.resolve()),files=records,weight_manifest_sha256=manifest,
        elapsed_seconds=time.monotonic()-started,models_loaded=False)
    Path('results/dream_acquisition_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='verified',manifest=manifest,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    main()
