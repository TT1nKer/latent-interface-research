"""Deterministic preparation checks before the existing bounded GPU replay.

This checks the exact evidence the successful preparation script produced; it
does not replace interpretation of novel failures or assert a scientific result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def verify(root, prepare, job, bundle):
    root = Path(root)
    assert str(job).isdigit()
    prepare = root / prepare
    assert prepare.parent.name == 'vla_prepare_v1_' + str(job)
    bundle = json.loads((root / bundle).read_text())
    for name, digest in bundle['sha256'].items():
        assert sha(root / name) == digest, ('changed submitted input', name)
    lock = json.loads((root / 'vla_latency_assets_v1.json').read_text())
    receipt = json.loads(prepare.read_text())
    assert receipt['completed'] is True and receipt['model_loaded'] is False
    assert receipt['asset_lock_sha256'] == sha(root / 'vla_latency_assets_v1.json')
    assert receipt['sources'] == {k: lock[k] for k in ('fastvla', 'native')}
    assert Path(receipt['model_path']).name == lock['model']['revision']
    expected_files = {r['path']: r for r in lock['model']['files']}
    assert set(receipt['model_sha256']) == set(expected_files)
    for name, info in expected_files.items():
        if info['lfs_sha256']:
            assert receipt['model_sha256'][name] == info['lfs_sha256']
    out_log = root / 'logs' / f'vla-prepare-v1-{job}.out'
    assert out_log.read_text().splitlines().count('CPU_IMPORT_OK') == 2
    source = (prepare.parent / 'container_source.txt').read_text().strip()
    assert source == 'docker://python@' + lock['container']['platform'][0]['digest']
    image_hash = (prepare.parent / 'container_sha256.txt').read_text().split()[0]
    assert sha(root / 'containers/python310_3a93854a.sif') == image_hash
    install = json.loads((prepare.parent / 'install_report.json').read_text())
    assert install['environment']['python_full_version'] == '3.10.14'
    installed = {r['metadata']['name'].lower().replace('_', '-'): r['metadata']['version'] for r in install['install']}
    for name, version in {'torch': '2.2.0', 'numpy': '1.26.4', 'tensorflow': '2.15.0', 'tokenizers': '0.19.1'}.items():
        assert installed[name].split('+')[0] == version, (name, installed.get(name))
    freeze = (prepare.parent / 'freeze.txt').read_text()
    for key in ('transformers', 'dlimp'): assert lock[key]['revision'] in freeze
    cohort_path = root / receipt['cohort_path']
    assert sha(cohort_path) == receipt['cohort_sha256']
    cohort = json.loads(cohort_path.read_text())
    assert cohort['asset_lock_sha256'] == receipt['asset_lock_sha256']
    assert cohort['selection'] == lock['data']['selection']
    assert len(cohort['episodes']) == len(lock['data']['files']) == 3
    total = 0
    for i, (ep, spec) in enumerate(zip(cohort['episodes'], lock['data']['files'])):
        assert ep['episode'] == i and ep['source_episode'] == 'demo_0'
        assert ep['source_file'] == spec['rfilename'] and ep['source_sha256'] == spec['lfs']['sha256']
        assert ep['image_convention'] == 'opengl'
        assert 8 <= len(ep['frames']) <= 24 and ep['frames'] == list(range(0, 8 * len(ep['frames']), 8))
        assert sha(root / ep['file']) == ep['sha256']
        total += len(ep['frames'])
    for key, rel in [('native', 'vendor/openvla_oft_e4287e9'), ('fastvla', 'vendor/fastvla_31644f9')]:
        repo = root / rel
        head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == lock[key]['revision']
        assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'])
    return {'passed': True, 'prepare_job': str(job), 'prepare_sha256': sha(prepare),
            'bundle_sha256': sha(root / 'vla_latency_chain_bundle_v1.json'),
            'cohort_sha256': receipt['cohort_sha256'], 'observations': total,
            'model_loaded': False, 'scientific_result': None}


def main():
    assert os.environ.get('SLURM_JOB_ID')
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', required=True)
    ap.add_argument('--prepare-job', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    result = verify(Path.cwd(), args.prepare, args.prepare_job, args.bundle)
    with Path(args.out).open('x') as f: json.dump(result, f, indent=2); f.write('\n')


if __name__ == '__main__': main()
