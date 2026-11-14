"""Acquire pinned inputs and extract a fixed timing-development cohort on a CPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Allocated CPU preparation only'
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    lock = json.loads(Path('vla_latency_assets_v1.json').read_text())
    start = time.time()
    roots = {'fastvla': Path('vendor/fastvla_31644f9'), 'native': Path('vendor/openvla_oft_e4287e9')}
    for key, dest in roots.items():
        spec = lock[key]
        if not dest.exists():
            subprocess.run(['git', 'init', str(dest)], check=True)
            subprocess.run(['git', '-C', str(dest), 'fetch', '--depth=1',
                            'https://github.com/' + spec['repo'] + '.git', spec['revision']], check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        head = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        assert head == spec['revision'], (key, head)
        assert not subprocess.check_output(['git', '-C', str(dest), 'status', '--porcelain', '--untracked-files=no'])
        # Imports only; never instantiate models during environment preparation.
        env = dict(os.environ, PYTHONPATH=str(dest.resolve()))
        subprocess.run([sys.executable, '-c',
                        'import tensorflow as tf; tf.config.set_visible_devices([], "GPU"); '
                        'from experiments.robot import openvla_utils; '
                        'from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction; '
                        'print("CPU_IMPORT_OK")'], env=env, check=True)

    from huggingface_hub import snapshot_download, hf_hub_download
    import h5py
    import numpy as np
    model = lock['model']
    model_path = Path(snapshot_download(repo_id=model['repo'], revision=model['revision'], max_workers=2))
    hashes = {}
    for item in model['files']:
        p = model_path / item['path']
        hashes[item['path']] = sha(p)
        if item['lfs_sha256']:
            assert hashes[item['path']] == item['lfs_sha256'], item['path']
    data_dir = Path('vla_assets_v1/cohort')
    data_dir.mkdir(exist_ok=True)
    episodes = []
    for index, spec in enumerate(lock['data']['files']):
        path = Path(hf_hub_download(repo_id=lock['data']['repo'], repo_type='dataset',
                                   revision=lock['data']['revision'], filename=spec['rfilename']))
        digest = sha(path)
        assert digest == spec['lfs']['sha256']
        with h5py.File(path, 'r') as f:
            data = f['data']
            convention = data.attrs['macros_image_convention']
            if isinstance(convention, bytes): convention = convention.decode()
            assert convention == 'opengl', ('unreviewed image convention', convention)
            info = json.loads(data.attrs['problem_info'])
            obs = data['demo_0/obs']
            n = len(obs['agentview_rgb'])
            frames = list(range(0, min(n, 192), 8))
            assert len(frames) >= 8, ('too short for history', n)
            camera = obs['agentview_rgb'][frames][:, ::-1, ::-1].copy()
            wrist = obs['eye_in_hand_rgb'][frames][:, ::-1, ::-1].copy()
            state = np.concatenate([obs['ee_states'][frames], obs['gripper_states'][frames]], axis=1)
            assert state.shape == (len(frames), 8)
            dest = data_dir / f'episode_{index}.npz'
            np.savez(dest, full_image=camera, wrist_image=wrist, state=state)
            episodes.append({'episode': index, 'file': str(dest), 'sha256': sha(dest),
                             'source_file': spec['rfilename'], 'source_sha256': digest,
                             'source_episode': 'demo_0', 'frames': frames, 'raw_image_shape': list(camera.shape[1:]),
                             'task': info['language_instruction'], 'image_convention': convention,
                             'transform': '180 degree rotation as official LIBERO evaluator; no resizing yet'})
    cohort = {'schema': 'vla_timing_cohort_v1', 'episodes': episodes,
              'selection': lock['data']['selection'], 'asset_lock_sha256': sha('vla_latency_assets_v1.json'),
              'role': 'Public demonstration replay for engineering feasibility; no task-success measurement.'}
    save('vla_assets_v1/cohort.json', cohort)
    save(out / 'summary.json', {'completed': True, 'model_path': str(model_path), 'model_sha256': hashes,
                              'cohort_path': 'vla_assets_v1/cohort.json',
                              'cohort_sha256': sha('vla_assets_v1/cohort.json'),
                              'asset_lock_sha256': sha('vla_latency_assets_v1.json'),
                              'sources': {k: lock[k] for k in roots},
                              'elapsed_seconds': time.time() - start,
                              'model_loaded': False, 'scientific_result': None})


if __name__ == '__main__':
    main()
