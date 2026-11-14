"""Small receipt fixtures verify fail-closed event-chain behavior; no ML imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from remote.vla_latency_prepare_gate_v1 import verify


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prep = 'results/vla_prepare_v1_123/summary.json'
        self.bundle = 'vla_latency_chain_bundle_v1.json'
        self.write('containers/python310_3a93854a.sif', 'fixture image')
        self.write('logs/vla-prepare-v1-123.out', 'CPU_IMPORT_OK\nCPU_IMPORT_OK\n')
        self.write('results/vla_prepare_v1_123/container_source.txt', 'docker://python@fixture_digest\n')
        self.write('results/vla_prepare_v1_123/container_sha256.txt', sha(self.root/'containers/python310_3a93854a.sif')+'  image\n')
        self.write('results/vla_prepare_v1_123/freeze.txt', 'transformers@fixture_transformers\ndlimp@fixture_dlimp\n')
        self.write('results/vla_prepare_v1_123/install_report.json', {'environment': {'python_full_version': '3.10.14'},
            'install': [{'metadata': {'name': n, 'version': v}} for n,v in [('torch','2.2.0'),('numpy','1.26.4'),('tensorflow','2.15.0'),('tokenizers','0.19.1')]]})
        lock = {'container': {'platform': [{'digest': 'fixture_digest'}]}, 'model': {'revision': 'fixture_model',
            'files': [{'path':'weights', 'lfs_sha256': 'fixture_weights'}]},
            'native': {'revision': 'fixture_source'}, 'fastvla': {'revision': 'fixture_source'},
            'transformers': {'revision': 'fixture_transformers'}, 'dlimp': {'revision': 'fixture_dlimp'},
            'data': {'selection': 'fixture selection', 'files': [{'rfilename': str(i), 'lfs': {'sha256': str(i)}} for i in range(3)]}}
        self.write('vla_latency_assets_v1.json', lock)
        lock_hash = sha(self.root/'vla_latency_assets_v1.json')
        episodes = []
        for i in range(3):
            self.write(f'episode_{i}.npz', 'fixture episode '+str(i))
            episodes.append({'episode':i,'source_episode':'demo_0','source_file':str(i),'source_sha256':str(i),
                'file':f'episode_{i}.npz','sha256':sha(self.root/f'episode_{i}.npz'), 'image_convention':'opengl', 'frames':list(range(0,64,8))})
        self.write('cohort.json', {'asset_lock_sha256':lock_hash,'selection':'fixture selection','episodes':episodes})
        self.write(self.prep, {'completed':True,'model_loaded':False,'asset_lock_sha256':lock_hash,
            'sources':{k:lock[k] for k in ['native','fastvla']},'model_path':'/fake/fixture_model',
            'model_sha256':{'weights':'fixture_weights'},'cohort_path':'cohort.json','cohort_sha256':sha(self.root/'cohort.json')})
        self.write('worker.py', '# fixture only\n')
        self.write(self.bundle, {'sha256':{'worker.py':sha(self.root/'worker.py')}})

    def write(self, name, value):
        p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(value if isinstance(value,str) else json.dumps(value))

    def run_gate(self):
        with patch('remote.vla_latency_prepare_gate_v1.subprocess.check_output', side_effect=lambda cmd, **kw: 'fixture_source\n' if 'rev-parse' in cmd else b''):
            return verify(self.root,self.prep,'123',self.bundle)

    def test_complete_receipt_passes(self): self.assertEqual(self.run_gate()['observations'],24)

    def test_missing_second_import_stops(self):
        self.write('logs/vla-prepare-v1-123.out','CPU_IMPORT_OK\n')
        with self.assertRaises(AssertionError): self.run_gate()

    def test_changed_code_stops(self):
        self.write('worker.py','# changed\n')
        with self.assertRaises(AssertionError): self.run_gate()

    def test_corrupt_cohort_stops(self):
        self.write('episode_0.npz','corruption')
        with self.assertRaises(AssertionError): self.run_gate()

    def test_incomplete_preparation_stops(self):
        p=self.root/self.prep; x=json.loads(p.read_text()); x['completed']=False; self.write(self.prep,x)
        with self.assertRaises(AssertionError): self.run_gate()


if __name__ == '__main__': unittest.main()
