"""Remote CPU-only import/config preflight. Does not instantiate neural weights."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
modules = ['torch','transformers','dllm','phase8_qasc_hiddenbench_dense_prefix',
           'reproduce_qasc']
for name in modules:
    importlib.import_module(name)
    print('imported',name,flush=True)
import torch
assert not torch.cuda.is_available()
import yaml
base=Path('vendor/hidden_messages_f85b6d8')
config=yaml.safe_load((base/'configs/qasc_original_evaluation.yaml').read_text())
report=dict(job_id=os.environ['SLURM_JOB_ID'],imports=modules,passed=True,
            models_loaded=False,packages={name:importlib.metadata.version(name) for name in
            ('torch','transformers','accelerate','numpy','peft','huggingface-hub')},
            surface_present='surface' in config)
Path('results/dream_import_preflight_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
