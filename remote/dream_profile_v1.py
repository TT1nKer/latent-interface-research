"""16-item implementation/resource check using the unmodified author runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm GPU allocation required'
    import torch
    import yaml
    import reproduce_qasc as author
    assert torch.cuda.is_available()
    assert 'L4' in torch.cuda.get_device_name()
    base=Path('vendor/hidden_messages_f85b6d8')
    assert subprocess.check_output(['git','-C',str(base),'rev-parse','HEAD'],text=True).strip()=='f85b6d8563a2236117407e0df9c6cd60ea406dac'
    out=Path('results')/('dream_profile_v1_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    config=yaml.safe_load((base/'configs/qasc_original_evaluation.yaml').read_text())
    full=base/'data/qasc/validation.jsonl'
    assert digest(full)==config['qasc']['development_sha256']
    protocol=json.loads(Path('dream_trace_protocol_v1.json').read_text())
    selected=protocol['validity_example_ids']
    lookup={json.loads(line)['example_id']:line for line in full.read_text().splitlines()}
    data=out/'validity.jsonl'
    data.write_text('\n'.join(lookup[key] for key in selected)+'\n')
    lookup={r['example_id']:r for r in map(json.loads,(base/'evidence/qasc_predictions.jsonl').read_text().splitlines())}
    reference=out/'reference.jsonl'
    reference.write_text('\n'.join(json.dumps(lookup[key]) for key in selected)+'\n')
    config['qasc']['development_rows']=len(selected)
    config['qasc']['development_sha256']=digest(data)
    cfg=out/'validity_config.yaml'
    cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    args=type('Args',(),dict(model_cache=Path('hf_cache/hub'),output_dir=out/'author_run',
        config=cfg,data=data,reference=reference,checkpoint=base/'checkpoints/qasc_dense_latent_prefix_channel.pt',
        conditions=['final_only','matched','self','no_message'],expected_gpu='L4',record_messages=True))()
    author.parse_args=lambda:args
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    report=dict(job_id=os.environ['SLURM_JOB_ID'],stage='validity_and_resource_only',
        examples=len(selected),selected_ids=selected,source_full_development_sha256=digest(full),
        modified_settings='Only rows/reference and corresponding data hash/count; original prompt, model, channel and batch settings retained',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,local_models_loaded=False,
        protocol_sha256=digest('dream_trace_protocol_v1.json'),code_sha256=digest(__file__))
    try:
        author.main()
        report['completed']=True
    except BaseException as exc:
        report['completed']=False
        report['exception']=repr(exc)
        raise
    finally:
        torch.cuda.synchronize()
        report.update(elapsed_seconds=time.monotonic()-started,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (out/'resource.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
