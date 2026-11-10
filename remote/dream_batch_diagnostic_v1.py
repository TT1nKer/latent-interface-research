"""Diagnose failed release parity without changing the frozen mechanism gate.

Run the same 16 examples again, then restore complete original 64-example
blocks around the two discrepancies. This preserves both 8-example model
batches and 128-agent channel batches for self/no_message. No training.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
    import torch
    import yaml
    import reproduce_qasc as author

    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    base = Path('vendor/hidden_messages_f85b6d8')
    baseline = Path('results/dream_profile_v1_12446426')
    output = Path('results') / ('dream_batch_diagnostic_v1_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    config = author.load_config(base / 'configs/qasc_original_evaluation.yaml')
    data = base / 'data/qasc/validation.jsonl'
    assert sha(data) == config['qasc']['development_sha256']
    all_rows = read_rows(data)
    row_indices = {r['example_id']: i for i, r in enumerate(all_rows)}
    references = {r['example_id']: r for r in read_rows(base / 'evidence/qasc_predictions.jsonl')}
    selected = json.loads(Path('dream_trace_protocol_v1.json').read_text())['validity_example_ids']
    assert selected == [r['example_id'] for r in read_rows(baseline / 'validity.jsonl')]
    differences = json.loads((baseline / 'author_run/mismatches.json').read_text())
    assert {(r['example_id'], r['condition']) for r in differences} == {
        ('qasc-dist-be1160f33b3ab8fe784b', 'self'),
        ('qasc-dist-cef40c63d2d76e8c0891', 'no_message')}
    assert all(len(r['private_contexts']) == 2 for r in all_rows)
    model_batch = int(config['evaluation']['qasc_batch_examples'])
    channel_batch = int(config['training']['diagnostic_batch_size'])
    assert channel_batch % 2 == 0 and model_batch == 8
    block_size = math.lcm(model_batch, channel_batch // 2)
    starts = sorted({row_indices[r['example_id']] // block_size * block_size for r in differences})
    indices = [i for start in starts for i in range(start, start + block_size)]
    assert max(indices) < len(all_rows) and len(indices) == len(set(indices))
    # Every model/channel minibatch has the same neighbours and ordering as
    # the corresponding full-cohort batch, not just the same batch size.
    for width in (model_batch, channel_batch // 2):
        for start in range(0, len(indices), width):
            chunk = indices[start:start + width]
            assert chunk[0] % width == 0
            assert chunk == list(range(chunk[0], chunk[0] + width))
    cases = {'same16': [all_rows[row_indices[k]] for k in selected],
             'original_blocks128': [all_rows[i] for i in indices]}
    assert len(cases['original_blocks128']) == 128
    summary = dict(stage='batch_layout_diagnostic',job_id=os.environ['SLURM_JOB_ID'],
        source_data_sha256=sha(data),code_sha256=sha(__file__),
        protocol_sha256=sha('dream_trace_protocol_v1.json'),
        original_block_starts=starts,block_size=block_size,conditions=['self','no_message'],
        is_validity_receipt=False,full_trace_authorized=False,completed=False,
        interpretation='Targeted post-failure diagnostic. Does not change or pass the mechanism validity gate.',
        cases={})
    (output / 'plan.json').write_text(json.dumps(summary,indent=2)+'\n')
    baseline_predictions = {r['example_id']:r for r in read_rows(baseline / 'author_run/predictions.jsonl')}
    baseline_tokens = {r['example_id']:r['agent_token_ids'] for r in read_rows(baseline / 'author_run/token_rows.jsonl')}
    for name, rows in cases.items():
        case = output / name
        case.mkdir()
        case_data, reference = case / 'data.jsonl', case / 'reference.jsonl'
        case_data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        reference.write_text(''.join(json.dumps(references[r['example_id']])+'\n' for r in rows))
        case_config = author.load_config(base / 'configs/qasc_original_evaluation.yaml')
        case_config['qasc'].update(development_rows=len(rows),development_sha256=sha(case_data))
        cfg = case / 'config.yaml'
        cfg.write_text(yaml.safe_dump(case_config,sort_keys=False))
        args = SimpleNamespace(model_cache=Path('hf_cache/hub'),output_dir=case/'author_run',
            config=cfg,data=case_data,reference=reference,
            checkpoint=base/'checkpoints/qasc_dense_latent_prefix_channel.pt',
            conditions=['self','no_message'],expected_gpu='L4',record_messages=True)
        author.parse_args = lambda: args
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = dict(examples=len(rows),author_label_mismatch_exit=False)
        try:
            author.main()
        except SystemExit as exc:
            if str(exc) != 'Prediction mismatches recorded; this is not an exact reproduction':
                raise
            result['author_label_mismatch_exit'] = True
        result.update(elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        receipt = json.loads((case/'author_run/verification.json').read_text())
        result['conditions'] = receipt['conditions']
        predictions = {r['example_id']:r for r in read_rows(case/'author_run/predictions.jsonl')}
        tokens = {r['example_id']:r['agent_token_ids'] for r in read_rows(case/'author_run/token_rows.jsonl')}
        overlap = [k for k in selected if k in predictions]
        result['token_rows_equal_on_overlap'] = all(tokens[k] == baseline_tokens[k] for k in overlap)
        assert result['token_rows_equal_on_overlap'], 'Input token differences invalidate batch-only comparison'
        result['changed_from_baseline_on_overlap'] = {
            c:[k for k in overlap if predictions[k]['conditions'][c]['label'] != baseline_predictions[k]['conditions'][c]['label']]
            for c in args.conditions}
        result['targets'] = [{**m,'diagnostic_observed':predictions[m['example_id']]['conditions'][m['condition']]['label']}
                             for m in differences]
        summary['cases'][name] = result
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['completed'] = True
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
