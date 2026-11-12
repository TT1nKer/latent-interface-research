"""Test uncertainty-matched, class-preserving packet substitutions.

Uses existing native messages; no new recurrence or model training. Slurm only.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from dream_label_confidence_plan_v1 import ARMS, build_plan, digest, resource_projection
from dream_trace_gate_v2 import implementation_checks_pass


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Model work requires Slurm'
    import torch
    import reproduce_qasc as author
    import phase8_qasc_hiddenbench_dense_prefix as original

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    protocol = json.loads(Path('dream_label_confidence_protocol_v1.json').read_text())
    for name, expected in protocol['implementation_sha256'].items():
        assert sha(name) == expected, f'Protocol implementation mismatch: {name}'
    baseline_dir = Path('results/dream_readout_full_v3_12454951')
    assert sha(baseline_dir/'summary.json') == protocol['baseline_summary_sha256']
    assert sha(baseline_dir/'per_item.jsonl') == protocol['baseline_per_item_sha256']
    assert sha(baseline_dir/'token_rows.jsonl') == protocol['baseline_token_rows_sha256']
    baseline = json.loads((baseline_dir/'summary.json').read_text())
    assert baseline['completed'] and baseline['numerical_validity_passed'] is True
    assert implementation_checks_pass(baseline['independent_author_checks'])
    projected = resource_projection(baseline)
    old_rows = [json.loads(line) for line in (baseline_dir/'per_item.jsonl').read_text().splitlines()]
    ids = [row['example_id'] for row in old_rows]
    plan = build_plan(ids, [r['native']['message_logits'] for r in old_rows])
    assert digest(plan) == protocol['donor_plan_sha256']
    base = Path('vendor/hidden_messages_f85b6d8')
    revision = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == baseline['revision']
    subprocess.run(['git', '-C', str(base), 'diff', '--quiet', 'HEAD'], check=True)
    config_path = base/'configs/qasc_original_evaluation.yaml'
    config = author.load_config(config_path)
    data = base/'data/qasc/validation.jsonl'
    ckpt = base/'checkpoints/qasc_dense_latent_prefix_channel.pt'
    for path, key in [(config_path, 'config_sha256'), (data, 'data_sha256'), (ckpt, 'checkpoint_sha256')]:
        assert sha(path) == baseline[key]
    flags = dict(bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 float32_matmul_precision=torch.get_float32_matmul_precision(),
                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    assert flags == baseline['numerical_flags']
    assert torch.__version__ == baseline['torch'] and torch.version.cuda == baseline['cuda']
    native_path = Path('results/dream_trace_full_v2_12448809/native.pt')
    assert sha(native_path) == baseline['state_file_sha256']['native']
    started = time.monotonic()
    summary = dict(job_id=os.environ['SLURM_JOB_ID'], stage='label_confidence_v1', completed=False,
        examples=1000, arms=list(ARMS), receivers=[0, 1],
        code_sha256=sha(__file__), helper_sha256=sha('dream_label_confidence_plan_v1.py'),
        protocol_sha256=sha('dream_label_confidence_protocol_v1.json'), donor_plan_sha256=digest(plan),
        baseline_summary_sha256=sha(baseline_dir/'summary.json'),
        baseline_per_item_sha256=sha(baseline_dir/'per_item.jsonl'), native_state_sha256=sha(native_path),
        revision=revision, data_sha256=sha(data), checkpoint_sha256=sha(ckpt),
        config_sha256=sha(config_path), numerical_flags=flags, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, resource_projection_seconds=projected,
        implementation_validity_passed=False, model_training=False, recurrence_recomputed=False,
        gold_labels_used_to_select_interventions=False)
    (args.out/'started.json').write_text(json.dumps(summary, indent=2)+'\n')
    rows = author.load_qasc_rows(data, expected_rows=1000, expected_split='validation')
    assert [r.example_id for r in rows] == ids
    mc = config['model']
    snapshot = Path('hf_cache/hub')/('models--'+mc['id'].replace('/', '--'))/'snapshots'/mc['revision']
    manifest, _ = author.model_weight_manifest(snapshot)
    assert manifest == baseline['model_weight_manifest_sha256']
    summary['model_weight_manifest_sha256'] = manifest
    author.seed_everything(int(config['training']['seed']))
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model_args = author.dllm.utils.ModelArguments(model_name_or_path=str(snapshot),
        dtype=mc['dtype'], attn_implementation=mc['requested_attention_implementation'])
    model = author.dllm.utils.get_model(model_args=model_args).eval().requires_grad_(False)
    tokenizer = author.dllm.utils.get_tokenizer(model_args=model_args)
    decoder, embedding, head = model.get_decoder(), model.get_input_embeddings(), model.get_output_embeddings()
    cohort = author.tokenize_cohort(rows, benchmark='qasc', tokenizer=tokenizer, config=config)
    saved_tokens = {r['example_id']: r['agent_token_ids'] for r in
                    map(json.loads, (baseline_dir/'token_rows.jsonl').read_text().splitlines())}
    assert all([list(t) for t in pair] == saved_tokens[row.example_id]
               for row, pair in zip(rows, cohort.token_rows, strict=True))
    summary['original_token_rows_equal'] = True
    cc = config['channel']
    channel = author.DenseLatentPrefixChannel(int(model.config.hidden_size), 8,
        input_embedding_rms=author.embedding_rms(embedding), message_slots=int(config['surface']['message_slots']),
        message_dim=int(cc['message_dim']), width=int(cc['width']),
        classifier_width=int(cc['classifier_width'])).to(device)
    channel.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    channel.eval().requires_grad_(False)
    state = torch.load(native_path, map_location='cpu', weights_only=True)
    assert state['example_ids'] == ids and state['agent_order'] == [0, 1]
    messages = state['messages']
    assert messages.shape == (1000, 2, 4, 128)
    assert torch.equal(state['message_logits'], torch.tensor([r['native']['message_logits'] for r in old_rows]))
    diagnostic_batch = int(config['training']['diagnostic_batch_size'])
    label_ids = torch.tensor(cohort.label_ids, device=device)
    summary['class_counts_by_receiver'] = [x['class_counts'] for x in plan['receivers']]
    summary['matching_quality'] = [r['matching_quality'] for r in plan['receivers']]
    timings, results, checks = [], {a: [None, None] for a in ARMS}, []
    matched_prefixes = []

    @torch.inference_mode()
    def read(prefixes, receiver):
        current = cohort if receiver == 0 else replace(cohort,
                    token_rows=tuple((pair[1], pair[0]) for pair in cohort.token_rows))
        captured = []
        def capture(_module, _inputs, logits):
            assert logits.ndim == 2
            captured.append(logits.float().index_select(-1, label_ids).cpu())
        hook = head.register_forward_hook(capture)
        torch.cuda.synchronize()
        t = time.monotonic()
        try:
            _, pred = original._forward_cohort(model, decoder, embedding, head, tokenizer, current,
                batch_examples=int(config['evaluation']['qasc_batch_examples']),
                output_position_offset=int(mc['output_position_offset']), prefixes=prefixes,
                receiver_only=True, reference_full_model=False, device=device)
        finally:
            hook.remove()
        torch.cuda.synchronize()
        timings.append(time.monotonic()-t)
        logits = torch.cat(captured)
        assert logits.shape == (1000, 8) and torch.isfinite(logits).all()
        assert torch.equal(logits.argmax(-1), pred)
        return pred, logits

    @torch.inference_mode()
    def projection(value):
        return original._project_all(channel, value, batch_size=diagnostic_batch, device=device)

    with torch.inference_mode():
        # Validate both receiver-only paths before new scientific interventions.
        for receiver in (0, 1):
            sender = 1-receiver
            matched_prefixes.append(projection(messages[:, sender]))
            predicted, logits = read(matched_prefixes[receiver], receiver)
            expected = torch.tensor([r['cross'][0]['predictions'][receiver] for r in old_rows])
            expected_logits = torch.tensor([r['cross'][0]['label_logits'][receiver] for r in old_rows])
            check = dict(receiver=receiver, predictions_equal=torch.equal(predicted, expected),
                         logits_exact=torch.equal(logits, expected_logits),
                         max_abs_logit_delta=(logits-expected_logits).abs().max().item())
            checks.append(check)
            summary['matched_replay_checks'] = checks
            (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            assert check['predictions_equal'] and check['logits_exact'], check
            results['matched'][receiver] = (predicted, logits)
        assert len(checks) == 2
        summary['implementation_validity_passed'] = True
        summary['remaining_projection_seconds_with_margin'] = 1.5*4*max(timings)+60
        assert time.monotonic()-started+summary['remaining_projection_seconds_with_margin'] <= 900
        assert torch.cuda.max_memory_reserved() < 22*1024**3
        for arm in ARMS[1:]:
            for receiver in (0, 1):
                rplan = plan['receivers'][receiver]
                indices = torch.tensor(rplan['donors'][arm])
                prefixes = matched_prefixes[receiver].index_select(0, indices)
                results[arm][receiver] = read(prefixes, receiver)
                pred, logits = results[arm][receiver]
                # Save each arm immediately so a later failure cannot erase its trace.
                (args.out/f'{arm}_agent{receiver}.json').write_text(json.dumps(
                    dict(predictions=pred.tolist(), label_logits=logits.tolist()))+'\n')
                print(json.dumps(dict(arm=arm, receiver=receiver, seconds=timings[-1])), flush=True)
                del prefixes
    with (args.out/'per_item.jsonl').open('x') as handle:
        for i, row in enumerate(old_rows):
            conditions = {}
            for arm in ARMS:
                cells = []
                for receiver in (0, 1):
                    rp = plan['receivers'][receiver]
                    donor = rp['donors'][arm][i]
                    pred, logits = results[arm][receiver]
                    cells.append(dict(prediction=int(pred[i]), label_logits=logits[i].tolist(),
                        incoming_class=rp['sender_predicted_labels'][donor], donor_index=donor))
                conditions[arm] = cells
            handle.write(json.dumps(dict(example_id=row['example_id'], gold=row['gold'], conditions=conditions))+'\n')
    summary.update(completed=True, elapsed_seconds=time.monotonic()-started, readout_seconds=timings,
                   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
