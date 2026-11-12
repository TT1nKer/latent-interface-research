"""Bounded diagnosis of full-v2 prediction mismatch using saved messages.

Does not repair, replace, or pass the failed v2 validity receipt. No training.
All 1000 questions retain their original ordering and token batches.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm allocation required'
    import torch
    import reproduce_qasc as author
    import phase8_qasc_hiddenbench_dense_prefix as original
    from hidden_messages.utils.checksums import sha256_tensor

    baseline = Path('results/dream_trace_full_v2_12448809')
    prior = json.loads((baseline/'summary.json').read_text())
    assert sha(baseline/'summary.json') == '1ef32e9d36269dafdbbdf1a800fd088cb6a0a589300090e4cfef8dc3252f0be3'
    assert prior['completed'] and prior['examples'] == 1000
    assert prior['numerical_validity_passed'] is False
    assert all(c['message_hashes_equal'] for c in prior['independent_author_checks']
               if c['condition'] in ('matched', 'self'))
    base = Path('vendor/hidden_messages_f85b6d8')
    revision = subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == prior['revision']
    subprocess.run(['git', '-C', str(base), 'diff', '--quiet', 'HEAD'], check=True)
    for filename, key in [('dream_recurrent_trace_v2.py', 'code_sha256'),
                          ('dream_trace_protocol_v1.json', 'protocol_sha256'),
                          ('dream_trace_amendment_v2.json', 'amendment_sha256')]:
        assert sha(filename) == prior[key]
    for filename, key in [('configs/qasc_original_evaluation.yaml', 'config_sha256'),
                          ('data/qasc/validation.jsonl', 'data_sha256'),
                          ('checkpoints/qasc_dense_latent_prefix_channel.pt', 'checkpoint_sha256'),
                          ('evidence/qasc_predictions.jsonl', 'reference_sha256')]:
        assert sha(base/filename) == prior[key]
    assert torch.cuda.is_available() and torch.cuda.get_device_name() == prior['gpu']
    assert torch.__version__ == prior['torch'] and torch.version.cuda == prior['cuda']
    flags = dict(bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 float32_matmul_precision=torch.get_float32_matmul_precision(),
                 matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    assert flags == prior['numerical_flags']
    output = Path('results') / ('dream_readout_diagnostic_v1_' + os.environ['SLURM_JOB_ID'])
    output.mkdir(exist_ok=False)
    summary = dict(job_id=os.environ['SLURM_JOB_ID'], stage='readout_diagnostic',
                   completed=False, is_validity_receipt=False, scientific_claim=False,
                   baseline_summary_sha256=sha(baseline/'summary.json'),
                   code_sha256=sha(__file__), numerical_flags=flags, examples=1000,
                   upstream_revision=revision, state_file_sha256={}, comparisons={})
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    start = time.monotonic()
    config = author.load_config(base/'configs/qasc_original_evaluation.yaml')
    rows = author.load_qasc_rows(base/'data/qasc/validation.jsonl', expected_rows=1000,
                                 expected_split='validation')
    ids = [r.example_id for r in rows]
    mc = config['model']
    snapshot = Path('hf_cache/hub')/('models--'+mc['id'].replace('/', '--'))/'snapshots'/mc['revision']
    manifest, _ = author.model_weight_manifest(snapshot)
    assert manifest == prior['model_weight_manifest_sha256'] == mc['weight_manifest_sha256']
    author.seed_everything(int(config['training']['seed']))
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model = author.dllm.utils.get_model(model_args=author.dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot), dtype=mc['dtype'],
        attn_implementation=mc['requested_attention_implementation'])).eval()
    model.requires_grad_(False)
    tokenizer = author.dllm.utils.get_tokenizer(model_args=author.dllm.utils.ModelArguments(
        model_name_or_path=str(snapshot), dtype=mc['dtype'],
        attn_implementation=mc['requested_attention_implementation']))
    decoder, embedding, head = model.get_decoder(), model.get_input_embeddings(), model.get_output_embeddings()
    cohort = author.tokenize_cohort(rows, benchmark='qasc', tokenizer=tokenizer, config=config)
    tokens = [json.loads(line) for line in (baseline/'token_rows.jsonl').read_text().splitlines()]
    assert [r['example_id'] for r in tokens] == ids
    assert [[list(t) for t in pair] for pair in cohort.token_rows] == [r['agent_token_ids'] for r in tokens]
    cc = config['channel']
    channel = author.DenseLatentPrefixChannel(int(model.config.hidden_size), len(config['surface']['qasc_labels']),
        input_embedding_rms=author.embedding_rms(embedding), message_slots=int(config['surface']['message_slots']),
        message_dim=int(cc['message_dim']), width=int(cc['width']), classifier_width=int(cc['classifier_width'])).to(device)
    channel.load_state_dict(torch.load(base/'checkpoints/qasc_dense_latent_prefix_channel.pt',
                                      map_location=device, weights_only=True), strict=True)
    channel.eval().requires_grad_(False)
    common = (model, decoder, embedding, head, tokenizer, cohort)
    batch = int(config['evaluation']['qasc_batch_examples'])
    diagnostic_batch = int(config['training']['diagnostic_batch_size'])
    label_ids = torch.tensor(cohort.label_ids, device=device)
    states = {}
    def state(name):
        if name not in states:
            path = baseline/(name+'.pt')
            value = torch.load(path, map_location='cpu', weights_only=True)
            assert value['example_ids'] == ids and value['agent_order'] == [0, 1]
            states[name] = value
            summary['state_file_sha256'][name] = sha(path)
        return states[name]

    @torch.inference_mode()
    def project(messages):
        return original._project_all(channel, messages.reshape(-1, channel.message_slots, channel.message_dim),
                                     batch_size=diagnostic_batch, device=device)

    @torch.inference_mode()
    def forward(prefixes=None, receiver_only=True, full_model=False):
        captured = []
        cursor = 0
        token_rows = [pair[0] for pair in cohort.token_rows] if receiver_only else original.flatten_tokens(cohort)
        def capture(_module, _inputs, logits):
            nonlocal cursor
            if logits.ndim == 3:
                lengths = torch.tensor([len(t) for t in token_rows[cursor:cursor+len(logits)]], device=logits.device)
                positions = lengths - 1 + int(mc['output_position_offset'])
                logits = logits[torch.arange(len(logits), device=logits.device), positions]
            assert logits.ndim == 2
            cursor += len(logits)
            captured.append(logits.index_select(-1, label_ids).float().cpu())
        hook = head.register_forward_hook(capture)
        try:
            hidden, predictions = original._forward_cohort(*common, batch_examples=batch,
                output_position_offset=int(mc['output_position_offset']), prefixes=prefixes,
                receiver_only=receiver_only, reference_full_model=full_model, device=device)
        finally:
            hook.remove()
        logits = torch.cat(captured)
        assert cursor == len(token_rows) and torch.equal(logits.argmax(-1), predictions)
        assert torch.isfinite(logits).all()
        return hidden, predictions, logits

    per_item = {key: dict(example_id=key, comparisons={}) for key in ids}
    def compare(name, a, b, logits_a=None, logits_b=None):
        changed = (a != b).nonzero().flatten().tolist()
        result = dict(equal=int((a == b).sum()), total=len(ids), mismatch_ids=[ids[i] for i in changed])
        if logits_a is not None and logits_b is not None:
            result['max_abs_label_logit_difference'] = float((logits_a-logits_b).abs().max())
        summary['comparisons'][name] = result
        for i, key in enumerate(ids):
            entry = dict(left_prediction=int(a[i]), right_prediction=int(b[i]))
            if logits_a is not None: entry['left_logits'] = logits_a[i].tolist()
            if logits_b is not None: entry['right_logits'] = logits_b[i].tolist()
            per_item[key]['comparisons'][name] = entry
        summary['elapsed_seconds'] = time.monotonic()-start
        save()
        print(json.dumps(dict(comparison=name, equal=result['equal'], total=len(ids),
                              elapsed_seconds=summary['elapsed_seconds'])), flush=True)

    with torch.inference_mode():
        endpoints = {}
        for route, depth in [('cross', 1), ('cross', 4), ('self', 1), ('self', 4)]:
            previous = state('native' if depth == 1 else f'{route}_{depth-1}')
            incoming = original._incoming_messages(previous['messages'], condition='matched' if route == 'cross' else 'self')
            prefixes = project(incoming).reshape(1000, 2, channel.message_slots, channel.model_dim)
            hidden, predictions, logits = forward(prefixes[:, 0])
            target = state(f'{route}_{depth}')
            compare(f'{route}{depth}_receiver_vs_saved_joint', predictions, target['predictions'][:, 0],
                    logits, target['label_logits'][:, 0])
            endpoints[(route, depth)] = (predictions, logits)
            if route == 'cross' and depth == 1:
                h2, p2, l2 = forward(prefixes, receiver_only=False)
                compare('cross1_joint_replay_vs_saved', p2.reshape(1000, 2)[:, 0], target['predictions'][:, 0],
                        l2.reshape(1000, 2, -1)[:, 0], target['label_logits'][:, 0])
                summary['joint_replay_hidden_exact'] = torch.equal(h2.reshape(1000, 2, -1), target['selected_hidden'])
                receiver_prefix = project(incoming[:, 0])
                summary['final_only_projection_equals_joint_receiver_slice'] = torch.equal(receiver_prefix, prefixes[:, 0])
                _, final_pred, final_logits = forward(receiver_prefix)
                compare('final_only_receiver_vs_saved_joint', final_pred, target['predictions'][:, 0],
                        final_logits, target['label_logits'][:, 0])
        native = state('native')
        _, native_pred, native_logits = forward()
        compare('native_receiver_decoder_vs_saved_joint', native_pred, native['predictions'][:, 0],
                native_logits, native['label_logits'][:, 0])
        _, reference_pred, reference_logits = forward(full_model=True)
        compare('native_receiver_full_vs_decoder', reference_pred, native_pred, reference_logits, native_logits)
        compare('native_receiver_full_vs_saved_joint', reference_pred, native['predictions'][:, 0],
                reference_logits, native['label_logits'][:, 0])
        # One fresh independent author evaluation checks the cached-message
        # reconstruction. This reruns one endpoint, not the recurrent trajectory.
        expected, metadata = author.evaluate_condition(channel, *common, condition='matched', exchange_rounds=1,
            derangement_order=author.qasc_derangement_order(rows, salt=config['evaluation']['derangement_salt']),
            batch_examples=batch, diagnostic_batch_size=diagnostic_batch,
            output_position_offset=int(mc['output_position_offset']),
            random_seed=int(config['evaluation']['random_message_seed']),
            no_message_receiver_batch_examples=8, device=device)
        summary['fresh_author_native_message_hash_equal'] = metadata['round_message_sha256'] == [sha256_tensor(native['messages'])]
        compare('cross1_cached_receiver_vs_fresh_author', endpoints[('cross', 1)][0], expected,
                endpoints[('cross', 1)][1])
    (output/'per_item.jsonl').write_text(''.join(json.dumps(per_item[key])+'\n' for key in ids))
    summary.update(completed=True, elapsed_seconds=time.monotonic()-start,
                   torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   scope='Failure localization only. Original v2 gate remains failed; no primary scientific statistic computed.')
    save()


if __name__ == '__main__':
    main()
