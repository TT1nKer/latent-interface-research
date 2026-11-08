"""Development experiment; model execution requires Slurm GPU allocation."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from late_binding_tasks import make_case, validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Model execution requires a Slurm allocation'
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='late_binding_arc_dev32_v1.json')
    p.add_argument('--out', default='results/late_binding_c2c_dev32_v1')
    p.add_argument('--limit', type=int, default=32)
    args = p.parse_args()
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rosetta.model.projector import load_projector
    from rosetta.model.wrapper import RosettaModel
    assert torch.cuda.is_available(), 'GPU allocation required'
    torch.manual_seed(20261106)
    torch.set_num_threads(2)
    start = time.monotonic()
    acquisition = json.loads(Path('results/c2c_acquisition_v1.json').read_text())
    assert acquisition['passed']
    paths = {x['repo']: Path(x['snapshot']) for x in acquisition['models']}
    receiver_path = paths['Qwen/Qwen3-0.6B']
    sender_path = paths['Qwen/Qwen2.5-0.5B-Instruct']
    ckpt = paths['nics-efc/C2C_Fuser'] / 'qwen3_0.6b+qwen2.5_0.5b_Fuser/final'
    tokenizer = AutoTokenizer.from_pretrained(receiver_path, local_files_only=True)
    sender_tokenizer = AutoTokenizer.from_pretrained(sender_path, local_files_only=True)
    def load(path):
        return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    receiver, sender = load(receiver_path), load(sender_path)
    projectors = []
    files = sorted(ckpt.glob('projector_*.pt'), key=lambda f: int(f.stem.split('_')[-1]))
    assert len(files) == 28
    for i, f in enumerate(files):
        assert f.stem == f'projector_{i}'
        proj = load_projector(str(f.with_suffix('.json')))
        proj.load_state_dict(torch.load(f, map_location='cpu', weights_only=True), strict=True)
        projectors.append(proj)
    model = RosettaModel([receiver, sender], projector_list=projectors).to('cuda').eval()
    model.load_projector_config(str(ckpt / 'projector_config.json'))
    data = json.loads(Path(args.data).read_text())
    rows = data['rows'][:args.limit]
    validate(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.with_suffix('.jsonl').exists(), 'Refusing to overwrite previous evidence'
    token_ids = []
    for letter in 'ABCD':
        tokens = tokenizer.encode(' ' + letter, add_special_tokens=False)
        assert len(tokens) == 1
        token_ids.append(tokens[0])
    label_tensor = torch.tensor(token_ids, device='cuda')
    response_ids = tokenizer.encode('The correct answer is', add_special_tokens=False)
    meta = dict(job_id=os.environ['SLURM_JOB_ID'], software=dict(torch=torch.__version__,
        transformers=transformers.__version__, python=platform.python_version()),
        gpu=torch.cuda.get_device_name(), dtype='bfloat16', attention='sdpa',
        acquisition=acquisition, data_sha256=sha(args.data), code_sha256=sha(__file__),
        task_code_sha256=sha('late_binding_tasks.py'), label_token_ids=token_ids,
        scoring='Four single-token space-prefixed option labels; not free generation.',
        scope='Development only; receiver has question and all choices in every condition.',
        checks=[], loaded_projectors=len(files), seed=20261106)
    def encode(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids == sender_tokenizer.encode(text, add_special_tokens=False), 'Tokenizer mismatch'
        assert max(ids) < sender.config.vocab_size
        for i, letter in enumerate('ABCD'):
            assert tokenizer.encode(text + ' ' + letter, add_special_tokens=False) == ids + [token_ids[i]]
        return ids
    def score(ids, mode, boundary):
        x = torch.tensor([ids], device='cuda')
        mask = torch.ones_like(x)
        if mode == 'native':
            output = receiver(input_ids=x, attention_mask=mask, use_cache=True)
        else:
            assert ids[-len(response_ids):] == response_ids
            split = boundary if mode in ('late', 'chunked_native') else len(ids) - len(response_ids)
            assert 0 < split < len(ids)
            a = torch.zeros((1, split, 2), dtype=torch.long, device='cuda')
            a[:, :, 0] = -1 if mode == 'chunked_native' else 1
            b = torch.zeros((1, len(ids) - split, 2), dtype=torch.long, device='cuda')
            b[:, :, 0] = -1
            output = model(input_ids=x, attention_mask=mask, kv_cache_index=[a, b], use_cache=True)
        last = output.logits[0, -1].float()
        choice_logits = last[label_tensor]
        prob = choice_logits.softmax(-1)
        result = dict(pred=int(prob.argmax()), logits=choice_logits.tolist(), probs=prob.tolist(),
                      option_mass=float(last.softmax(-1)[label_tensor].sum()),
                      top_token=int(last.argmax()), top_text=tokenizer.decode([int(last.argmax())]))
        del output
        model.kv_cache_dict.clear()
        return result
    results = []
    with torch.inference_mode(), out.with_suffix('.jsonl').open('x') as stream:
        for n, row in enumerate(rows):
            base = make_case(row, 0)
            orig_ids = encode(base['original'])
            original = {m: score(orig_ids, m, len(orig_ids)-1) for m in ('native', 'early')}
            for shift in range(4):
                c = make_case(row, shift)
                full_text = c['prefix'] + c['suffix']
                ids = encode(full_text)
                prefix_ids = tokenizer.encode(c['prefix'], add_special_tokens=False)
                assert ids[:len(prefix_ids)] == prefix_ids, 'Prefix tokenization boundary changed'
                scores = {m: score(ids, m, len(prefix_ids)) for m in ('native', 'late', 'early')}
                if n < 2 and shift == 0:
                    chunked = score(ids, 'chunked_native', len(prefix_ids))
                    repeat = score(ids, 'late', len(prefix_ids))
                    delta = max(abs(x-y) for x,y in zip(chunked['probs'], scores['native']['probs']))
                    repeat_delta = max(abs(x-y) for x,y in zip(repeat['probs'], scores['late']['probs']))
                    check = dict(id=row['id'], chunked_max_probability_error=delta,
                        chunked_prediction_matches=chunked['pred'] == scores['native']['pred'],
                        repeated_fusion_max_probability_error=repeat_delta)
                    meta['checks'].append(check)
                    assert delta < 0.02 and check['chunked_prediction_matches'], check
                    assert repeat_delta < 1e-6, check
                record = dict(id=row['id'], shift=shift, correct=c['answer_index'],
                    original_correct=c['old_answer_index'], order=c['order'],
                    prefix_sha256=hashlib.sha256(c['prefix'].encode()).hexdigest(),
                    prefix_tokens=len(prefix_ids), prompt_tokens=len(ids),
                    prompt=full_text, original_prompt=base['original'], original=original, scores=scores)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                results.append(record)
            print(json.dumps(dict(done=n+1, total=len(rows), elapsed=time.monotonic()-start)), flush=True)
    def acc(records, mode):
        return sum(r['scores'][mode]['pred'] == r['correct'] for r in records) / len(records) if records else None
    summary = {}
    for subset, selected in [('all', results), ('identity', [r for r in results if r['shift']==0]),
                             ('changed', [r for r in results if r['shift']!=0])]:
        summary[subset] = {m: acc(selected, m) for m in ('native', 'late', 'early')}
        summary[subset]['n'] = len(selected)
    meta.update(elapsed_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                summary=summary, completed=True)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    subprocess.run([sys.executable, 'analyze_late_binding_c2c_v1.py', str(out.with_suffix('.jsonl')),
        '--metadata', str(out.with_suffix('.json')), '--out', str(out)+'.analysis.json'], check=True)


if __name__ == '__main__':
    main()
