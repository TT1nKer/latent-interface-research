"""Bounded Slurm-only replay. No training, simulator, or task-success claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_processor(checkpoint):
    # ProcessorMixin delegates to AutoImageProcessor even when its outer class is
    # explicit. Construct both components from the pinned local implementations.
    from transformers import AutoTokenizer
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    return PrismaticProcessor(
        PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True),
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False))


def check_processor(checkpoint, out, source, prepare):
    """Slurm CPU regression: fixed components equal locally registered HF dispatch."""
    import builtins
    import inspect
    from unittest.mock import patch
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    with patch.object(builtins, 'input', side_effect=AssertionError('interactive loader')):
        processor = load_processor(checkpoint)
        # This is the old ProcessorMixin path with the missing local registration
        # supplied. Explicit false prevents executing the checkpoint's Python.
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        reference = PrismaticProcessor.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        image = Image.fromarray(np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3))
        outputs = []
        for prompt in ['In: What action should the robot take to pick up the bowl?\nOut:',
                       'In: What action should the robot take to open the drawer?\nOut:']:
            a, b = processor(prompt, image), reference(prompt, image)
            assert set(a) == set(b)
            assert all(torch.equal(a[k], b[k]) for k in a), 'processor dispatch differs'
            outputs.append({k: sha_bytes(a[k].contiguous().numpy().tobytes()) for k in a})
    write(out / (source + '_processor_check.json'), {
        'passed': True, 'model_loaded': False, 'source': source,
        'worker_sha256': sha(__file__), 'prepare_sha256': sha(prepare),
        'processor_source_sha256': sha(inspect.getfile(PrismaticProcessor)),
        'config_sha256': {n: sha(checkpoint / n) for n in [
            'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json']},
        'checks': outputs, 'scientific_result': None})


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Models may only run in Slurm'
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['native', 'fastvla'], required=True)
    ap.add_argument('--prepare', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--processor-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = json.loads(Path(args.prepare).read_text())
    assert prep['completed'] and prep['asset_lock_sha256'] == sha('vla_latency_assets_v1.json')
    assert prep['cohort_sha256'] == sha(prep['cohort_path'])
    cohort = json.loads(Path(prep['cohort_path']).read_text())
    lock = json.loads(Path('vla_latency_assets_v1.json').read_text())
    source = Path(os.environ['PYTHONPATH'].split(':')[0])
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == lock[args.source]['revision']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'])
    checkpoint = Path(prep['model_path'])
    if args.processor_only:
        check_processor(checkpoint, out, args.source, args.prepare)
        return
    start = time.perf_counter()
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # Image preprocessing stays on CPU; no TF GPU preallocation.
    import numpy as np
    import torch
    import transformers
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    from experiments.robot.openvla_utils import get_vla_action, load_component_state_dict
    from experiments.robot.robot_utils import set_seed_everywhere, normalize_gripper_action, invert_gripper_action
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert ACTION_DIM == 7 and NUM_ACTIONS_CHUNK == 8
    set_seed_everywhere(7)
    processor = load_processor(checkpoint)  # Fail before loading model weights.
    # Use explicit pinned classes. The upstream helper mutates checkpoint source/config files;
    # this loader deliberately does not call that helper or execute snapshot Python files.
    cfg = OpenVLAConfig.from_pretrained(checkpoint, local_files_only=True)
    model, loading = OpenVLAForActionPrediction.from_pretrained(
        checkpoint, config=cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation='sdpa', local_files_only=True, output_loading_info=True)
    assert not any(loading[k] for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    if args.source == 'fastvla':
        model.set_num_images_in_input(2)
    else:
        model.vision_backbone.set_num_images_in_input(2)
    model.eval().to('cuda')
    model.norm_stats = json.loads((checkpoint / 'dataset_statistics.json').read_text())
    key = 'libero_spatial'
    if key not in model.norm_stats: key += '_no_noops'
    assert key in model.norm_stats
    action_head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
    action_head.load_state_dict(load_component_state_dict(str(checkpoint / 'action_head--150000_checkpoint.pt')))
    action_head.eval().to(dtype=torch.bfloat16, device='cuda')
    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
    proprio.load_state_dict(load_component_state_dict(str(checkpoint / 'proprio_projector--150000_checkpoint.pt')))
    proprio.eval().to(dtype=torch.bfloat16, device='cuda')
    call_cfg = SimpleNamespace(num_images_in_input=2, center_crop=True, use_proprio=True, unnorm_key=key)
    pruner = model.language_model.model.pruner if args.source == 'fastvla' else None
    backbone = model.language_model.model
    episodes = []
    for e in cohort['episodes']:
        assert sha(e['file']) == e['sha256']
        with np.load(e['file'], allow_pickle=False) as z:
            episodes.append((e, {k: z[k].copy() for k in z.files}))
    baseline = {}
    if args.source == 'fastvla':
        native = out / 'native.jsonl'
        baseline = {(r['episode'], r['frame']): r for r in map(json.loads, native.read_text().splitlines())}
        assert len(baseline) == sum(len(e['frames']) for e, _ in episodes)

    def configure(mode):
        if pruner is not None:
            pruner.configure(enable=mode != 'disabled', rho=0.8,
                             tau=0.1 if mode == 'adaptive' else 0.0, window=10)

    def run_one(e, data, i, mode, repeat, measured):
        # get_vla_action normalizes obs['state'] IN PLACE. Never reuse its mutable input.
        obs = {k: v[i].copy() for k, v in data.items()}
        history = list(pruner.history) if pruner is not None else []
        old_tokens = getattr(backbone, 'total_kept_patches', 0)
        old_llm_ms = getattr(backbone, 'total_cuda_time', 0.0)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        raw = np.stack(get_vla_action(call_cfg, model, processor, obs, e['task'],
                                     action_head=action_head, proprio_projector=proprio))
        actions = invert_gripper_action(normalize_gripper_action(raw, binarize=True))
        ev1.record()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        assert raw.shape == (8, 7) and np.isfinite(raw).all()
        assert torch.cuda.max_memory_reserved() < 22 * 1024 ** 3, 'exceeds 22 GiB planning limit'
        record = {'episode': e['episode'], 'frame': e['frames'][i], 'mode': mode, 'repeat': repeat,
                  'wall_seconds': seconds, 'cuda_span_ms': ev0.elapsed_time(ev1),
                  'cuda_span_note': 'Stream event span includes host launch gaps, not sum of kernel execution times.',
                  'llm_cuda_ms': getattr(backbone, 'total_cuda_time', 0.0) - old_llm_ms if pruner is not None else None,
                  'retained_tokens': getattr(backbone, 'total_kept_patches', 0) - old_tokens if pruner is not None else None,
                  'controller_history_before': history,
                  'controller_step_after': pruner.step if pruner is not None else None,
                  'raw_actions': raw.tolist(), 'actions': actions.tolist(),
                  'max_reserved_bytes': torch.cuda.max_memory_reserved()}
        if mode == 'disabled' and measured:
            ref = baseline[(e['episode'], e['frames'][i])]
            expected = np.asarray(ref['raw_actions'])
            record['native_max_abs_error'] = float(np.abs(raw - expected).max())
            record['native_exact'] = bool(np.array_equal(raw, expected))
            record['native_tolerance_passed'] = bool(np.allclose(raw, expected, rtol=0.005, atol=0.002)
                and np.array_equal(actions[:, -1], np.asarray(ref['actions'])[:, -1]))
        return record

    orders = [['native']] if args.source == 'native' else [
        ['disabled', 'adaptive', 'fixed'], ['fixed', 'disabled', 'adaptive'], ['adaptive', 'fixed', 'disabled']]
    checks = []
    nrows = 0
    setup_seconds = time.perf_counter() - start
    write(out / (args.source + '_started.json'), {
        'job_id': os.environ['SLURM_JOB_ID'], 'source_revision': head, 'prepare_sha256': sha(args.prepare),
        'cohort_sha256': prep['cohort_sha256'], 'worker_sha256': sha(__file__),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(), 'sdpa': model.language_model.config._attn_implementation,
        'bf16_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'tf32': torch.backends.cuda.matmul.allow_tf32, 'setup_seconds': setup_seconds,
        'upstream_fastvla_instrumentation': 'Retained unchanged; has internal synchronization and FLOP bookkeeping.',
        'native_tolerance': {'atol': 0.002, 'rtol': 0.005, 'same_binarized_gripper': True}})
    with (out / (args.source + '.jsonl')).open('x') as f:
        for repeat, modes in enumerate(orders):
            for mode in modes:
                configure(mode)
                e, data = episodes[0]
                for _ in range(3): run_one(e, data, 0, mode, repeat, False)
                for e, data in episodes:
                    configure(mode)  # Episode and replay reset, after warmup.
                    for i in range(len(e['frames'])):
                        rec = run_one(e, data, i, mode, repeat, True)
                        f.write(json.dumps(rec, allow_nan=False) + '\n')
                        f.flush()
                        nrows += 1
                        if mode == 'disabled':
                            checks.append(rec['native_tolerance_passed'])
                            assert rec['native_tolerance_passed'], ('native control fails', e['episode'], e['frames'][i])
    write(out / (args.source + '_summary.json'), {
        'completed': True, 'rows': nrows, 'elapsed_seconds': time.perf_counter() - start,
        'setup_seconds': setup_seconds, 'cohort_sha256': prep['cohort_sha256'],
        'max_allocated_bytes': torch.cuda.max_memory_allocated(),
        'max_reserved_bytes': torch.cuda.max_memory_reserved(), 'cpu_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native_check_count': len(checks), 'native_check_all': all(checks) if checks else None,
        'rows_sha256': sha(out / (args.source + '.jsonl')), 'scientific_task_result': None})


if __name__ == '__main__':
    main()
