"""Meta-train two matched-capacity policies from unsegmented demonstration episodes."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

import torch
from karel_online_control import demonstration
from karel_trace_control import World, sha
from karel_unaligned_model import ACTION_ID, MetricPolicy, NeuralPolicy, SEEDS, encode_trace, pack_episodes


ZIP_SHA = '967dc1c557595e7c4a5ff08a636ec3fe2b5a5320fd6548be50295cf36c5ab4cd'
STEPS = 1200
BATCH = 16


def training_bank(path):
    bank = defaultdict(list)
    withheld = set()
    with zipfile.ZipFile(path) as archive, archive.open('val.json') as src:
        for line in src:
            row = json.loads(line)
            code = hashlib.sha256(' '.join(row['program_tokens']).encode()).hexdigest()
            if int(code[:8], 16) % 5 == 0:
                withheld.add(code)
                continue
            episode = []
            for world in row['examples'][:6]:
                trace = demonstration(World.parse(world['inpgrid_json']), world['actions'],
                                      World.parse(world['outgrid_json']).state())
                episode.append(encode_trace(trace))
            if len(episode) != 6:
                raise ValueError('Expected six training worlds per program')
            bank[code].append(episode)
    if len(bank) != 1650 or len(withheld) != 395 or set(bank) & withheld:
        raise ValueError('Unexpected or overlapping development split')
    return bank


def draw(bank, codes, rng):
    batch = []
    for _ in range(BATCH):
        ep = rng.choice(bank[rng.choice(codes)])
        q = rng.randrange(6)
        batch.append([ep[q]] + ep[:q] + ep[q+1:])
    return pack_episodes(batch)


def validate_execution():
    """Check stepwise inference and absence of future-query leakage, without using study outcomes."""
    o0, o1 = (0, 1, 1, 0), (1, 0, 1, 1)
    a = [(o0, 'turnLeft'), (o1, 'move'), (o0, 'STOP')]
    b = [(o0, 'putMarker'), (o0, 'STOP')]
    support = [b, a, b, a, [(o1, 'STOP')]]
    x, y = pack_episodes([[encode_trace(t) for t in [a] + support]])
    for variant in ('retrieval','centroid'):
        model = MetricPolicy(variant).eval()
        with torch.no_grad():
            pred = model(x, y)
            altered = x.clone()
            altered[:, 0, 1:] = 0
            assert torch.allclose(pred[:, :1], model(altered, y)[:, :1], atol=1e-6)
            agent = NeuralPolicy(model, support)
            for t, (obs, expected) in enumerate(a):
                assert ACTION_ID[agent.act(obs)] == pred[0,t].argmax().item()
                if expected != 'STOP':
                    agent.advance(obs, expected)


def fit_one(variant, seed, bank, codes):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = MetricPolicy(variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    totals = torch.zeros(3)
    then = time.monotonic()
    for step in range(1, STEPS+1):
        x, y = draw(bank, codes, rng)
        p = model(x, y)
        target = y[:, 0]
        valid = target >= 0
        correct_prob = p.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        n = valid.sum(dim=1)
        loss = (-(.999 * correct_prob + .001/6).log() * valid).sum(1).div(n).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            accuracy = (p.argmax(-1).eq(target) * valid).sum(1).div(n).mean()
            covered = (correct_prob.gt(0) * valid).sum(1).div(n).mean()
            totals += torch.stack((loss.detach(), accuracy, covered))
        if step % 100 == 0:
            loss_value, acc, coverage = (totals / 100).tolist()
            history.append(dict(step=step, loss=loss_value, teacher_forced_accuracy=acc, demonstrated_label_coverage=coverage))
            totals.zero_()
            print(variant, seed, step, f'loss={loss_value:.3f}', f'seconds={time.monotonic()-then:.1f}', flush=True)
    return model.state_dict(), {'trajectory':history, 'seconds':time.monotonic()-then,
                                'parameters':sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('zip','protocol','out'):
        parser.add_argument('--'+key, type=Path, required=True)
    a = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit() or torch.cuda.is_available():
        raise RuntimeError('An allocated school CPU without a GPU is required')
    if sha(a.zip) != ZIP_SHA:
        raise ValueError('Development archive differs from fixed input')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    validate_execution()
    a.out.mkdir(parents=True, exist_ok=False)
    bank = training_bank(a.zip)
    codes = sorted(bank)
    model_sha = sha(Path(__file__).with_name('karel_unaligned_model.py'))
    shared = {'protocol_sha256':sha(a.protocol), 'model_sha256':model_sha, 'source_sha256':ZIP_SHA,
              'trainer_sha256':sha(Path(__file__)), 'seeds':list(SEEDS)}
    checkpoint = dict(shared, models={}, training_codes=codes)
    reports = {}
    for seed in SEEDS:
        for variant in ('retrieval','centroid'):
            params, info = fit_one(variant, seed, bank, codes)
            key = f'{variant}_{seed}'
            checkpoint['models'][key] = params
            reports[key] = info
    file = a.out/'checkpoint.pt'
    torch.save(checkpoint, file)
    meta = dict(shared, job=job, completed=True, checkpoint_sha256=sha(file),
                training_code_count=len(codes), training_records=sum(map(len,bank.values())),
                training_code_sha256=hashlib.sha256(('\n'.join(codes)+'\n').encode()).hexdigest(),
                torch_version=str(torch.__version__), runs=reports)
    (a.out/'training_summary.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
