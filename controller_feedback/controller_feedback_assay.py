"""Run the frozen post-return routing diagnostic on a school CPU allocation."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
from statistics import mean
import time

import torch
from torch import nn
from torch.nn import functional as F


ROLES, OPS, WIDTH = 5, 4, 48
SEEDS = (17, 29, 43, 71, 97)
BINDINGS = list(itertools.permutations(range(ROLES)))  # slot -> role


def after(first, second):
    return tuple(second[v] for v in first)


def world(seed):
    rng = random.Random(seed)
    rules = []
    candidates = BINDINGS.copy()
    rng.shuffle(candidates)
    for rule in candidates:
        if rule != tuple(range(ROLES)) and all(after(rule, other) != after(other, rule) for other in rules):
            rules.append(rule)
        if len(rules) == OPS:
            return rules
    raise ValueError("Could not construct a noncommuting operator set")


def word_sets():
    pairs = list(itertools.product(range(OPS), repeat=2))
    held = [p for p in pairs if p[1] == (p[0] + 1) % OPS]
    seen = [p for p in pairs if p not in held]
    triples = [w for w in itertools.product(range(OPS), repeat=3) if w[:2] in seen and w[1:] in seen]
    return {"train": [(i,) for i in range(OPS)] + seen, "new_pairs": held, "new_triples": triples}


def cases(words, rules):
    rows = []
    for word in words:
        for z in range(ROLES):
            role = z
            for op in word:
                role = rules[op][role]
            rows.append((z, word, role))
    return rows


def encode(rows):
    z = F.one_hot(torch.tensor([r[0] for r in rows]), ROLES).float()
    ops = torch.zeros(len(rows), 3, OPS)
    for i, (_, word, _) in enumerate(rows):
        for t, op in enumerate(word):
            ops[i, t, op] = 1
    return z, ops, torch.tensor([r[2] for r in rows])


class FlatPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ROLES + OPS * 3, WIDTH), nn.ReLU(),
                                 nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, ROLES))

    def forward(self, z, ops):
        return self.net(torch.cat([z, ops.flatten(1)], dim=1))


class RecurrentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.start = nn.Linear(ROLES, WIDTH)
        self.cell = nn.GRUCell(OPS, WIDTH)
        self.out = nn.Linear(WIDTH, ROLES)

    def forward(self, z, ops):
        h = self.start(z)
        for op in ops.unbind(dim=1):
            h = torch.where(op.any(dim=1, keepdim=True), self.cell(op, h), h)
        return self.out(h)


def learn(model, data):
    z, ops, target = data
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(400):
        opt.zero_grad()
        loss = F.cross_entropy(model(z, ops), target)
        loss.backward()
        opt.step()


def interpret(rows, singleton_training):
    tables = {(word[0], z): target for z, word, target in singleton_training if len(word) == 1}
    assert len(tables) == ROLES * OPS
    predictions = []
    for z, word, _ in rows:
        for op in word:
            z = tables[op, z]
        predictions.append(z)
    return predictions


def measure(rows, predictions):
    pred = {(w, z): role for (z, w, _), role in zip(rows, predictions, strict=True)}
    truth = {(w, z): role for z, w, role in rows}
    correct = paired = 0
    for bank in BINDINGS:
        for (w, z), target in truth.items():
            # Changing the probe return alone leaves the word and all five leaf contracts fixed.
            other = (w, (z + 1) % ROLES)
            assert truth[other] != target
            address = bank.index(pred[w, z])
            hit = address == bank.index(target)
            correct += hit
            paired += hit and bank.index(pred[other]) == bank.index(truth[other])
    n = len(pred) * len(BINDINGS)
    return {"route": correct / n, "both_twins": paired / n,
            "distinct_word_state_pairs": len(rows), "bindings_checked": len(BINDINGS)}


def run(seed):
    rules = world(seed)
    datasets = {name: cases(words, rules) for name, words in word_sets().items()}
    tensors = {name: encode(rows) for name, rows in datasets.items()}
    metrics = {}
    for label, cls in (("flat_mlp", FlatPolicy), ("flat_gru", RecurrentPolicy)):
        torch.manual_seed(seed)
        model = cls()
        learn(model, tensors["train"])
        model.eval()
        with torch.no_grad():
            metrics[label] = {name: measure(rows, model(*tensors[name][:2]).argmax(1).tolist())
                              for name, rows in datasets.items()}
        metrics[label]["parameters"] = sum(p.numel() for p in model.parameters())
    metrics["fitted_table"] = {name: measure(rows, interpret(rows, datasets["train"]))
                               for name, rows in datasets.items()}
    metrics["pre_return_constant"] = {name: measure(rows, [0] * len(rows)) for name, rows in datasets.items()}
    return {"seed": seed, "operators": rules, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or torch.cuda.is_available():
        raise RuntimeError("Expected an allocated school CPU and no visible GPU")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    runs = [run(s) for s in SEEDS]
    summary = {}
    for policy in runs[0]["metrics"]:
        summary[policy] = {}
        for split in word_sets():
            values = [r["metrics"][policy][split]["route"] for r in runs]
            twins = [r["metrics"][policy][split]["both_twins"] for r in runs]
            summary[policy][split] = {"mean": mean(values), "range": [min(values), max(values)],
                                      "both_twins_mean": mean(twins)}
    result = {"job": job, "torch_version": torch.__version__, "seeds": SEEDS,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic() - start, "splits": word_sets(),
              "worlds": runs, "summary": summary}
    with args.out.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
