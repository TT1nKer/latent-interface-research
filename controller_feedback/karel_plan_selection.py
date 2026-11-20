"""Cross-fit prospective Karel plan rankers on programs with disjoint token streams."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import zipfile

import numpy as np

import karel_predicates
import karel_trace_control
from karel_predicates import observe
from karel_trace_control import World, sha


N_FOLDS = 5
METHODS = ("mode", "legality_mode", "sensor", "ranker", "oracle")
ARCHIVE_SHA = "967dc1c557595e7c4a5ff08a636ec3fe2b5a5320fd6548be50295cf36c5ab4cd"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(world):
    return sum(int(value) << index for index, value in enumerate(observe(world)))


def simulate(initial, actions):
    world = initial.clone()
    sensors = [signature(world)]
    for index, action in enumerate(actions):
        if not world.step(action):
            return None, None, index + 1
        sensors.append(signature(world))
    return world.state(), tuple(sensors), len(actions)


def candidate_features(target_input, supports, counters):
    """Inputs include no target actions, output, program tokens or learned labels."""
    candidates = defaultdict(list)
    for plan, sensors in supports:
        candidates[plan].append(sensors)
    rows = []
    for plan, donors in candidates.items():
        end, trace, attempted = simulate(target_input, plan)
        counters["target_candidate_simulations"] += 1
        counters["target_attempted_steps"] += attempted
        counters["target_empty_candidate_simulations"] += not plan
        agreement = 0.
        if trace is not None:
            # Both signatures include the terminal decision to stop.
            disagreements = min(sum((a ^ b).bit_count() for a, b in zip(trace, donor))
                                for donor in donors)
            agreement = 1. - disagreements / (4 * len(trace))
        rows.append((plan, [float(end is not None), len(donors) / len(supports), agreement], end))
    return rows


def program_cases(record, counters):
    worlds = record["examples"][:6]
    initial = [World.parse(w["inpgrid_json"]) for w in worlds]
    targets = [World.parse(w["outgrid_json"]).state() for w in worlds]
    plans = [tuple(w["actions"]) for w in worlds]
    donors = []
    for world, target, plan in zip(initial, targets, plans):
        end, trace, attempted = simulate(world, plan)
        require(end == target, f"Failed own-world replay for {record['guid']}")
        counters["support_simulations"] += 1
        counters["support_attempted_steps"] += attempted
        counters["support_empty_plans"] += not plan
        donors.append((plan, trace))
    cases = []
    for i, (world, target) in enumerate(zip(initial, targets)):
        candidates = candidate_features(world, donors[:i] + donors[i + 1:], counters)
        x = np.asarray([features for _, features, _ in candidates])
        # Outcomes and reference actions appear only in validation and scoring.
        y = np.array([end == target for _, _, end in candidates], dtype=float)
        literal = np.array([plan == plans[i] for plan, _, _ in candidates])
        cases.append((x, y, literal))
    return cases


def fit(training):
    matrices, targets, weights = [], [], []
    for rec in training:
        for x, y, _ in rec["cases"]:
            matrices.append(x)
            targets.append(y)
            weights.extend([1. / (len(training) * len(rec["cases"]) * len(y))] * len(y))
    x = np.vstack(matrices)
    x = np.column_stack((np.ones(len(x)), x))
    y = np.concatenate(targets)
    w = np.array(weights)
    require(abs(w.sum()-1.) < 1e-9, "Program/target/candidate training weights incorrect")
    fraction = np.clip(w @ y, 1e-8, 1-1e-8)
    beta = np.array([np.log(fraction / (1.-fraction)), 0., 0., 0.])
    penalty = np.diag([0., .01, .01, .01])
    for iteration in range(1, 41):
        prob = 1. / (1. + np.exp(-np.clip(x @ beta, -40., 40.)))
        gradient = x.T @ (w * (prob - y)) + penalty @ beta
        hessian = (x.T * (w * prob * (1.-prob))) @ x + penalty
        delta = np.linalg.solve(hessian, gradient)
        beta -= delta
        if np.max(np.abs(delta)) < 1e-9:
            break
    require(np.all(np.isfinite(beta)), "Non-finite fitted weights")
    return beta, iteration


def score_case(case, beta):
    features, y, literal = case
    legal, freq, similar = features.T
    scores = (
        freq,
        2.*legal + freq,
        # This candidate set is small; lexicographic masking keeps ties explicit.
        None,
        features @ beta[1:] + beta[0],
        y,
    )
    values = {}
    for name, scores_for_method in zip(METHODS, scores):
        if name == "sensor":
            keep = legal == max(legal)
            keep &= np.abs(similar - max(similar[keep])) <= 1e-12
            keep &= np.abs(freq - max(freq[keep])) <= 1e-12
        else:
            keep = np.abs(scores_for_method - max(scores_for_method)) <= 1e-12
        values[name] = float(y[keep].mean())
    certified = (legal == 1.) & (similar == 1.)
    return values, Counter(has_exact_signature=int(certified.any()),
                           exact_signature_candidates=int(certified.sum()),
                           exact_signature_endpoint_hits=int(y[certified].sum()),
                           exact_signature_trace_matches=int(literal[certified].sum()))


def paired_bootstrap(outputs):
    """Resample exact-program groups, retaining every program record in a sampled group."""
    groups = defaultdict(list)
    for row in outputs:
        groups[row["code_sha256"]].append([row["ranker"] - row[base]
                                           for base in ("legality_mode", "sensor")])
    by_group = list(groups.values())
    sizes = np.array([len(g) for g in by_group])
    totals = np.array([np.sum(g, axis=0) for g in by_group])
    rng = np.random.default_rng(20261120)
    draws = []
    for _ in range(20):
        samples = rng.integers(0, len(groups), size=(100, len(groups)))
        draws.extend(totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)[:, None])
    diffs = totals.sum(axis=0) / sizes.sum()
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {name: {"mean":float(diffs[i]), "paired_program_bootstrap_95":intervals[:, i].tolist()}
            for i, name in enumerate(("legality_mode", "sensor"))}


def study(archive, receipt):
    primary = json.loads(receipt.read_text())
    require(primary["inputs"]["val.zip"] == sha(archive) == ARCHIVE_SHA and
            primary["author_programs_in_mirror"] == 1000 and
            primary["worker_sha256"] == sha(Path(karel_trace_control.__file__)), "Primary receipt mismatch")
    counters = Counter()
    programs = []
    with zipfile.ZipFile(archive) as container, container.open("val.json") as raw:
        for line in itertools.islice(raw, 2500):
            rec = json.loads(line)
            if len(rec["examples"]) < 2:
                counters["fewer_than_two_worlds"] += 1
                continue
            canonical = " ".join(rec["program_tokens"]).encode()
            code_id = hashlib.sha256(canonical).hexdigest()
            fold = int.from_bytes(hashlib.sha256(b"karel-plan-selector-v1/" + canonical).digest(), "big") % N_FOLDS
            programs.append({"code_sha256":code_id, "fold":fold, "cases":program_cases(rec, counters)})
    require(len(programs) == primary["validation"]["programs"] and
            counters["support_simulations"] == primary["validation"]["worlds"], "Cohorts differ")
    outputs, fits = [], []
    for fold in range(N_FOLDS):
        train = [p for p in programs if p["fold"] != fold]
        test = [p for p in programs if p["fold"] == fold]
        train_ids = {p["code_sha256"] for p in train}
        require(train and test and not train_ids.intersection(p["code_sha256"] for p in test), "Invalid fold")
        beta, updates = fit(train)
        fits.append({"fold":fold,"train_programs":len(train),"test_programs":len(test),
                     "coefficients":beta.tolist(), "newton_iterations":updates})
        for record in test:
            sums = Counter()
            for case in record["cases"]:
                values, diagnostic = score_case(case, beta)
                sums.update(values)
                counters.update(diagnostic)
            outputs.append({"fold":fold, "code_sha256":record["code_sha256"],
                            **{key:sums[key]/len(record["cases"]) for key in METHODS}})
    distinct = len({p["code_sha256"] for p in programs})
    gains = paired_bootstrap(outputs)
    estimates = {k:float(np.mean([row[k] for row in outputs])) for k in METHODS}
    return {
        "programs":len(programs), "canonical_programs":distinct, "worlds":counters["support_simulations"],
        "author_code_order_matches_prefix":primary["author_codes_exact_prefix"],
        "source":"Independent validation mirror corroborated by 1000 author-published program sequences.",
        "archive_sha256":sha(archive), "primary_receipt_sha256":sha(receipt),
        "program_mean_endpoint_accuracy":estimates, "ranker_paired_gains":gains,
        "fitted_selector_retained":all(g["mean"] >= .01 and g["paired_program_bootstrap_95"][0] > 0
                                      for g in gains.values()),
        "simulation_and_signature_counts":counters,
        "coefficient_order":["intercept","legal","support_frequency","best_sensor_similarity"],
        "folds":fits,
        "inference_scope":"The selected plan is fixed before acting. Simulation and equality of local sensor traces use a known, deterministic model of the target world. New exact programs may share templates; no test of unseen primitive sets, hidden task outcomes or the necessity of online control.",
    }, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "primary-receipt", "protocol", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit(), "School CPU Slurm allocation required")
    report, rows = study(args.archive, args.primary_receipt)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "program_rows.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",",":"), allow_nan=False)+"\n")
    report.update(job=job, protocol_sha256=sha(args.protocol),
                  worker_sha256=sha(Path(__file__)), observation_sha256=sha(Path(karel_predicates.__file__)),
                  program_rows_sha256=sha(args.out / "program_rows.jsonl"))
    with (args.out / "summary.json").open("x") as f:
        f.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["program_mean_endpoint_accuracy"]))


if __name__ == "__main__":
    main()
