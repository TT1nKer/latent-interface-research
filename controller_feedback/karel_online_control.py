"""Evaluate a flat support-fitted next-action policy on held-out Karel worlds."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
from statistics import mean
import zipfile

from karel_predicates import observe
from karel_trace_control import World, replay, sha


STOP = "STOP"
BEGIN = ("begin",)
MODES = ("live", "initial", "masked")
HISTORY = 3


def contexts(obs, history):
    for length in range(len(history), 0, -1):
        yield obs, history[-length:]
    yield obs, ()


def next_history(history, observation, action):
    return (history + ((observation, action),))[-HISTORY:]


def sensor(mode, live, initial):
    return live if mode == "live" else initial if mode == "initial" else None


def demonstration(start, commands, target):
    w = start.clone()
    pairs = []
    for action in commands:
        pairs.append((observe(w), action))
        if not w.step(action):
            raise ValueError("Illegal reference action")
    if w.state() != target:
        raise ValueError("Reference execution disagrees with recorded endpoint")
    return pairs + [(observe(w), STOP)]


def fit(support, mode):
    votes = defaultdict(Counter)
    for trace in support:
        history = (BEGIN,)
        first = trace[0][0]
        for live, action in trace:
            obs = sensor(mode, live, first)
            for key in contexts(obs, history):
                votes[key][action] += 1
            history = next_history(history, obs, action)
    return {key: min(hist, key=lambda a: (-hist[a], a != STOP, a)) for key, hist in votes.items()}


def predict(lut, obs, history):
    for key in contexts(obs, history):
        if key in lut:
            return lut[key]
    return None


def execute(start, model, fallback, mode, limit):
    world = start.clone()
    initial = observe(world) if mode != "masked" else None
    history = blind_history = (BEGIN,)
    actions = []
    while True:
        live = observe(world) if mode == "live" else None
        obs = sensor(mode, live, initial)
        act = predict(model, obs, history)
        if act is None:
            act = predict(fallback, None, blind_history)
        if act == STOP:
            return "stop", tuple(actions), world.state()
        if len(actions) >= limit:
            return "cap", tuple(actions), None
        if not world.step(act):
            return "crash", tuple(actions), None
        actions.append(act)
        history = next_history(history, obs, act)
        blind_history = next_history(blind_history, None, act)


def evaluate(record):
    cases = record["examples"][:6]
    if len(cases) < 2:
        return None
    starts = [World.parse(c["inpgrid_json"]) for c in cases]
    ends = [World.parse(c["outgrid_json"]).state() for c in cases]
    traces = [tuple(c["actions"]) for c in cases]
    demos = [demonstration(s, t, end) for s, t, end in zip(starts, traces, ends)]
    outcomes = []
    for j, start in enumerate(starts):
        support = [d for i, d in enumerate(demos) if i != j]
        support_traces = {t for i, t in enumerate(traces) if i != j}
        ceiling_hit = any(replay(start, t) == ends[j] for t in support_traces)
        limit = min(512, max(32, 2 * max(map(len, support_traces)) + 8))
        fallback = fit(support, "masked")
        outcome = {"support_has_successful_fixed_plan": ceiling_hit}
        for mode in MODES:
            model = fallback if mode == "masked" else fit(support, mode)
            status, actions, end = execute(start, model, fallback, mode, limit)
            success = end == ends[j]
            if success and not ceiling_hit and actions in support_traces:
                raise ValueError("A supposedly unavailable successful trace was in the support")
            outcome[mode] = {"endpoint": success, "trace_match": status == "stop" and actions == traces[j],
                             "status": status, "steps": len(actions)}
        outcomes.append(outcome)
    code = " ".join(record["program_tokens"])
    return {"id": record["guid"], "program_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "conditional": any(t in {"IF", "IFELSE", "WHILE"} for t in record["program_tokens"]),
            "targets": outcomes}


def summary(rows):
    worlds = [t for row in rows for t in row["targets"]]
    uncovered = [t for t in worlds if not t["support_has_successful_fixed_plan"]]
    result = {"programs": len(rows), "distinct_code": len({r["program_sha256"] for r in rows}),
              "worlds": len(worlds), "worlds_outside_fixed_plan_ceiling": len(uncovered)}
    macro = {}
    for mode in MODES:
        macro[mode] = [mean(t[mode]["endpoint"] for t in row["targets"]) for row in rows]
        result[mode] = {"macro_endpoint": mean(macro[mode]), "endpoint_count": sum(t[mode]["endpoint"] for t in worlds),
                        "trace_count": sum(t[mode]["trace_match"] for t in worlds),
                        "outside_ceiling_successes": sum(t[mode]["endpoint"] for t in uncovered),
                        "lost_vs_ceiling": sum(t["support_has_successful_fixed_plan"] and not t[mode]["endpoint"] for t in worlds),
                        "termination": dict(Counter(t[mode]["status"] for t in worlds))}
    rng = random.Random(20261120)
    for comparator in ("masked", "initial"):
        delta = [x-y for x, y in zip(macro["live"], macro[comparator])]
        draws = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(5000))
        result["live_minus_" + comparator] = {"mean": mean(delta), "paired_program_bootstrap_95": [draws[124], draws[4874]],
                                               "live_only": sum(t["live"]["endpoint"] and not t[comparator]["endpoint"] for t in worlds),
                                               "comparator_only": sum(t[comparator]["endpoint"] and not t["live"]["endpoint"] for t in worlds)}
    result["by_syntax"] = {}
    for flag in (True, False):
        subset = [r for r in rows if r["conditional"] == flag]
        if subset:
            result["by_syntax"][str(flag)] = {mode: mean(mean(t[mode]["endpoint"] for t in r["targets"]) for r in subset) for mode in MODES}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("zip", "baseline", "protocol", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    baseline = json.loads(args.baseline.read_text())
    if baseline["inputs"]["val.zip"] != sha(args.zip) or baseline["worker_sha256"] != sha(Path(__file__).with_name("karel_trace_control.py")):
        raise ValueError("Mismatch with validated replay cohort or simulator")
    rows = []
    with zipfile.ZipFile(args.zip) as archive, archive.open("val.json") as source:
        for line in itertools.islice(source, 2500):
            row = evaluate(json.loads(line))
            if row is not None:
                rows.append(row)
    s = summary(rows)
    if s["programs"] != baseline["validation"]["programs"] or s["worlds"] != baseline["validation"]["worlds"]:
        raise ValueError("Eligibility mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    detail = args.out / "programs.jsonl"
    detail.write_text("".join(json.dumps(r, separators=(",", ":"))+"\n" for r in rows))
    s.update(job=job, baseline_sha256=sha(args.baseline), protocol_sha256=sha(args.protocol),
             worker_sha256=sha(Path(__file__)), predicates_sha256=sha(Path(__file__).with_name("karel_predicates.py")),
             results_sha256=sha(detail))
    (args.out / "summary.json").write_text(json.dumps(s, indent=2)+"\n")
