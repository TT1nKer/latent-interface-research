"""Teacher-forced next-command counterfactuals at the first arrival in Karel."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import mean
import zipfile

from karel_online_control import (BEGIN, MODES, STOP, demonstration, fit, next_history,
                                  predict, sensor)
from karel_predicates import observe
from karel_trace_control import ACTIONS, World, sha


class SilentLimit(Exception):
    """The oracle did not emit another action within the silent-operation limit."""


def condition(test, world):
    kind = test["type"]
    if kind == "not":
        return not condition(test["condition"], world)
    front, left, right, marker = observe(world)
    return {"frontIsClear": front, "leftIsClear": left, "rightIsClear": right,
            "markersPresent": marker, "noMarkersPresent": not marker}[kind]


def decisions(program, world):
    # A yielded primitive is applied by the caller, before the interpreter resumes.
    # The suspended generator preserves nested repeats, loops, and pending return addresses.
    tick = 0

    def count():
        nonlocal tick
        tick += 1
        if tick > 4096:
            raise SilentLimit

    def block(nodes):
        nonlocal tick
        for node in nodes:
            kind = node["type"]
            if kind in ACTIONS:
                tick = 0
                yield kind
            elif kind == "repeat":
                for _ in range(node["times"]):
                    count()
                    yield from block(node["body"])
            elif kind == "while":
                while True:
                    count()
                    if not condition(node["condition"], world):
                        break
                    yield from block(node["body"])
            elif kind in ("if", "ifElse"):
                count()
                take = condition(node["condition"], world)
                if kind == "ifElse":
                    yield from block(node["ifBody"] if take else node["elseBody"])
                elif take:
                    yield from block(node["body"])
            else:
                raise ValueError(f"Unsupported reference AST node {kind!r}")
    yield from block(program["run"])


def records(path):
    with zipfile.ZipFile(path) as archive, archive.open("val.json") as source:
        for line in itertools.islice(source, 2500):
            row = json.loads(line)
            if len(row["examples"]) >= 2:
                yield row


def native_check(record):
    for j, case in enumerate(record["examples"][:6]):
        world = World.parse(case["inpgrid_json"])
        cursor = decisions(record["program_json"], world)
        for i, expected in enumerate(case["actions"]):
            actual = next(cursor, STOP)
            if actual != expected or not world.step(actual):
                raise ValueError(f"Invalid native action {record['guid']}/{j}/{i}: {actual} vs {expected}")
        if next(cursor, STOP) != STOP or world.state() != World.parse(case["outgrid_json"]).state():
            raise ValueError(f"Invalid native terminal {record['guid']}/{j}")
    return len(record["examples"][:6])


def oracle_at_arrival(start, program, prefix, count):
    world = start.clone()
    cursor = decisions(program, world)
    for action in prefix:
        if next(cursor, STOP) != action or not world.step(action):
            raise ValueError("Identical native prefix not reproduced")
    if world.hero == start.hero:
        raise ValueError("First forward move must reach a fresh cell")
    originally = world.markers.get(world.hero, 0)
    if count:
        world.markers[world.hero] = count
    else:
        world.markers.pop(world.hero, None)
    try:
        next_action = next(cursor, STOP)
    except SilentLimit:
        next_action = None
    return world, originally, next_action


def legal(world, action):
    return action == STOP or world.clone().step(action)


def fit_decisions(support, common, pair_worlds):
    fallback = fit(support, "masked")
    predictions = {}
    for mode in MODES:
        policy = fallback if mode == "masked" else fit(support, mode)
        history = blind_history = (BEGIN,)
        initial = common[0][0]
        for live, action in common:
            history = next_history(history, sensor(mode, live, initial), action)
            blind_history = next_history(blind_history, None, action)
        predictions[mode] = []
        for world in pair_worlds:
            obs = sensor(mode, observe(world), initial)
            action = predict(policy, obs, history)
            if action is None:
                action = predict(fallback, None, blind_history)
            if action not in ACTIONS and action != STOP:
                raise ValueError("No predicted next decision")
            predictions[mode].append(action)
        if mode != "live" and predictions[mode][0] != predictions[mode][1]:
            raise ValueError("A paired-invariant predictor switched")
    return predictions


def program_pairs(row):
    cases = row["examples"][:6]
    starts = [World.parse(c["inpgrid_json"]) for c in cases]
    traces = [c["actions"] for c in cases]
    demos = [demonstration(start, actions, World.parse(c["outgrid_json"]).state())
             for start, actions, c in zip(starts, traces, cases)]
    pairs = []
    for j, actions in enumerate(traces):
        if "move" not in actions or len(demos) != 6:
            pairs.append({"excluded": "no_move" if "move" not in actions else "no_five_supports",
                          "no_move": "move" not in actions, "no_five_supports": len(demos) != 6})
            continue
        size = actions.index("move") + 1
        arms = [oracle_at_arrival(starts[j], row["program_json"], actions[:size], m)
                for m in (0, 1)]
        worlds, original, labels = zip(*arms)
        if original[0] != original[1] or observe(worlds[0])[:3] != observe(worlds[1])[:3]:
            raise ValueError("Arms differ in non-marker sensor or pre-change marker")
        if observe(worlds[0])[3] or not observe(worlds[1])[3]:
            raise ValueError("Marker intervention failed")
        # All Karel predicates only observe marker presence before the next attempted action.
        original_next = actions[size] if size < len(actions) else STOP
        if labels[int(original[0] > 0)] != original_next:
            raise ValueError("The arm preserving original marker presence changed the next command")
        if None in labels:
            pairs.append({"excluded": "silent_limit", "zero_limit": labels[0] is None,
                          "one_limit": labels[1] is None})
            continue
        support = [demo for i, demo in enumerate(demos) if i != j]
        preds = fit_decisions(support, demos[j][:size], worlds)
        pairs.append({"prefix_actions": size, "original_marker": original[0], "oracle": labels,
                      "oracle_illegal": [not legal(w, a) for w, a in zip(worlds, labels)],
                      "pred": preds,
                      "pred_illegal": {m: [not legal(w, a) for w, a in zip(worlds, preds[m])] for m in MODES}})
    code = hashlib.sha256(" ".join(row["program_tokens"]).encode()).hexdigest()
    return {"guid": row["guid"], "code": code, "pairs": pairs}


def measures(selected):
    n = len(selected)
    metrics = {}
    for mode in MODES:
        counts = Counter()
        per_code = defaultdict(list)
        for code, pair in selected:
            p, y = pair["pred"][mode], pair["oracle"]
            z, o = p[0] == y[0], p[1] == y[1]
            sample = {"zero_correct": z, "one_correct": o, "both_correct": z and o,
                      "prediction_switched": p[0] != p[1],
                      "zero_illegal": pair["pred_illegal"][mode][0], "one_illegal": pair["pred_illegal"][mode][1]}
            counts.update(sample)
            per_code[code].append(sample)
        rates = {k: v/n for k, v in counts.items()} if n else {}
        macro = {k: mean(mean(t[k] for t in samples) for samples in per_code.values())
                 for k in counts}
        if n:
            for d in (rates, macro):
                d["balanced_correct"] = (d["zero_correct"] + d["one_correct"]) / 2
        metrics[mode] = {"counts": dict(counts), "world_rates": rates, "distinct_code_macro": macro}
    return {"worlds": n, "distinct_codes": len({code for code, _ in selected}),
            "oracle_illegal_by_arm": [sum(row["oracle_illegal"][i] for _, row in selected) for i in (0, 1)],
            "oracle_transitions": dict(Counter("->".join(r["oracle"]) for _, r in selected)),
            "policies": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("zip", "baseline", "protocol", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    receipt = json.loads(args.baseline.read_text())
    protocol = json.loads(args.protocol.read_text())
    helpers = {name: sha(Path(__file__).with_name(name)) for name in protocol["pinned_workers"]}
    if helpers != protocol["pinned_workers"] or receipt["inputs"]["val.zip"] != sha(args.zip):
        raise ValueError("Frozen cohort or helper changed")
    if receipt["worker_sha256"] != helpers["karel_trace_control.py"]:
        raise ValueError("Simulator differs from validated replay")
    count, worlds = 0, 0
    for row in records(args.zip):
        worlds += native_check(row)
        count += 1
    if (count, worlds) != (receipt["validation"]["programs"], receipt["validation"]["worlds"]):
        raise ValueError("Native interpreter coverage differs from replay receipt")
    rows = [program_pairs(row) for row in records(args.zip)]
    both = [(r["code"], p) for r in rows for p in r["pairs"] if "excluded" not in p]
    removed = [p for r in rows for p in r["pairs"] if "excluded" in p]
    switch = [(c, p) for c, p in both if p["oracle"][0] != p["oracle"][1]]
    stay = [(c, p) for c, p in both if p["oracle"][0] == p["oracle"][1]]
    if len(both) + len(removed) != worlds:
        raise ValueError("Incomplete pair accounting")
    groups = {"switch": measures(switch), "stay": measures(stay)}
    for mode in ("initial", "masked"):
        s = groups["switch"]["policies"][mode]["counts"]
        if s.get("both_correct", 0) or s.get("zero_correct", 0) + s.get("one_correct", 0) > len(switch):
            raise ValueError("Upper bound invalid")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    detail = out / "program_pairs.jsonl"
    with detail.open("x") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"))+"\n")
    result = {"job": job, "reference_checked_program_rows": count, "reference_checked_worlds": worlds,
              "distinct_program_codes": len({r["code"] for r in rows}), "eligible_pairs": len(both),
              "eligible_distinct_codes": len({c for c, _ in both}), "exclusions": dict(Counter(p["excluded"] for p in removed)),
              "excluded_attribute_counts": {k: sum(bool(p.get(k, False)) for p in removed)
                                            for k in ("no_move", "no_five_supports", "zero_limit", "one_limit")},
              "codes_without_eligible_pair": len({r["code"] for r in rows} - {c for c, _ in both}),
              "groups": groups, "continue_breadth_gate": groups["switch"]["distinct_codes"] >= 50,
              "baseline_sha256": sha(args.baseline), "validation_zip_sha256": sha(args.zip),
              "protocol_sha256": sha(args.protocol), "worker_sha256": sha(Path(__file__)),
              "pinned_helpers": helpers, "detail_sha256": sha(detail)}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"job": job, "native_worlds": worlds, "eligible": len(both),
                      "switch_pairs": len(switch), "switch_codes": groups["switch"]["distinct_codes"]}))


if __name__ == "__main__":
    main()
