"""Fit first-move-aligned next-command votes on five other Karel demonstrations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import mean
import zipfile

from karel_predicates import observe
from karel_trace_control import World, sha


STOP = "STOP"
METHODS = ("pooled", "marker", "gated")


def first_arrival(case):
    world = World.parse(case["inpgrid_json"])
    for i, action in enumerate(case["actions"]):
        if not world.step(action):
            raise ValueError("Invalid original support primitive")
        if action == "move":
            actions = case["actions"]
            return observe(world), actions[i + 1] if i + 1 < len(actions) else STOP
    return None


def modes(labels):
    counts = Counter(labels)
    if not counts:
        return []
    most = max(counts.values())
    return sorted((a for a, n in counts.items() if n == most), key=lambda a: (a != STOP, a))


def vote(support):
    if len(support) != 5 or None in support:
        raise ValueError("Expected five eligible aligned supports")
    pooled = modes(action for _, action in support)[0]
    by_marker = [modes(a for obs, a in support if obs[3] == level) for level in (False, True)]
    split = all(len(v) == 1 for v in by_marker) and by_marker[0][0] != by_marker[1][0]
    marker = [v[0] if v else pooled for v in by_marker]
    return {"pooled": [pooled, pooled], "marker": marker, "gated": marker if split else [pooled, pooled]}


def extract(zip_path, rows_path):
    result, ignored = [], Counter()
    arrivals = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as archive, archive.open("val.json") as source, rows_path.open() as old:
        records = (json.loads(line) for line in itertools.islice(source, 2500))
        for data, line in itertools.zip_longest(records, old):
            if data is None or line is None:
                raise ValueError("Frozen input row count mismatch")
            key = hashlib.sha256(" ".join(data["program_tokens"]).encode()).hexdigest()
            prior = json.loads(line)
            cases = data["examples"][:6]
            if (data["guid"], key, len(cases)) != (prior["guid"], prior["code"], 6) or len(prior["pairs"]) != 6:
                raise ValueError("Frozen input row mismatch")
            # Each support contributes exactly one observation and one command.
            # The returned choices below have no access to either arm's reference label.
            demos = [first_arrival(c) for c in cases]
            arrivals.update((json.dumps([data["guid"], demos],separators=(",",":"))+"\n").encode())
            decisions = []
            for j, original in enumerate(prior["pairs"]):
                if "excluded" in original:
                    ignored[original["excluded"]] += 1
                    continue
                choices = vote(demos[:j] + demos[j+1:])
                labels = tuple(original["oracle"])
                if demos[j] is None or labels[int(demos[j][0][3])] != demos[j][1]:
                    raise ValueError("Aligned decision disagrees with verified original native arm")
                decisions.append({"target_index": j, "labels": labels, "both_legal": not any(original["oracle_illegal"]),
                                  "choices": choices})
            result.append({"guid": data["guid"], "code": key, "decisions": decisions})
    return result, ignored, arrivals.hexdigest()


def score_group(items):
    out = {"worlds": len(items), "distinct_codes": len({code for code, _ in items}), "methods": {}, "gated_difference": {}}
    if not items:
        return out
    for method in METHODS:
        counts = Counter()
        by_code = defaultdict(Counter)
        sizes = Counter()
        for code, item in items:
            a, b = item["choices"][method]
            y, z = item["labels"]
            score = {"zero_correct": a == y, "one_correct": b == z, "both_correct": a == y and b == z,
                     "predicted_switch": a != b, "false_switch": y == z and a != b}
            counts.update(score)
            by_code[code].update(score)
            sizes[code] += 1
        rates = {k: n/len(items) for k, n in counts.items()}
        macro = {k: mean(vals[k]/sizes[code] for code, vals in by_code.items()) for k in counts}
        for d in (rates, macro):
            d["balanced_correct"] = (d["zero_correct"] + d["one_correct"])/2
        out["methods"][method] = {"counts": dict(counts), "world_rates": rates, "code_macro": macro}
    for comparator in ("pooled", "marker"):
        diffs = {}
        for target in ("both_correct", "predicted_switch"):
            gated_only, comparator_only = 0, 0
            for _, item in items:
                pair = item["choices"]
                if target == "both_correct":
                    g = tuple(pair["gated"]) == item["labels"]
                    c = tuple(pair[comparator]) == item["labels"]
                else:
                    g = pair["gated"][0] != pair["gated"][1]
                    c = pair[comparator][0] != pair[comparator][1]
                gated_only += g and not c
                comparator_only += c and not g
            diffs[target] = {"gated_only": gated_only, "comparator_only": comparator_only,
                             "world_rate_delta": (gated_only-comparator_only)/len(items),
                             "code_macro_delta": out["methods"]["gated"]["code_macro"][target]
                             - out["methods"][comparator]["code_macro"][target]}
        out["gated_difference"][comparator] = diffs
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("zip", "paired", "protocol", "out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    a = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    prior_summary = a.paired/"summary.json"
    prior_rows = a.paired/"program_pairs.jsonl"
    files = {"validation_zip_sha256": a.zip, "paired_summary_sha256": prior_summary,
             "paired_rows_sha256": prior_rows, "simulator_sha256": Path(__file__).with_name("karel_trace_control.py"),
             "predicates_sha256": Path(__file__).with_name("karel_predicates.py")}
    digests = {k: sha(p) for k, p in files.items()}
    if digests != json.loads(a.protocol.read_text())["inputs"]:
        raise ValueError("Frozen artifact mismatch")
    rows, ignored, arrival_digest = extract(a.zip, prior_rows)
    prior = json.loads(prior_summary.read_text())
    items = [(r["code"], t) for r in rows for t in r["decisions"]]
    if (len(rows), len(items), dict(ignored)) != (prior["reference_checked_program_rows"], prior["eligible_pairs"], prior["exclusions"]):
        raise ValueError("Target eligibility disagrees with frozen cohort")
    out = {"job": job, "source_digests": digests, "arrival_rows_sha256": arrival_digest,
           "program_rows": len(rows), "distinct_codes": len({r["code"] for r in rows}), "excluded": dict(ignored),
           "protocol_sha256": sha(a.protocol), "worker_sha256": sha(Path(__file__))}
    for name, selected in (("original", items), ("both_oracle_legal", [(c, t) for c, t in items if t["both_legal"]])):
        out[name] = {"all": score_group(selected), "oracle_switch": score_group([(c, t) for c, t in selected if t["labels"][0] != t["labels"][1]]),
                     "oracle_stay": score_group([(c, t) for c, t in selected if t["labels"][0] == t["labels"][1]])}
    for group in ("switch", "stay"):
        got, want = out["original"]["oracle_"+group], prior["groups"][group]
        if (got["worlds"], got["distinct_codes"]) != (want["worlds"], want["distinct_codes"]):
            raise ValueError("Reference branch membership changed")
    a.out.mkdir(parents=True, exist_ok=False)
    detail = a.out/"decisions.jsonl"
    with detail.open("x") as f:
        for r in rows:
            f.write(json.dumps(r,separators=(",",":"))+"\n")
    out["detail_sha256"] = sha(detail)
    (a.out/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"job": job, "eligible_worlds": len(items), "programs": len(rows),
                      "detail_sha256": out["detail_sha256"]}))


if __name__ == "__main__":
    main()
