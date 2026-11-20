"""Paired first-arrival Karel comparison on a separately prepared LEAPS release."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

from karel_aligned_vote import vote
from karel_online_control import STOP, demonstration
from karel_return_pairs import decisions, fit_decisions, legal, oracle_at_arrival
from karel_trace_control import sha
from leaps_karel_source_format import LeapsWorld as SourceWorld


POLICIES = ("pooled", "marker", "gated", "live", "initial", "masked")


def check_original(row):
    cases = row["examples"]
    if len(cases) != 6:
        raise ValueError("Source cohort must have six demonstrations per program")
    for i, case in enumerate(cases):
        w = SourceWorld.parse(case["inpgrid_json"])
        end = SourceWorld.parse(case["outgrid_json"]).state()
        stream = decisions(row["program_json"], w)
        for t, expected in enumerate(case["actions"]):
            if next(stream, STOP) != expected or not w.step(expected):
                raise ValueError(f"Native trace/legality mismatch at {row['guid']}/{i}/{t}")
        if next(stream, STOP) != STOP or w.state() != end:
            raise ValueError(f"Native terminal mismatch at {row['guid']}/{i}")


def evaluate(row):
    cases = row["examples"]
    starts = [SourceWorld.parse(c["inpgrid_json"]) for c in cases]
    demos = [demonstration(w, c["actions"], SourceWorld.parse(c["outgrid_json"]).state())
             for w, c in zip(starts, cases)]
    sizes, aligned = [], []
    for c, demo in zip(cases, demos):
        actions = c["actions"]
        n = actions.index("move")+1 if "move" in actions else None
        sizes.append(n)
        aligned.append(demo[n] if n is not None else None)
    rows = []
    for j, n in enumerate(sizes):
        support = aligned[:j] + aligned[j+1:]
        if n is None or None in support:
            rows.append({"excluded": "target_no_move" if n is None else "support_no_move",
                         "target_no_move": n is None, "support_no_move": None in support})
            continue
        actions = cases[j]["actions"]
        arms = [oracle_at_arrival(starts[j], row["program_json"], actions[:n], m) for m in (0, 1)]
        worlds, original, labels = zip(*arms)
        if original[0] != original[1] or labels[int(original[0] > 0)] != aligned[j][1]:
            raise ValueError("Original marker-preserving arm failed")
        if None in labels:
            rows.append({"excluded": "silent_limit", "zero_limit": labels[0] is None,
                         "one_limit": labels[1] is None})
            continue
        choices = vote(support)
        choices.update(fit_decisions(demos[:j] + demos[j+1:], demos[j][:n], worlds))
        if any(choices[m][0] != choices[m][1] for m in ("pooled", "initial", "masked")):
            raise ValueError("Input-invariant comparator switched")
        rows.append({"labels": labels, "choices": choices,
                     "oracle_illegal": [not legal(w, a) for w, a in zip(worlds, labels)]})
    code = hashlib.sha256(" ".join(row["program_tokens"]).encode()).hexdigest()
    return {"guid": row["guid"], "code": code, "pairs": rows}


def group(items):
    result = {"worlds": len(items), "distinct_codes": len({c for c, _ in items}),
              "oracle_illegal_by_arm": [sum(p["oracle_illegal"][i] for _, p in items) for i in (0, 1)],
              "methods": {}, "gated_difference": {}}
    if not items:
        return result
    for method in POLICIES:
        by_code, totals, sizes = defaultdict(Counter), Counter(), Counter()
        for c, p in items:
            y, z = p["labels"]
            a, b = p["choices"][method]
            scores = {"zero_correct": a == y, "one_correct": b == z, "both_correct": a == y and b == z,
                      "predicted_switch": a != b, "false_switch": a != b and y == z}
            by_code[c].update(scores)
            totals.update(scores)
            sizes[c] += 1
        macro = {k: mean(v[k] / sizes[c] for c, v in by_code.items()) for k in totals}
        rates = {k: n / len(items) for k, n in totals.items()}
        for d in (macro, rates):
            d["balanced_correct"] = (d["zero_correct"] + d["one_correct"])/2
        result["methods"][method] = {"counts": dict(totals), "world_rates": rates, "code_macro": macro}
    for comp in ("pooled", "marker"):
        measures = {}
        for metric in ("both_correct", "false_switch"):
            win, loss = 0, 0
            for _, p in items:
                preds, y = p["choices"], p["labels"]
                if metric == "both_correct":
                    g, c = tuple(preds["gated"]) == y, tuple(preds[comp]) == y
                else:
                    g = y[0] == y[1] and preds["gated"][0] != preds["gated"][1]
                    c = y[0] == y[1] and preds[comp][0] != preds[comp][1]
                win += g and not c
                loss += c and not g
            measures[metric] = {"gated_only": win, "comparator_only": loss, "delta": (win-loss)/len(items),
                                "code_macro_delta": result["methods"]["gated"]["code_macro"][metric] -
                                result["methods"][comp]["code_macro"][metric]}
        result["gated_difference"][comp] = measures
    return result


def analyze(rows):
    items = [(r["code"], p) for r in rows for p in r["pairs"] if "excluded" not in p]
    dropped = [p for r in rows for p in r["pairs"] if "excluded" in p]
    s = {"program_rows": len(rows), "distinct_codes": len({r["code"] for r in rows}), "worlds": 6*len(rows),
         "eligible_worlds": len(items), "exclusions": dict(Counter(p["excluded"] for p in dropped)),
         "overlapping_exclusions": {k: sum(p.get(k, False) for p in dropped)
                                    for k in ("target_no_move", "support_no_move", "zero_limit", "one_limit")}}
    if len(rows) != s["distinct_codes"] or 6*len(rows) != len(items) + len(dropped):
        raise ValueError("External sample identities or eligibility accounting invalid")
    for name, selected in (("original", items), ("both_oracle_legal", [(c,p) for c,p in items if not any(p["oracle_illegal"])])):
        s[name] = {"all": group(selected),
                   "oracle_switch": group([(c,p) for c,p in selected if p["labels"][0] != p["labels"][1]]),
                   "oracle_stay": group([(c,p) for c,p in selected if p["labels"][0] == p["labels"][1]])}
    s["continue_breadth_gate"] = s["original"]["oracle_switch"]["distinct_codes"] >= 50
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("cohort", "receipt", "protocol", "out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    for arg in ("cohort-sha", "receipt-sha"):
        parser.add_argument("--"+arg,required=True)
    parser.add_argument("--program-count", type=int, required=True)
    a = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    protocol = json.loads(a.protocol.read_text())
    helpers = {n:sha(Path(__file__).with_name(n)) for n in protocol["pinned_workers"]}
    if helpers != protocol["pinned_workers"] or sha(a.cohort) != a.cohort_sha or sha(a.receipt) != a.receipt_sha:
        raise ValueError("Bad frozen cohort, receipt or worker digest")
    receipt = json.loads(a.receipt.read_text())
    if receipt["completed"] is not True or receipt["native_validity_passed"] is not True:
        raise ValueError("Native source preparation failed")
    source_format_sha = sha(Path(__file__).with_name("leaps_karel_source_format.py"))
    if receipt["code_sha256"]["leaps_karel_source_format.py"] != source_format_sha or receipt["protocol_sha256"] != protocol["source_protocol_sha256"]:
        raise ValueError("Scoring world or source preparation protocol differs from the receipt")
    if receipt["artifact_sha256"] != a.cohort_sha or receipt["validated_stop"] != receipt["selected_worlds"]:
        raise ValueError("Source receipt does not cover all terminations or the supplied cohort")
    if receipt["native_states"] != receipt["native_steps"] + receipt["selected_worlds"] or receipt["native_observations"] != receipt["native_states"]:
        raise ValueError("Source intermediate-state/observation coverage incomplete")
    with a.cohort.open() as f:
        data = [json.loads(line) for line in f]
    if len(data) != a.program_count or a.program_count != 640 or receipt["selected_programs"] != len(data) or receipt["selected_worlds"] != 6*len(data):
        raise ValueError("Incomplete/preparation-mismatched external cohort")
    for row in data:
        check_original(row)
    if len({r["guid"] for r in data}) != len(data):
        raise ValueError("A selected external program ID was duplicated")
    eligible = [r for r in data if r["eligible"] is True]
    withheld = Counter(r["exclusion"] for r in data if r["eligible"] is False)
    if len(eligible) != receipt["eligible_programs"] or len(eligible) + sum(withheld.values()) != len(data):
        raise ValueError("Source eligibility mismatch")
    rows = [evaluate(row) for row in eligible]
    s = analyze(rows)
    a.out.mkdir(parents=True, exist_ok=False)
    details = a.out/"decisions.jsonl"
    with details.open("x") as f:
        for r in rows:
            f.write(json.dumps(r,separators=(",",":"))+"\n")
    s.update(job=job, cohort_sha256=a.cohort_sha, source_receipt_sha256=a.receipt_sha, helpers=helpers,
             protocol_sha256=sha(a.protocol), worker_sha256=sha(Path(__file__)), details_sha256=sha(details),
             source_format_sha256=source_format_sha, selected_programs=len(data), source_excluded_programs=dict(withheld),
             native_exact_trace_and_terminal_worlds=6*len(data))
    (a.out/"summary.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"job": job, "program_rows": len(rows), "eligible": s["eligible_worlds"],
                      "switching_distinct_codes": s["original"]["oracle_switch"]["distinct_codes"]}))


if __name__ == "__main__":
    main()
