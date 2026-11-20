"""Post-hoc slot ranking for the exact support-weighted prefix aligner."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
from statistics import mean

from karel_trace_control import sha
from karel_unaligned_baselines import PrefixAlignment, build, canonical_support, choose, obs_code, support_weights
from karel_unaligned_sites import GROUPS, label_record, tied_score


def target(trajectories, idx, old):
    query, sites = trajectories[idx]
    other = trajectories[:idx]+trajectories[idx+1:]
    demos = [demo for demo, _ in other]
    normalized = canonical_support(demos)
    expected = tuple(tuple((tuple(obs), a) for obs, a in demo) for demo in demos)
    if normalized != expected:
        raise ValueError("Frozen support ordering differs from the reference-site ordering")
    aligner = PrefixAlignment(normalized, support_weights(normalized))
    acting = build("dtw_support_weighted", demos)
    candidates = defaultdict(list)
    slots = []
    for d, (demo, gold) in enumerate(other):
        if len(demo) != len(gold)+1 or demo[-1][1] != "STOP":
            raise ValueError("Donor positions do not end at STOP")
        for j, (x, a) in enumerate(demo):
            if aligner.donors[d][j] != (obs_code(x), a) or acting.donors[d][j] != aligner.donors[d][j]:
                raise ValueError("Donor slot features or primitive differ from the original")
            if j < len(gold):
                candidates[a].append((len(slots),j,gold[j]))
            slots.append((d, j))
    counts, correct = Counter(), Counter()
    for t, (obs, executed) in enumerate(query):
        q = obs_code(obs)
        scores = []
        winners = []
        # No access to the upcoming expert action or source site while ranking donor slots.
        for donor, distances in zip(aligner.donors, aligner.prefix):
            block = [-dist-aligner.cost[q ^ bits] for dist, (bits, _) in zip(distances, donor)]
            best = max(block)
            labels = {a for v, (_, a) in zip(block, donor) if v == best}
            winners.append((best, labels))
            scores.extend(block)
        top = max(best for best, _ in winners)
        votes = Counter()
        for best, labels in winners:
            if best == top:
                for a in labels:
                    votes[a] += 60//len(labels)
        if choose(votes) != acting.act(obs) or len(scores) != len(slots):
            raise ValueError("Unfiltered slot costs disagree with the actual acting aligner")
        if t < len(sites):
            avail = candidates[executed]
            same = any(s == sites[t] for _, _, s in avail)
            hard = same and any(s != sites[t] for _, _, s in avail)
            hit = tied_score(scores, avail, sites[t])
            for g, yes in (("all",True),("site_available",same),("hard",hard)):
                if yes:
                    counts[g] += 1
                    correct[g] += hit
            aligner.advance(obs, executed)
            acting.advance(obs, executed)
    if old["primitive_steps"] != len(sites) or old["no_move"] != all(a != "move" for _, a in query):
        raise ValueError("Original world identity mismatch")
    for g in GROUPS:
        if counts[g] != old["groups"][g]["steps"]:
            raise ValueError("Original teacher-prefix site coverage mismatch")
    return {"world":idx,"no_move":old["no_move"],"action_vote_checks":len(query),
            "groups":{g:{"steps":counts[g],"dp_hits":correct[g],
                         "trained_hits":old["groups"][g]["scores"]["retrieval_17"]} for g in GROUPS}}


def aggregate(rows):
    result = {}
    for g in GROUPS:
        by_code = defaultdict(list)
        worlds, n_no_move = 0, 0
        for row in rows:
            for t in row["targets"]:
                if t["groups"][g]["steps"]:
                    by_code[row["code"]].append(t["groups"][g])
                    worlds += 1
                    n_no_move += t["no_move"]
        score = {"distinct_codes":len(by_code),"worlds":worlds,"no_move_worlds":n_no_move,
                 "steps":sum(x["steps"] for vals in by_code.values() for x in vals)}
        rates = {}
        for label in ("trained","dp"):
            ordered = []
            for code in sorted(by_code):
                vals = by_code[code]
                ordered.append(sum(x[label+"_hits"] for x in vals)/sum(x["steps"] for x in vals))
            rates[label] = ordered
            hits = sum(x[label+"_hits"] for vals in by_code.values() for x in vals)
            score[label] = {"micro":hits/score["steps"] if score["steps"] else None,
                            "code_macro":mean(ordered) if ordered else None, "expected_hits":hits}
        delta = [a-b for a,b in zip(rates["trained"],rates["dp"])]
        score["trained_minus_dp_code_macro"] = mean(delta) if delta else None
        if g == "hard" and delta:
            rng = random.Random(20261120)
            draws = sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(5000))
            score["paired_program_descriptive_95"] = [draws[124],draws[4874]]
        result[g] = score
    return result


def main(a):
    from karel_unaligned_model import read_checkpoint
    from karel_unaligned_evaluate import load_banks, code
    job = os.environ["SLURM_JOB_ID"]
    protocol = json.loads(a.dp_protocol.read_text())
    for name, wanted in protocol["pinned"].items():
        if sha(Path(__file__).with_name(name)) != wanted:
            raise ValueError(f"Frozen code changed: {name}")
    old_summary = a.prior/"summary.json"
    old_detail = a.prior/"per_program.jsonl"
    if sha(old_summary) != protocol["original_summary_sha256"] or sha(old_detail) != protocol["original_per_program_sha256"]:
        raise ValueError("Original diagnostic artifacts changed")
    prior = json.loads(old_summary.read_text())
    if not prior["completed"] or sha(a.checkpoint) != prior["checkpoint_sha256"] or sha(a.source_receipt) != prior["source_receipt_sha256"]:
        raise ValueError("Checkpoint or original source identity changed")
    codes = set(read_checkpoint(a.checkpoint)["training_codes"])
    banks = load_banks(a,codes)
    original = {}
    with old_detail.open() as f:
        for line in f:
            r = json.loads(line)
            key = r["domain"],r["program_id"]
            if key in original:
                raise ValueError("Duplicate previous row")
            original[key] = r
    results = {}
    a.out.mkdir(parents=True,exist_ok=False)
    detail = a.out/"per_program.jsonl"
    with detail.open("x") as out:
        for domain, cls, records in banks:
            rows = []
            for record in records:
                old = original.pop((domain,record["guid"]))
                code_id = code(record)
                if old["code"] != code_id or len(old["targets"]) != 6:
                    raise ValueError("Identity or fold count changed")
                t = label_record(record, cls)
                r = {"domain":domain,"program_id":record["guid"],"code":code_id,
                     "targets":[target(t,i,old["targets"][i]) for i in range(6)]}
                out.write(json.dumps(r,separators=(",",":"))+"\n")
                rows.append(r)
            results[domain] = aggregate(rows)
            for g in GROUPS:
                now, before = results[domain][g], prior["domains"][domain]["groups"][g]
                if any(now[k] != before[k] for k in ("worlds","distinct_codes","steps","no_move_worlds")):
                    raise ValueError("Original stratum denominators changed")
                trained = before["methods"]["retrieval_17"]["distinct_code_macro"]
                if abs(now["trained"]["code_macro"]-trained) > 1e-12:
                    raise ValueError("Original trained model's summary did not reproduce")
            results[domain]["checked_teacher_boundaries_including_stop"] = sum(t["action_vote_checks"] for r in rows for t in r["targets"])
    if original:
        raise ValueError("Missing original diagnostic rows")
    s = {"job":job,"completed":True,"posthoc_method_selected_on_endpoints":True,"domains":results,
         "checkpoint_sha256":sha(a.checkpoint),"original_summary_sha256":sha(old_summary),
         "original_per_program_sha256":sha(old_detail),"protocol_sha256":sha(a.dp_protocol),
         "pinned_workers":protocol["pinned"],"worker_sha256":sha(Path(__file__)),"per_program_sha256":sha(detail)}
    (a.out/"summary.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["hard"] for k,v in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint","prior","dp-protocol","dev-zip","external","source-receipt","external-source-protocol","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    main(parser.parse_args())
