"""Post-fit position ranking relative to native Karel static primitive sites."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import time

from karel_predicates import observe
from karel_return_pairs import condition
from karel_trace_control import ACTIONS, sha


METHODS = ("retrieval_17", "random_17", "nearest_index")
GROUPS = ("all", "site_available", "hard")


def sites(program, world):
    """Yield a primitive and the lexical execution site; caller commits its state change."""
    silent = 0

    def tick():
        nonlocal silent
        silent += 1
        if silent > 4096:
            raise ValueError("Unbounded silent traversal in original native program")

    def block(nodes, path):
        nonlocal silent
        for i, node in enumerate(nodes):
            here = path + (i,)
            k = node["type"]
            if k in ACTIONS:
                silent = 0
                yield k, here
            elif k == "repeat":
                for _ in range(node["times"]):
                    tick()
                    yield from block(node["body"], here+("body",))
            elif k == "while":
                while True:
                    tick()
                    if not condition(node["condition"], world):
                        break
                    yield from block(node["body"], here+("body",))
            elif k in ("if", "ifElse"):
                tick()
                go = condition(node["condition"], world)
                if k == "ifElse":
                    body = "ifBody" if go else "elseBody"
                    yield from block(node[body], here+(body,))
                elif go:
                    yield from block(node["body"], here+("body",))
            else:
                raise ValueError(f"Unsupported AST site {k!r}")
    yield from block(program["run"], ("run",))


def label_record(row, world_cls):
    trajectories = []
    for j, example in enumerate(row["examples"][:6]):
        w = world_cls.parse(example["inpgrid_json"])
        cursor = sites(row["program_json"], w)
        demo, gold_sites = [], []
        for t, expected in enumerate(example["actions"]):
            found = next(cursor, None)
            if found is None or found[0] != expected:
                raise ValueError(f"Native site/action mismatch at {row['guid']}/{j}/{t}")
            demo.append((observe(w), expected))
            gold_sites.append(found[1])
            if not w.step(expected):
                raise ValueError("Native reference primitive crashed")
        if next(cursor, None) is not None or w.state() != world_cls.parse(example["outgrid_json"]).state():
            raise ValueError(f"Native site/termination mismatch at {row['guid']}/{j}")
        demo.append((observe(w), "STOP"))
        trajectories.append((demo, gold_sites))
    if len(trajectories) != 6:
        raise ValueError("Missing native world")
    return trajectories


class PositionReader:
    def __init__(self, model, supports):
        from karel_unaligned_model import NeuralPolicy
        self.policy = NeuralPolicy(model, supports)
        self.verified = False

    def weights(self, obs):
        import torch
        from karel_unaligned_model import FEATURES
        policy = self.policy
        x = torch.zeros(1, 1, FEATURES)
        x[0, 0, :4] = torch.tensor(obs, dtype=torch.float32)
        x[0, 0, 4 + policy.previous] = 1.
        q, policy.pending_state = policy.model.encode(x, policy.state)
        keys, log_prior, votes = policy.memory
        weights = policy.model.predict(q, (keys, log_prior, None))
        if not self.verified:
            if not torch.equal(weights.bmm(votes), policy.model.predict(q, policy.memory)):
                raise ValueError("Position weights do not reproduce actual action votes")
            self.verified = True
        return weights[0, 0].tolist()

    def advance(self, obs, action):
        self.policy.advance(obs, action)


def tied_score(values, candidates, gold, clock=None):
    if not candidates:
        return 0.
    positions = [(values[k] if clock is None else -abs(clock-j), site == gold)
                 for k, j, site in candidates]
    maximum = max(value for value, _ in positions)
    tied = [correct for value, correct in positions if value == maximum]
    return sum(tied)/len(tied)


def target_stats(trajectories, target_idx, models):
    query, target_sites = trajectories[target_idx]
    supports = trajectories[:target_idx]+trajectories[target_idx+1:]
    demos = [d for d, _ in supports]
    padded = max(map(len, demos))
    candidates = defaultdict(list)
    for donor, (demo, labels) in enumerate(supports):
        for j, site in enumerate(labels):
            candidates[demo[j][1]].append((donor*padded+j, j, site))
    readers = {name: PositionReader(model, demos) for name, model in models.items()}
    groups = {g: {"steps": 0, "scores": dict.fromkeys(METHODS, 0.)} for g in GROUPS}
    action_covered = 0
    for t, site in enumerate(target_sites):
        obs = query[t][0]
        # The next target action is read by the external scorer only after obtaining both distributions.
        distributions = {m: reader.weights(obs) for m, reader in readers.items()}
        action = query[t][1]
        matched = candidates[action]
        available = any(s == site for _, _, s in matched)
        hard = available and any(s != site for _, _, s in matched)
        scores = {m: tied_score(p, matched, site) for m, p in distributions.items()}
        scores["nearest_index"] = tied_score(None, matched, site, clock=t)
        action_covered += bool(matched)
        for group, include in (("all", True), ("site_available", available), ("hard", hard)):
            if include:
                dest = groups[group]
                dest["steps"] += 1
                for m in METHODS:
                    dest["scores"][m] += scores[m]
        for reader in readers.values():
            reader.advance(obs, action)
    return {"primitive_steps": len(target_sites), "no_move": all(a != "move" for _, a in query),
            "action_available_steps": action_covered, "groups": groups}


def summarize(rows):
    worlds = [(r["code"], r["program_id"], t) for r in rows for t in r["targets"]]
    result = {"program_rows": len(rows), "distinct_codes": len({r["code"] for r in rows}),
              "worlds": len(worlds), "primitive_steps": sum(t["primitive_steps"] for _, _, t in worlds),
              "zero_primitive_worlds": sum(t["primitive_steps"] == 0 for _, _, t in worlds),
              "no_move_worlds": sum(t["no_move"] for _, _, t in worlds),
              "action_available_steps": sum(t["action_available_steps"] for _, _, t in worlds), "groups": {}}
    for group in GROUPS:
        eligible = [(c, g, t) for c, g, t in worlds if t["groups"][group]["steps"]]
        by_code = defaultdict(list)
        for c, _, t in eligible:
            by_code[c].append(t["groups"][group])
        n = sum(t["groups"][group]["steps"] for _, _, t in eligible)
        metrics = {}
        for m in METHODS:
            total = sum(t["groups"][group]["scores"][m] for _, _, t in eligible)
            rates = [sum(q["scores"][m] for q in part)/sum(q["steps"] for q in part)
                     for part in by_code.values()]
            metrics[m] = {"expected_top1_hits": total, "step_micro": total/n if n else None,
                          "distinct_code_macro": mean(rates) if rates else None}
        result["groups"][group] = {"program_rows": len({g for _, g, _ in eligible}),
                                    "distinct_codes": len(by_code), "worlds": len(eligible), "steps": n,
                                    "no_move_worlds": sum(t["no_move"] for _, _, t in eligible), "methods": metrics}
    return result


def main(a):
    import torch
    from karel_unaligned_evaluate import load_banks, code
    from karel_unaligned_model import load_trained, read_checkpoint
    job = os.environ["SLURM_JOB_ID"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training_path = a.checkpoint.with_name("training_summary.json")
    tr = json.loads(training_path.read_text())
    payload = read_checkpoint(a.checkpoint)
    if not tr["completed"] or sha(a.checkpoint) != tr["checkpoint_sha256"] or tr["protocol_sha256"] != sha(a.protocol):
        raise ValueError("Training receipt differs from actual checkpoint and protocol")
    for k in ("model_sha256", "trainer_sha256", "protocol_sha256", "source_sha256", "seeds"):
        if tr[k] != payload[k]:
            raise ValueError(f"Checkpoint metadata mismatch: {k}")
    for k, name in (("model_sha256", "karel_unaligned_model.py"), ("trainer_sha256", "karel_unaligned_train.py")):
        if tr[k] != sha(Path(__file__).with_name(name)):
            raise ValueError("Fitted code mismatch")
    codes = set(payload["training_codes"])
    reg = hashlib.sha256(("\n".join(sorted(codes))+"\n").encode()).hexdigest()
    if reg != tr["training_code_sha256"] or len(codes) != tr["training_code_count"] or tr["source_sha256"] != sha(a.dev_zip):
        raise ValueError("Fitting data manifest mismatch")
    panels = load_banks(a,codes)
    native = {}
    for domain, cls, rows in panels:
        native[domain] = [(code(row),row["guid"],label_record(row,cls)) for row in rows]
    models = load_trained(a.checkpoint)
    readers = {"retrieval_17": models["retrieval", 17], "random_17": models["initial", 17]}
    a.out.mkdir(parents=True, exist_ok=False)
    private = a.out/"per_program.jsonl"
    start = time.monotonic()
    summary = {}
    with private.open("x") as out, torch.inference_mode():
        for domain, records in native.items():
            rows = []
            for code_id, guid, trajectories in records:
                row = {"domain":domain,"program_id":guid,"code":code_id,
                       "targets":[target_stats(trajectories,j,readers) for j in range(6)]}
                rows.append(row)
                out.write(json.dumps(row,separators=(",",":"))+"\n")
            summary[domain] = summarize(rows)
    result = {"job":job, "completed":True, "all_native_actions_stop_and_endpoint_verified":True, "domains":summary,
              "diagnostic_is_separate_from_primary_endpoint":True,"checkpoint_sha256":sha(a.checkpoint),
              "original_protocol_sha256":sha(a.protocol),"diagnostic_protocol_sha256":sha(a.site_protocol),
              "weights_definition":"original MetricPolicy.predict, before the support-label vote",
              "source_receipt_sha256":sha(a.source_receipt),"external_sha256":sha(a.external),
              "training_summary_sha256":sha(training_path),"worker_sha256":sha(Path(__file__)),
              "model_sha256":tr["model_sha256"],"evaluator_sha256":sha(Path(__file__).with_name("karel_unaligned_evaluate.py")),
              "per_program_sha256":sha(private),"scoring_wall_s":round(time.monotonic()-start,3)}
    (a.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["groups"]["hard"] for k,v in summary.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("checkpoint","protocol","site-protocol","dev-zip","external","source-receipt","external-source-protocol","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    main(parser.parse_args())
