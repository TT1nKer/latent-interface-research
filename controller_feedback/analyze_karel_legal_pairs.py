"""Post-hoc sensitivity: retain pairs where both expert commands are legal."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def analyze(folder):
    summary = json.loads((folder / "summary.json").read_text())
    path = folder / "program_pairs.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["detail_sha256"]:
        raise ValueError("Paired rows do not match the summary")
    counts = defaultdict(Counter)
    codes = defaultdict(set)
    with path.open() as rows:
        for line in rows:
            record = json.loads(line)
            for pair in record["pairs"]:
                if "excluded" in pair:
                    continue
                first, second = pair["oracle"]
                group = "stay" if first == second else "switch"
                strata = ["all"] if any(pair["oracle_illegal"]) else ["all", "both_legal"]
                for stratum in strata:
                    key = (stratum, group)
                    counts[key]["pairs"] += 1
                    codes[key].add(record["code"])
                    for policy, (a, b) in pair["pred"].items():
                        counts[key][policy + "_both_correct"] += (a == first and b == second)
                        counts[key][policy + "_switched"] += a != b
    output = {}
    for (stratum, group), row in sorted(counts.items()):
        if stratum == "all":
            reference = summary["groups"][group]
            assert row["pairs"] == reference["worlds"]
            for policy, original in reference["policies"].items():
                assert row[policy + "_both_correct"] == original["counts"]["both_correct"]
                assert row[policy + "_switched"] == original["counts"]["prediction_switched"]
        output.setdefault(stratum, {})[group] = dict(row, distinct_codes=len(codes[stratum, group]))
    assert sum(row["pairs"] for row in output["all"].values()) == summary["eligible_pairs"]
    return {"type": "post_hoc_descriptive", "definition": "Both next commands requested by the expert program can legally execute, or are STOP.", "counts": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as output:
        json.dump(analyze(args.folder), output, indent=2)
        output.write("\n")
