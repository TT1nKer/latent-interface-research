import csv
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path


path = Path(sys.argv[1])
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "454f45ed1a7ac8575bf544228eec456eab7b8fca0673c909b6506e3fdf545f4a"
)
with path.open(newline="") as stream:
    all_rows = list(csv.DictReader(stream))

permutation_rows = [row for row in all_rows if row["block"] == "perm"]
problems = sorted({row["instance"] for row in permutation_rows})
configs = sorted({row["config"] for row in permutation_rows})
forms = (0, 1, 2, 3)
table = {}
for row in all_rows:
    if row["instance"] not in problems or row["block"] not in ("main", "perm"):
        continue
    assert row["eps"] == "1e-06" and row["repeat"] == "0"
    key = row["instance"], int(row["k"]), row["config"]
    assert key not in table
    verified = row["verified_solve"] == "True" and float(row["solve_wall_s"]) <= 10
    cpu = min(10.0, float(row["solve_cpu_s"])) if verified else 10.0
    # Both columns are losses: negate the verified-completion indicator.
    table[key] = -int(verified), cpu
assert (len(problems), len(configs)) == (30, 7)
assert set(table) == set(itertools.product(problems, forms, configs))

mean = statistics.mean
results = {}
for column, (metric, display_sign) in enumerate([
    ("verified_completion", -1), ("capped_cpu_seconds", 1)
]):
    by_problem = []
    for problem in problems:
        losses = [[table[problem, k, arm][column] for arm in configs] for k in forms]
        per_form_min = [min(row) for row in losses]
        invariant_min = min(mean(loss_per_config) for loss_per_config in zip(*losses))
        canonical_winners = [i for i, loss in enumerate(losses[0]) if loss == per_form_min[0]]
        transfer_loss = mean(mean(row[i] for i in canonical_winners) for row in losses[1:])
        has_shared_winner = any(
            all(row[i] == best for row, best in zip(losses, per_form_min))
            for i in range(len(configs))
        )
        by_problem.append({
            "orbit": mean(per_form_min),
            "invariant": invariant_min,
            "regret": invariant_min - mean(per_form_min),
            "transfer": transfer_loss,
            "best_other_three": mean(per_form_min[1:]),
            "has_shared_winner": has_shared_winner,
        })
    regrets = [p["regret"] for p in by_problem]
    results[metric] = {
        "invariant_instance_hindsight": display_sign * mean(p["invariant"] for p in by_problem),
        "per_representation_hindsight": display_sign * mean(p["orbit"] for p in by_problem),
        "potential_gap": mean(regrets),
        "nominal_winner_transfer_to_three": display_sign * mean(p["transfer"] for p in by_problem),
        "per_representation_hindsight_on_three": display_sign * mean(p["best_other_three"] for p in by_problem),
        "instances_with_common_winner_across_four": sum(p["has_shared_winner"] for p in by_problem),
        "instances_with_positive_invariant_regret": sum(gap > 1e-12 for gap in regrets),
        "largest_instance_regret": max(regrets),
        "median_instance_regret": statistics.median(regrets),
    }

noise_rows = [row for row in all_rows if row["block"] == "noise"]
noise_problems = {row["instance"] for row in noise_rows}
noise_configs = sorted({row["config"] for row in noise_rows})
for problem, config in itertools.product(noise_problems, noise_configs):
    matched = [row for row in all_rows if (row["instance"], row["config"]) == (problem, config)
               and row["k"] == "0" and row["eps"] == "1e-06" and row["block"] in ("noise", "main")]
    assert len(matched) == 3
assert (len(noise_problems), len(noise_configs)) == (10, 2)
population = {
    "problems": len(problems), "configs": configs, "records": len(table),
    "permutations": forms, "per_problem_configs_and_forms_complete": True,
    "noise_problems": len(noise_problems), "noise_configs": noise_configs,
    "noise_three_repeats_per_cell": True,
    "noise_and_permutation_problems_overlap": len(noise_problems.intersection(problems)),
}
print(json.dumps({"population": population, "exploratory": results}))
