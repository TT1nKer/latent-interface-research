"""Open-loop demonstration replay on released Karel worlds. Standard library only."""
import argparse
import collections
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from statistics import mean
import zipfile

DIRECTIONS = ("north", "east", "south", "west")
DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
ACTIONS = {"move", "turnLeft", "turnRight", "putMarker", "pickMarker"}


class World:
    def __init__(self, rows, cols, blocked, markers, hero, direction):
        self.rows = rows
        self.cols = cols
        self.blocked = frozenset(blocked)
        self.markers = {cell: n for cell, n in markers.items() if n}
        self.hero = hero
        self.direction = direction

    @classmethod
    def parse(cls, data):
        if data["crashed"]:
            raise ValueError("crashed reference world")
        row, col, heading = data["hero"].split(":")
        blocked = [tuple(map(int, field.split(":"))) for field in data["blocked"].split()]
        markers = {}
        for field in data["markers"].split():
            r, c, n = map(int, field.split(":"))
            markers[r, c] = n
        return cls(data["rows"], data["cols"], blocked, markers,
                   (int(row), int(col)), DIRECTIONS.index(heading))

    def state(self):
        return (self.rows, self.cols, self.blocked, self.hero, self.direction,
                frozenset(self.markers.items()))

    def clone(self):
        result = copy.copy(self)
        result.markers = self.markers.copy()
        return result

    def next_cell(self):
        dr, dc = DELTAS[self.direction]
        return self.hero[0] + dr, self.hero[1] + dc

    def clear(self):
        r, c = self.next_cell()
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.blocked

    def step(self, action):
        if action == "move":
            if not self.clear():
                return False
            self.hero = self.next_cell()
        elif action == "turnLeft":
            self.direction = (self.direction - 1) % 4
        elif action == "turnRight":
            self.direction = (self.direction + 1) % 4
        elif action == "putMarker":
            self.markers[self.hero] = self.markers.get(self.hero, 0) + 1
            if self.markers[self.hero] > 101:
                return False
        elif action == "pickMarker":
            if self.hero not in self.markers:
                return False
            self.markers[self.hero] -= 1
            if not self.markers[self.hero]:
                del self.markers[self.hero]
        else:
            raise ValueError(f"Unrecognized primitive {action!r}")
        return True


def replay(world, actions):
    w = world.clone()
    for action in actions:
        if not w.step(action):
            return None
    return w.state()


def staircase_worlds(path):
    # The exercise uses 1-based top-left cell coordinates; internal north increases row.
    lines = path.read_text().splitlines()
    worlds = []
    for start, header in enumerate(lines):
        if not header.startswith(("pregrid_", "postgrid_")):
            continue
        cells = []
        i = start + 1
        while not lines[i].startswith("agentloc_"):
            cells.append(lines[i].split()[1:])
            i += 1
        rows, cols = len(cells), len(cells[0])
        blocked, markers = set(), {}
        for r, line in enumerate(reversed(cells)):
            if len(line) != cols:
                raise ValueError("Ragged staircase")
            for c, value in enumerate(line):
                if value == "#":
                    blocked.add((r, c))
                elif value.isdigit():
                    markers[r, c] = int(value)
                elif value != ".":
                    raise ValueError(f"Unknown staircase tile {value!r}")
        c, r = map(int, re.search(r"col=(\d+),row=(\d+)", lines[i]).groups())
        direction = DIRECTIONS.index(lines[i + 1].split()[1])
        worlds.append(World(rows, cols, blocked, markers, (rows-r, c-1), direction))
    if len(worlds) != 6:
        raise ValueError("Published Stairway exercise must have three input/output pairs")
    return zip(worlds[::2], worlds[1::2])


def staircase_trace(program, source):
    world = source.clone()
    actions = []
    names = {"pick_marker":"pickMarker", "turn_left":"turnLeft", "turn_right":"turnRight", "move":"move"}

    def visit(node):
        kind = node["type"]
        if kind in names:
            action = names[kind]
            if len(actions) >= 1000 or not world.step(action):
                raise ValueError("Invalid reference execution")
            actions.append(action)
        elif kind in ("run", "do"):
            for child in node["children"]:
                visit(child)
        elif kind == "if(bool_marker)":
            if world.markers.get(world.hero, 0):
                for child in node["children"]:
                    visit(child)
        elif kind == "while(bool_no_path_ahead)":
            for _ in range(1000):
                if world.clear():
                    break
                for child in node["children"]:
                    visit(child)
            else:
                raise ValueError("Nonterminating authored reference")
        else:
            raise ValueError(f"Unknown published AST node {kind!r}")
    visit(program)
    return tuple(actions)


def summarize_program(guid, worlds, targets, traces, conditional):
    n = len(worlds)
    final = []
    exact = []
    prefix = {p:collections.Counter() for p in (1, 3)}
    same_input = 0
    for j in range(n):
        outcomes = [replay(worlds[j], t) == targets[j] for t in traces]
        if not outcomes[j]:
            raise ValueError(f"Recorded trace did not yield recorded output: program={guid} world={j}")
        final.append(outcomes)
        exact.append([t == traces[j] for t in traces])
        for i, source in enumerate(traces):
            if i == j:
                continue
            same_input += worlds[i].state() == worlds[j].state()
            for length, counts in prefix.items():
                if min(len(source),len(traces[j])) > length and source[:length] == traces[j][:length]:
                    counts["pairs"] += 1
                    counts["same_next"] += source[length] == traces[j][length]
                    counts["terminal_hits"] += outcomes[i]
    off_diagonal = [(j, i) for j in range(n) for i in range(n) if j != i]
    schedule_options = {}
    for k in (1, 3, 5):
        if n - 1 < k:
            continue
        selections = []
        for j in range(n):
            for subset in itertools.combinations([i for i in range(n) if i != j], k):
                freq = collections.Counter(traces[i] for i in subset)
                modes = [trace for trace, count in freq.items() if count == max(freq.values())]
                selections.append((mean(final[j][traces.index(t)] for t in modes),
                                   mean(t == traces[j] for t in modes),
                                   any(final[j][i] for i in subset), any(exact[j][i] for i in subset)))
        names = ("mode_terminal","mode_exact","hindsight_terminal","hindsight_exact")
        schedule_options[k] = dict(zip(names, map(mean, zip(*selections))))
    end_rate = mean(final[j][i] for j, i in off_diagonal)
    trace_rate = mean(exact[j][i] for j, i in off_diagonal)
    return {"guid":guid,"worlds":n,"pairs":len(off_diagonal),"equal_input_pairs":same_input,
            "distinct_traces":len(set(traces)), "conditional_syntax":conditional,
            "terminal_rate":end_rate,"trace_rate":trace_rate,
            "different_trace_terminal_rate":end_rate-trace_rate, "k_support":schedule_options,
            "prefix":{p:dict(c) for p,c in prefix.items()}}


def summarize(rows):
    metrics = ("terminal_rate","trace_rate","different_trace_terminal_rate")
    def rates(part):
        return {"programs":len(part), **{m:mean(r[m] for r in part) if part else None for m in metrics}}
    groups = {"all":rates(rows),
              "conditional_syntax":rates([r for r in rows if r["conditional_syntax"]]),
              "no_conditional_syntax":rates([r for r in rows if not r["conditional_syntax"]]),
              "varied_observed_traces":rates([r for r in rows if r["distinct_traces"]>1]),
              "constant_observed_traces":rates([r for r in rows if r["distinct_traces"]==1])}
    ks = {}
    for k in (1,3,5):
        eligible = [r["k_support"][k] for r in rows if k in r["k_support"]]
        if eligible:
            ks[k] = {"programs":len(eligible), **{name:mean(item[name] for item in eligible)
                                                 for name in eligible[0]}}
    pref = {}
    for p in (1,3):
        summed = collections.Counter()
        for row in rows:
            summed.update(row["prefix"][p])
        pref[p] = dict(summed)
    return {"programs":len(rows),"worlds":sum(r["worlds"] for r in rows),
            "ordered_source_target_pairs":sum(r["pairs"] for r in rows),
            "equal_input_pairs":sum(r["equal_input_pairs"] for r in rows),"groups":groups,
            "k_support":ks,"prefix_counts":pref}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the empirical comparison on school Slurm")
    sources = args.source
    filenames = {"val.zip":"967dc1c557595e7c4a5ff08a636ec3fe2b5a5320fd6548be50295cf36c5ab4cd",
                 "short_val_code.txt":"d6908129926120a7ec0c58e2c60dbb7938dc1e19554b7e76a999f4034542f2f4",
                 "karel-stairway_code.json":"ff542ff88162918a4c02e0a27c6a1490696c4f5cacc6d3006663c51f569c4ceb",
                 "karel-stairway_task.txt":"3be328fc36ef854fe6ccadada943cf9ad084cdae7a73b3c885f9493689e86c80"}
    if any(sha(sources / name) != value for name, value in filenames.items()):
        raise ValueError("Input digest mismatch")
    author_codes = (sources/"short_val_code.txt").read_text().splitlines()
    rows, codes = [], []
    excluded = 0
    with zipfile.ZipFile(sources/"val.zip") as file, file.open("val.json") as raw:
        for line in itertools.islice(raw, 2500):
            record = json.loads(line)
            code = record["program_tokens"]
            codes.append(" ".join(code))
            cases = record["examples"][:6]
            if len(cases)<2:
                excluded += 1
                continue
            rows.append(summarize_program(record["guid"], [World.parse(c["inpgrid_json"]) for c in cases],
                                         [World.parse(c["outgrid_json"]).state() for c in cases],
                                         [tuple(c["actions"]) for c in cases],
                                         any(t in {"IF","IFELSE","WHILE"} for t in code)))
    published = collections.Counter(author_codes)
    rawcodes = collections.Counter(codes)
    matched = sum((published & rawcodes).values())
    if len(author_codes)!=1000 or matched!=1000:
        raise ValueError(f"Independent published program corroboration mismatch: {matched} / {len(author_codes)}")
    task = list(staircase_worlds(sources/"karel-stairway_task.txt"))
    program = json.loads((sources/"karel-stairway_code.json").read_text())
    challenge = summarize_program("karel-stairway", [w for w,_ in task],[s.state() for _,s in task],
                                  [staircase_trace(program,w) for w,_ in task],True)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    with (output/"program_rows.jsonl").open("x") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",",":"))+"\n")
    summary = {"job":os.environ["SLURM_JOB_ID"],"inputs":filenames,
               "protocol_sha256":sha(args.protocol),"worker_sha256":sha(Path(__file__)),
               "program_rows_sha256":sha(output/"program_rows.jsonl"),
               "validation_source":"Independent, non-author HF mirror of val.json; program-level corroboration only",
               "author_programs_in_mirror":matched,"author_codes_exact_prefix":codes[:len(author_codes)]==author_codes,
               "fewer_than_two_worlds":excluded, "validation":summarize(rows),
               "authored_exercise":summarize([challenge]), "authored_exercise_row":challenge}
    with (output/"summary.json").open("x") as f:
        json.dump(summary,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"records":len(rows),"native_worlds":summary["validation"]["worlds"],
                      "published_codes":matched,"end":summary["validation"]["groups"]["all"]}))


if __name__ == "__main__":
    main()
