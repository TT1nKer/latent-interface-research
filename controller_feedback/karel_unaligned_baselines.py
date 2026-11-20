"""Simple whole-trajectory Karel policies with no supplied execution landmarks."""
from collections import Counter
from itertools import combinations

from karel_online_control import BEGIN, STOP, fit, next_history, predict, sensor
from karel_trace_control import ACTIONS


METHODS = ("suffix_live", "suffix_initial", "suffix_masked", "modal_plan", "index_vote",
           "dtw_uniform", "dtw_actions_only", "dtw_support_weighted")
GAP = 16


def choose(votes):
    return min(votes, key=lambda a: (-votes[a], a != STOP, a))


def observation_bits(obs):
    if len(obs) != 4 or any(x not in (0, 1) for x in obs):
        raise ValueError("Expected four Boolean Karel predicates")
    return tuple(bool(x) for x in obs)


def canonical_support(support):
    traces = tuple(tuple((observation_bits(obs), action) for obs, action in demo) for demo in support)
    if len(traces) != 5:
        raise ValueError("Expected exactly five support demonstrations")
    for t in traces:
        if not t or t[-1][1] != STOP or any(a not in ACTIONS for _, a in t[:-1]):
            raise ValueError("Each support must have primitive commands followed by one STOP")
    return traces


class SuffixVote:
    def __init__(self, support, mode):
        self.mode = mode
        self.fallback = fit(support, "masked")
        self.model = self.fallback if mode == "masked" else fit(support, mode)
        self.initial = None
        self.history = self.blind_history = (BEGIN,)

    def act(self, obs):
        obs = observation_bits(obs)
        key = sensor(self.mode, obs, obs if self.initial is None else self.initial)
        action = predict(self.model, key, self.history)
        return action if action is not None else predict(self.fallback, None, self.blind_history)

    def advance(self, obs, action):
        if action == STOP:
            return
        obs = observation_bits(obs)
        if self.initial is None:
            self.initial = obs
        key = sensor(self.mode, obs, self.initial)
        self.history = next_history(self.history, key, action)
        self.blind_history = next_history(self.blind_history, None, action)


class ClockVote:
    def __init__(self, support, modal):
        self.plans = tuple(tuple(a for _, a in t[:-1]) for t in support)
        if modal:
            counts = Counter(self.plans)
            self.plans = (min(counts, key=lambda p: (-counts[p], p)),)
        self.position = 0

    def act(self, obs):
        return choose(Counter(p[self.position] if self.position < len(p) else STOP for p in self.plans))

    def advance(self, obs, action):
        if action != STOP:
            self.position += 1


def support_weights(support):
    # [sensor bit][two levels differ?][disagreement count, total matched contexts]
    table = [[[0, 0] for _ in range(2)] for _ in range(4)]
    for left, right in combinations(support, 2):
        previous_agreed = True
        for (x, a), (y, b) in zip(left, right):
            if previous_agreed:
                for i in range(4):
                    cell = table[i][x[i] != y[i]]
                    cell[0] += a != b
                    cell[1] += 1
            previous_agreed = a == b
    weights = []
    for (same_no, same_n), (diff_no, diff_n) in table:
        if not same_n or not diff_n:
            weights.append(0)
            continue
        # Exact add-one-smoothed probability contrast, rounded once, half up.
        denominator = (diff_n+2) * (same_n+2)
        numerator = max(0, (diff_no+1)*(same_n+2) - (same_no+1)*(diff_n+2))
        weights.append((2*GAP*numerator + denominator) // (2*denominator))
    return tuple(weights)


def obs_code(obs):
    return sum(int(value) << i for i, value in enumerate(obs))


class PrefixAlignment:
    def __init__(self, support, weights):
        self.cost = tuple(sum(w for i, w in enumerate(weights) if bits & (1 << i)) for bits in range(16))
        self.donors = [tuple((obs_code(obs), a) for obs, a in t) for t in support]
        self.prefix = [[GAP*j for j in range(len(t))] for t in support]

    def act(self, obs):
        current = obs_code(observation_bits(obs))
        best = None
        votes = Counter()
        for donor, prefix in zip(self.donors, self.prefix):
            distances = [old + self.cost[current ^ bits] for old, (bits, _) in zip(prefix, donor)]
            local = min(distances)
            if best is None or local < best:
                votes.clear()
                best = local
            if local == best:
                labels = {action for d, (_, action) in zip(distances, donor) if d == local}
                # Six possible labels; 60 divides equally by every possible tie size.
                for label in labels:
                    votes[label] += 60//len(labels)
        return choose(votes)

    def advance(self, obs, action):
        if action == STOP:
            return
        current = obs_code(observation_bits(obs))
        for i, (donor, before) in enumerate(zip(self.donors, self.prefix)):
            after = [before[0]+GAP]
            for j, (bits, label) in enumerate(donor[:-1], start=1):
                cost = self.cost[current ^ bits] + GAP*(action != label)
                after.append(min(before[j-1]+cost, before[j]+GAP, after[-1]+GAP))
            self.prefix[i] = after


def build(method, support):
    """Create a fresh policy; act only reads state, advance consumes one executed event."""
    if method not in METHODS:
        raise ValueError(f"Unknown unaligned Karel baseline {method!r}")
    traces = canonical_support(support)
    if method.startswith("suffix_"):
        return SuffixVote(traces, method.removeprefix("suffix_"))
    if method in ("modal_plan", "index_vote"):
        return ClockVote(traces, modal=method == "modal_plan")
    if method == "dtw_support_weighted":
        weights = support_weights(traces)
    else:
        weights = (4,)*4 if method == "dtw_uniform" else (0,)*4
    return PrefixAlignment(traces, weights)
