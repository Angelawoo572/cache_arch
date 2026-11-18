"""Check whether earlier LLC-visible deltas change held-out timely-miss associations."""
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
import sys

# Shared parser for completed LLC outcomes, arrival identities and observed service.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "intervention"))
from earlier_issue import read_llc, identity

OFFSETS = (-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32)
WIDTH = len(OFFSETS)
MODES = ("current", "ordered", "unordered")


def token(a, b):
    if a is None:
        return "start"
    if a // 64 != b // 64:
        return "page"
    return max(-32, min(32, b - a))


def context(arrivals):
    last = {}
    histories = defaultdict(lambda: deque(["start"] * 4, maxlen=4))
    output = []
    for a in arrivals:
        k = a.cpu
        current_delta = token(last.get(k), a.line)
        preceding = tuple(histories[k])
        base = (a.pc, a.line % 64, current_delta)
        output.append((base, preceding))
        histories[k].append(current_delta)
        last[k] = a.line
    return output


def dataset(folder):
    rows, lookups, services = read_llc(folder)
    if not services:
        raise ValueError("missing demand service observations")
    service = int(median(services))
    rows.sort(key=lambda a: a.cycle)
    contexts = context(rows)
    deadline = {a.cpu: a.cycle for a in rows}
    next_touch = {}
    observations = []
    for a, (base, prefix) in zip(reversed(rows), reversed(contexts)):
        targets = {}
        if deadline[a.cpu] - a.cycle >= 4 * service:
            for i, delta in enumerate(OFFSETS):
                line = a.line + delta
                if line // 64 != a.line // 64:
                    continue
                following = next_touch.get((a.cpu, line))
                if following is None:
                    continue
                target, missed = following
                if missed and service <= target.cycle - a.cycle <= 4 * service:
                    targets[i] = identity(target)
            observations.append((a, base, prefix, targets))
        next_touch[a.cpu, a.line] = (a, lookups.get(identity(a)) == 0)
    observations.reverse()
    return observations, service


def keys(base, prefix):
    return (base, (base, prefix), (base, tuple(sorted(prefix, key=str))))


def count_contexts(observations):
    tables = [defaultdict(lambda: [0] * (WIDTH + 1)) for _ in MODES]
    global_counts = [0] * (WIDTH + 1)
    for _, base, prefix, targets in observations:
        groups = [global_counts]
        for table, key in zip(tables, keys(base, prefix)):
            groups.append(table[key])
        for entry in groups:
            entry[-1] += 1
            for k in targets:
                entry[k] += 1
    return tables, global_counts


def evaluate(train, test):
    observations, training_service = dataset(train)
    training_pcs = {a.pc for a, _, _, _ in observations}
    tables, global_counts = count_contexts(observations)
    observations, test_service = dataset(test)
    if not observations or not global_counts[-1]:
        raise ValueError("no eligible anchor for the history comparison")
    shared_pc_anchors = sum(a.pc in training_pcs for a, _, _, _ in observations)
    reachable = {key for _, _, _, targets in observations for key in targets.values()}
    ranks = [[] for _ in MODES]
    global_rate = [value / global_counts[-1] for value in global_counts[:-1]]
    supported = [0] * len(MODES)
    for j, (a, base, prefix, targets) in enumerate(observations):
        offset = a.line % 64
        legal = [k for k, delta in enumerate(OFFSETS) if 0 <= offset + delta < 64]
        rates = None
        base_rates = None
        for m, (table, key) in enumerate(zip(tables, keys(base, prefix))):
            prior = global_rate if m == 0 else base_rates
            count = table.get(key)
            if count is not None and count[-1] >= 8:
                supported[m] += 1
                rates = [(count[k] + 16 * prior[k]) / (count[-1] + 16) for k in range(WIDTH)]
            else:
                rates = prior
            if m == 0:
                base_rates = rates
            best = max(legal, key=lambda k: (rates[k], -abs(OFFSETS[k]), OFFSETS[k]))
            ranks[m].append((rates[best], j, targets.get(best)))
    cap = max(1, len(observations) // 50)
    for m, name in enumerate(MODES):
        chosen = sorted(ranks[m], key=lambda item: (-item[0], item[1]))[:cap]
        targets = [entry[2] for entry in chosen if entry[2] is not None]
        yield (name, training_service, test_service, len(observations), supported[m], cap,
               len(targets), len(set(targets)), shared_pc_anchors, len(reachable))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--programs", nargs="+", default=["mcf", "cactus", "xalan", "lbm"])
    args = parser.parse_args()
    with args.out.open("w", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(("program","context","train_median_service","test_median_service","eligible_test_anchors",
                      "test_anchors_with_training_support","offline_issue_budget","reached_at_decisions","distinct_misses_reached","test_anchors_with_seen_pc","oracle_distinct_misses_in_vocabulary"))
        for program in args.programs:
            for row in evaluate(args.runs / (program + "-r1-nopf"), args.runs / (program + "-r2-nopf")):
                out.writerow((program,) + row)
            f.flush()


if __name__ == "__main__":
    main()
