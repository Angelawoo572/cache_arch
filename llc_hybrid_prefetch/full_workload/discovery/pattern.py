"""Prefix-only feature and physical-candidate controls shared by training and evaluation."""

from collections import Counter, defaultdict, deque
import csv
import os
import json
from pathlib import Path
from statistics import median
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = HERE.parent / "raw/discovery"
ORIGINAL = PROJECT / "workload_expansion/raw/screen/history"
CASES = ("spec_mcf", "graph_bfs3")
MODES = ("local", "ordered", "shape", "bag", "temporal", "anonymous_local")
MASK = 2**64 - 1


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, data, fields):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    temp.replace(path)


def mix(words):
    key = 1469598103934665603
    for word in words:
        key = ((key ^ word) * 1099511628211) & MASK
    return key


def snapshots(z):
    slots = {}
    previous = (0, 0, 0)
    history = deque()
    counts = Counter()
    seen = {}
    records = []
    for i, (pc0, pa0) in enumerate(zip(z["pc"], z["pa"])):
        pc, line = int(pc0), int(pa0)
        at = ((pc >> 2) ^ (pc >> 9)) & 63
        old, prev, prior_delta = slots.get(at, (-1, 0, 65))
        d = max(-64, min(64, line - prev)) if old == pc else 65
        if old != pc:
            prior_delta = 65
        slots[at] = (pc, line, d)
        page = line // 64
        if len(history) == 64:
            head = history.popleft()
            counts[head] -= 1
            if not counts[head]:
                del counts[head]
        counts[page] += 1
        history.append(page)
        seen.setdefault(line, i)
        shape = tuple((*previous, pc).index(x) for x in (*previous, pc))
        local = (pc, line % 64, d + 64, prior_delta + 64)
        keys = (
            local,
            local + previous,
            local[1:] + shape,
            local + tuple(sorted(previous)),
            (line,),
            local[1:],
        )
        records.append((keys, frozenset(counts)))
        previous = (*previous[1:], pc)
    return records, seen


def legal(line, target, pages):
    return target > 0 and 0 < abs(target - line) <= 512 and target // 64 in pages


def eligible(z, i, stop, service, pages):
    """Retrospective first-demand label per distinct line; future visibility ends at split edge."""
    line = int(z["pa"][i])
    at = int(z["t"][i])
    seen = set()
    for j in range(i + 1, min(stop, i + 17)):
        target = int(z["pa"][j])
        if target in seen:
            continue
        seen.add(target)
        gap = int(z["t"][j]) - at
        if gap > 4096:
            break
        if int(z["baseline"][j]) == 0 and gap >= service and legal(line, target, pages):
            yield target - line, j


def reference(z):
    cutoff = int(0.6 * len(z["t"]))
    vals = (z["fill"] - z["dispatch"])[
        (z["dispatch"] >= 0)
        & (z["fill"] >= z["dispatch"])
        & (z["fill"] <= z["t"][cutoff - 1])
    ]
    if not len(vals):
        raise ValueError("no completed donor service")
    return int(np.median(vals))


def hist(case, policy):
    with np.load(ORIGINAL / case / (policy + ".npz")) as f:
        return {k: f[k] for k in f if k != "trace"}


def fit():
    support = [Counter() for _ in MODES]
    positives = [defaultdict(Counter) for _ in MODES]
    services = {}
    for case in CASES:
        z = hist(case, "stride")
        services[case] = reference(hist(case, "nopf"))
        records, _ = snapshots(z)
        end = int(0.6 * len(records))
        for i in range(end - 16):
            keys, pages = records[i]
            labels = list(eligible(z, i, end, services[case], pages))
            for k, (key, cnt, pos) in enumerate(zip(keys, support, positives)):
                cnt[key] += 1
                for d, j in labels:
                    action = int(z["pa"][j]) if MODES[k] == "temporal" else d
                    pos[key][action] += 1
    tables = {}
    all_rows = []
    for name, counts, positive in zip(MODES, support, positives):
        entries = {}
        for key, n in counts.most_common():
            if n < 4 or not positive[key]:
                continue
            action, count = min(
                positive[key].items(), key=lambda v: (-v[1], abs(v[0]), v[0])
            )
            if count >= 2 and 5 * count >= n:
                entries[key] = (action, count, n)
        selected = dict(sorted(entries.items(), key=lambda x: (-x[1][2], x[0]))[:256])
        tables[name] = selected
        encoded = {mix(k): v[0] for k, v in selected.items()}
        assert len(encoded) == len(selected)
        if name in ("local", "ordered", "shape"):
            path = HERE / (name + ".rules")
            content = "".join(
                f"{key} {action}\n" for key, action in sorted(encoded.items())
            )
            if not path.exists() or path.read_text() != content:
                temp = path.with_suffix(".tmp")
                temp.write_text(content)
                temp.replace(path)
        for k, (a, y, n) in selected.items():
            all_rows.append(
                dict(
                    mode=name,
                    key=" ".join(map(str, k)),
                    action=a,
                    positive=y,
                    support=n,
                )
            )
    save(
        HERE / "frozen_rows.csv",
        all_rows,
        ["mode", "key", "action", "positive", "support"],
    )
    (HERE / "service.json").write_text(
        json.dumps(
            dict(donors=services, external=int(median(services.values()))), indent=2
        )
        + "\n"
    )
    return tables


def load():
    tables = {m: {} for m in MODES}
    for r in table(HERE / "frozen_rows.csv"):
        tables[r["mode"]][tuple(map(int, r["key"].split()))] = tuple(
            map(int, (r["action"], r["positive"], r["support"]))
        )
    return tables


if __name__ == "__main__":
    print({m: len(t) for m, t in fit().items()})
