"""Fit three fixed-size conventional cue tables, score chronological splits and replay the frozen suffix."""

import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RAW = STUDY / "raw"
BASE = STUDY.parent / "raw"
OFFSETS = (-32, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 32)
MODES = ("current", "previous", "transition")
CASES = ("mcf-r1", "mcf-cactus-r1", "lbm-r1", "gap_bfs3", "gap_bfs8")


def source(case):
    for root in (RAW / "runs", BASE / "runs"):
        run = root / (case + "-stride")
        if (run / "complete.json").is_file():
            return run
    raise FileNotFoundError(case)


def token(old, line):
    return 33 if old // 64 != line // 64 else 16 + max(-16, min(16, line - old))


def signatures(z):
    recent = marker = 65535
    mark = -1000
    last = None
    tracks = [None] * 16
    keys = []
    for i, (pc, line) in enumerate(zip(z["pc"].tolist(), z["pa"].tolist())):
        dx = 34 if last is None else token(last, line)
        base = (((pc & 65535) << 9) | (((line & 63) >> 3) << 6) | dx) << 16
        keys.append(
            (base | 65535, base | recent, base | (marker if i - mark <= 64 else 65535))
        )
        recent = ((pc & 255) << 6) | dx
        slot = (pc >> 2) & 15
        entry = tracks[slot]
        step, repeats = 34, 0
        if entry and entry[0] == pc:
            step = token(entry[1], line)
            if entry[3] >= 2 and entry[2] != step:
                marker, mark = recent, i
            if step != 16 and step < 33:
                repeats = min(2, entry[3] + 1) if entry[2] == step else 1
        tracks[slot] = (pc, line, step, repeats)
        last = line
    return keys


def set_of(key):
    return (key ^ (key >> 11) ^ (key >> 23) ^ (key >> 33)) & 31


def segments(z, meta):
    start = meta["skip"] + meta["warmup"]
    watermark = np.maximum.accumulate(z["instr"])
    gates = [start + (meta["roi"] * f) // 10 + 1 for f in (5, 7)]
    return [0, *map(int, np.searchsorted(watermark, gates)), len(watermark)], gates[-1]


def read_case(case):
    with np.load(RAW / "history" / (case + "-c0.npz")) as f:
        z = {k: f[k] for k in f.files if k != "trace"}
    meta = json.loads((source(case) / "run.json").read_text())
    cuts, gate = segments(z, meta)
    return z, meta, cuts, gate, signatures(z)


def lead(z, cut):
    durations = z["fill"][:cut] - z["dispatch"][:cut]
    mask = (
        (z["dispatch"][:cut] >= 0) & (durations > 0) & (z["fill"][:cut] < z["t"][cut])
    )
    if not np.any(mask):
        raise ValueError("missing measurable training service")
    return int(np.median(durations[mask]))


def opportunities(z, lo, hi, service):
    pos = {}
    times = z["t"].tolist()
    lines = z["pa"].tolist()
    eligible = ((z["hit"] == 0) & (z["baseline"] == 0)).tolist()
    proposal = z["pf_gen"].tolist()
    claims = z["pf_kind"].tolist()
    for i in reversed(range(lo, hi)):
        ts, line = times[i], lines[i]
        if times[hi - 1] - ts >= 4 * service:
            matches = []
            for d in OFFSETS:
                target = line + d
                if target // 64 != line // 64:
                    continue
                j = pos.get(target)
                if (
                    j is not None
                    and eligible[j]
                    and service <= times[j] - ts <= 4 * service
                    and (claims[j] != 2 or (proposal[j] >= 0 and ts < proposal[j]))
                ):
                    matches.append((d, j))
            yield i, matches
        pos[line] = i


def fit(keys, labels):
    tables = []
    for m in range(3):
        supports = Counter()
        counts = defaultdict(Counter)
        for i, matches in labels:
            key = keys[i][m]
            supports[key] += 1
            for d, _ in matches:
                counts[key][d] += 1
        groups = defaultdict(list)
        for k, c in counts.items():
            d, n = min(c.items(), key=lambda x: (-x[1], abs(x[0]), x[0]))
            if supports[k] >= 8 and n >= 2 and 10 * n >= supports[k]:
                groups[set_of(k)].append((n, n / supports[k], k, d))
        table = {}
        for candidates in groups.values():
            for _, _, k, d in sorted(candidates, key=lambda x: (-x[0], -x[1], x[2]))[
                :4
            ]:
                table[k] = d
        tables.append(table)
    return tables


def prepare(train, test):
    tz, tm, tc, tg, tk = read_case(train)
    service = lead(tz, tc[1])
    fitted = fit(tk, list(opportunities(tz, 0, tc[1], service)))
    z, meta, cuts, gate, keys = (
        (tz, tm, tc, tg, tk) if test == train else read_case(test)
    )
    runname = test if test == train else train + "_to_" + test
    d = RAW / "cue_tables" / runname
    d.mkdir(parents=True, exist_ok=True)
    (d / "noop.txt").write_text("0 " + str(gate) + "\n")
    rows = []
    for m, table in enumerate(fitted):
        (d / (MODES[m] + ".txt")).write_text(
            f"{m} {gate}\n" + "".join(f"{k} {v}\n" for k, v in sorted(table.items()))
        )
    for split, l, h in [("development", cuts[1], cuts[2]), ("test", cuts[2], cuts[3])]:
        labels = list(opportunities(z, l, h, service))
        for m, table in enumerate(fitted):
            hits = correct = 0
            distinct = set()
            for i, matches in labels:
                d = table.get(keys[i][m])
                if d is None or (z["pa"][i] + d) // 64 != z["pa"][i] // 64:
                    continue
                hits += 1
                for offset, j in matches:
                    if offset == d:
                        correct += 1
                        distinct.add(j)
            rows.append(
                (
                    runname,
                    split,
                    MODES[m],
                    service,
                    len(table),
                    len(labels),
                    hits,
                    correct,
                    len(distinct),
                )
            )
    return rows


def replay(train, test, mode):
    name = test if train == test else train + "_to_" + test
    dest = RAW / "cue_runs" / name / mode
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((source(test) / "run.json").read_text())
    traces = []
    for old in meta["command"][meta["command"].index("-traces") + 1 :]:
        trace = Path(old)
        choices = (trace, RAW / "traces" / trace.name, BASE / "traces" / trace.name)
        traces.append(str(next(p for p in choices if p.is_file())))
    binary = RAW / "cue_build/native/pythia/bin/champsim"
    cmd = [
        str(binary),
        f"--warmup_instructions={meta['warmup']}",
        f"--simulation_instructions={meta['roi']}",
        "-traces",
        *traces,
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HYBRID_", "CUE_", "LLC_PF_"))
    }
    env.update(
        meta["environment"],
        HYBRID_OUT=str(dest),
        CUE_TABLE=str(RAW / "cue_tables" / name / (mode + ".txt")),
    )
    with (dest / "stdout.txt").open("w") as log:
        subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=dest
        )
    with (dest / "cores.csv").open() as f:
        cores = list(csv.DictReader(f))
    if len(cores) != len(traces) or any(
        int(c["instructions"]) != meta["roi"] for c in cores
    ):
        raise ValueError("fixed work mismatch")
    (dest / "complete.json").write_text(
        json.dumps(dict(train=train, test=test, mode=mode, roi=meta["roi"])) + "\n"
    )


def pairs():
    return [(c, c) for c in CASES] + [("gap_bfs3", "gap_bfs8")]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--fit-only", action="store_true")
    args = p.parse_args()
    with (HERE / "held_out_scores.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            (
                "case",
                "split",
                "memory",
                "training_service_cycles",
                "table_entries",
                "scored_anchors",
                "legal_table_predictions",
                "correct_anchors",
                "unique_target_loads",
            )
        )
        for a, b in pairs():
            w.writerows(prepare(a, b))
    if not args.fit_only:
        with ThreadPoolExecutor(args.jobs) as ex:
            tasks = [
                ex.submit(replay, a, b, m) for a, b in pairs() for m in ("noop", *MODES)
            ]
            for task in tasks:
                task.result()
