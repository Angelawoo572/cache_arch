"""Sample streams released only after fixed-work retirement; retain native first-use attribution."""

import gzip
import json
import os
import numpy as np
from pattern import HERE, RAW

FIELDS = "instr op pc pa va t tag hit dispatch fill head gen pf_fill kind".split()


def lines(folder, name):
    with gzip.open(folder / (name + ".csv.gz"), "rt") as f:
        h = next(f).strip().split(",")
        yield {k: i for i, k in enumerate(h)}
        for row in f:
            yield row.rstrip("\n").split(",")


def parse(folder):
    rec = lines(folder, "loads")
    h = next(rec)
    mapped = {}
    ordered = []
    for row in rec:
        if row[h["level"]] != "LLC" or row[h["cpu"]] != "0":
            continue
        key = (int(row[h["instr_id"]]), int(row[h["operand"]]))
        stage = row[h["stage"]]
        if stage == "ingress":
            if key in mapped:
                raise ValueError(f"duplicate ingress {key}")
            v = [
                *key,
                int(row[h["pc"]]),
                int(row[h["physical_line"]]),
                int(row[h["virtual_line"]]),
                int(row[h["cycle"]]),
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                -1,
                0,
            ]
            mapped[key] = v
            ordered.append(v)
        elif key in mapped:
            if stage == "lookup":
                mapped[key][6:8] = [int(row[h["cycle"]]), int(row[h["hit"]])]
            if stage == "dispatch":
                mapped[key][8] = int(row[h["cycle"]])
            if stage == "fill":
                mapped[key][9] = int(row[h["cycle"]])
    rec = lines(folder, "head")
    h = next(rec)
    for row in rec:
        key = (int(row[h["instr_id"]]), int(row[h["operand"]]))
        if row[h["cpu"]] == "0" and key in mapped:
            mapped[key][10] += int(row[h["cycles"]])
    rec = lines(folder, "candidates")
    h = next(rec)
    pf = {}
    uses = {}
    for row in rec:
        stage = row[h["stage"]]
        pid = row[h["id"]]
        if stage == "proposed":
            pf[pid] = [int(row[h["cycle"]]), -1]
        elif stage == "fill" and pid in pf:
            pf[pid][1] = int(row[h["cycle"]])
        elif stage in ("timely", "late") and row[h["demand_cpu"]] == "0":
            key = (int(row[h["demand_instr_id"]]), int(row[h["demand_operand"]]))
            if key in mapped:
                uses[pid] = (key, 1 if stage == "timely" else 2, int(row[h["target"]]))
    for pid, (key, kind, target) in uses.items():
        if mapped[key][3] != target:
            raise ValueError(f"wrong physical first use {key}")
        mapped[key][11:] = [*pf.get(pid, (-1, -1)), kind]
    return {
        k: np.fromiter(
            (r[i] for r in ordered),
            dtype=("uint64" if k in ("pc", "va") else "int64"),
            count=len(ordered),
        )
        for i, k in enumerate(FIELDS)
    }


def location(case, policy, scope):
    if scope == "full":
        return (
            (
                RAW / "full"
                if policy in ("shape", "sparse3")
                else HERE.parent / "raw/native"
            )
            / "runs"
            / (case + "__" + policy)
        )
    return RAW / "runs" / case / policy


def events(case, policy, scope):
    folder = location(case, policy, scope)
    if not (folder / "complete.json").exists():
        raise FileNotFoundError(folder)
    dst = RAW / "parsed" / scope / case / (policy + ".npz")
    if not dst.exists():
        dst.parent.mkdir(exist_ok=True, parents=True)
        data = parse(folder)
        assert np.all(np.diff(data["t"]) >= 0)
        temp = dst.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temp, **data)
        temp.replace(dst)
    with np.load(dst) as f:
        return {k: f[k] for k in f}


def keys(z):
    return [(int(i), int(op)) for i, op in zip(z["instr"], z["op"])]


def label(z, b):
    idx = {key: i for i, key in enumerate(keys(b))}
    labels = []
    base = b["hit"]
    for i, key in enumerate(keys(z)):
        j = idx.get(key)
        if j is not None and any(b[k][j] != z[k][i] for k in ("pc", "pa", "va")):
            raise ValueError(f"misaligned dynamic operand {key}")
        labels.append(-1 if j is None else base[j])
    return dict(z, baseline=np.asarray(labels, dtype="int8"))


def parts(z, scope, case):
    if scope != "full":
        return [(1, z)]
    cfg = json.loads((location(case, "nopf", "full") / "config.json").read_text())
    start = cfg.get("source_ordinal_base", 0) + cfg["skip"] + cfg["warmup"]
    delta = z["instr"] - 1 - start
    ids = 4 * delta // cfg["roi"]
    offset = delta - (ids * cfg["roi"] + 3) // 4
    if np.any((delta < 0) | (ids > 3) | (offset >= cfg["sample_width"])):
        raise ValueError(f"Non-sampled dynamic operand in {case}")
    return [(q + 1, {k: v[ids == q] for k, v in z.items()}) for q in range(4)]
