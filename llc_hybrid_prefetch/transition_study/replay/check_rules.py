"""Independently recalculate every recorded native and diagnostic choice, with all warmup history."""

import argparse
import csv
import gzip
import os
import platform
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

from run import CASES, HERE, POLICIES, RAW, RUNS, csvs, original, stats


def check(job):
    case, mode = job
    folder = RUNS / (case + "__" + mode)
    assert (folder / "complete.json").exists()
    previous = {}
    n = measured = 0
    with gzip.open(folder / "ingress_history.csv.gz", "rb") as f:
        next(f)
        for row in f:
            core, pc, line, t, match, native, mshr, rq, out, src, roi, iid, operand = (
                row.strip().split(b",")
            )
            core, pc, line, t, match, native, mshr, rq, out, roi = map(
                int, (core, pc, line, t, match, native, mshr, rq, out, roi)
            )
            slot = (core, ((pc >> 2) ^ (pc >> 9)) & 63)
            old = previous.get(slot)
            if not old or old[0] != pc:
                old = (pc, None, 0, 0, 0, None, 0)
            _, last, when, delta, count, learned, votes = old
            step = 0 if last is None else line - last
            if step and step == delta:
                count = min(255, count + 1)
            else:
                if step and count >= 3:
                    triple = (delta, count, step)
                    votes = min(3, votes + 1) if triple == learned else 1
                    learned = triple
                delta = step
                count = int(bool(step))
            previous[slot] = (pc, line, t, delta, count, learned, votes)
            d = native if match else 0
            reason = b"stride" if match else b"silent"

            def valid(x):
                return x != 0 and line + x > 0 and (line + x) // 64 == line // 64

            if match and mode == "look2":
                d, reason = 2 * native, b"look2"
            if (
                match
                and mode in ("paced", "turn")
                and count >= 4
                and delta == native
                and t - when <= 128
                and max(mshr, rq) < 16
                and valid(2 * native)
            ):
                d, reason = 2 * native, b"paced"
            if mode == "turn" and votes >= 2:
                old_step, length, jump = learned
                rel = (length - count) * old_step + jump
                if delta == old_step and count <= length <= count + 2 and valid(rel):
                    d, reason = rel, b"turn"
            assert (d, reason) == (out, src), (case, mode, n, (d, reason), (out, src))
            n += 1
            measured += roi
    old = original(case, "stride")
    equal = csvs(folder / "cores.csv") == csvs(old / "cores.csv") and stats(folder) == stats(old)
    if mode == "stride":
        assert equal
    return dict(
        case=case,
        policy=mode,
        checked_all_ingresses=n,
        checked_measured_ingresses=measured,
        choices_equal_independent_reference=1,
        all_core_and_system_totals_equal_native=int(equal),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(check, [(c, m) for c in CASES for m in POLICIES]))
    cc = os.environ.get("CXX") or (
        "g++-15" if platform.system() == "Darwin" and shutil.which("g++-15") else "g++"
    )
    fixture = RAW / "rule-prefix-check"
    subprocess.run(
        [cc, "-std=c++14", "-O2", str(HERE / "test_prefix.cc"), "-o", str(fixture)], check=True
    )
    subprocess.run([str(fixture)], check=True)
    with (HERE / "validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(
        "Prefix perturbation, rewrites and",
        sum(r["checked_all_ingresses"] for r in rows),
        "actual event decisions reproduced.",
    )
