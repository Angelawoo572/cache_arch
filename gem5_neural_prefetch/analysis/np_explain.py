#!/usr/bin/env python3
"""Describe one window in words a reader can check: which relations carry the
residual stall, how each can be reached early, and what the "none" class is.

    python analysis/np_explain.py WORKLOAD OFFSET [--arm fast]

Prints the largest relations with the instructions at their two PCs, the
share of each that an early start reaches, and the PCs of residual targets
that no construction names.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_join import Arm, fast_outcome  # noqa: E402
from np_measure import CHEAPEST_NAMES, LONG_LATENCY, Window, chain_depth, cheapest  # noqa: E402
from np_report import RUNS, edges_for  # noqa: E402
from np_walker import walker_bound  # noqa: E402


def form(code):
    if code >= 0:
        return f"(value << {code})"
    return f"((value >> {-code}) << {-code - 3})"


def code_at(binary, pc, before=20, after=10):
    """The instructions around one PC, from objdump."""
    done = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", f"--start-address={pc - before:#x}",
         f"--stop-address={pc + after:#x}", binary],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in done.stdout.splitlines() if ":\t" in l]
    return lines[-4:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload")
    parser.add_argument("offset", type=int)
    parser.add_argument("--arm", default="fast")
    args = parser.parse_args()
    nopf_dir = RUNS / f"{args.workload}__o{args.offset}__nopf"
    window = Window(nopf_dir, edges_for(nopf_dir))
    fast = Arm(RUNS / f"{args.workload}__o{args.offset}__{args.arm}")
    binary = json.loads((nopf_dir / "m5out" / "run_info.json").read_text())["binary"]
    load, relations = window.load, window.relations

    residual = window.nopf.miss[2] & np.isin(fast_outcome(fast, 2), (1, 2))
    stall = fast.stall
    total = float(stall[residual].sum())
    depth = chain_depth(relations, fast.latency >= LONG_LATENCY)
    label = cheapest(window.classes, depth)
    print(f"{args.workload} +{args.offset}, arm {args.arm}: residual targets {int(residual.sum())}, stall {total:.0f} cycles")

    detail = {}
    walker_bound(window, fast, detail)
    target, weight = detail["target"], detail["stall"]
    rel = relations["relation"][target]
    table = relations["relations"]
    print("\nrelations by value-derived residual stall")
    order = sorted(set(rel.tolist()), key=lambda r: -weight[rel == r].sum())[:5]
    for r in order:
        chosen = rel == r
        w = float(weight[chosen].sum())
        ppc, cpc = int(table["producer_pc"][r]), int(table["consumer_pc"][r])
        reach = {k: float(weight[chosen & detail[k]].sum()) / w for k in ("stepped", "same_line", "fill_tap", "any")}
        producer = relations["producer"][target[chosen]]
        print(
            f"  {100 * w / total:5.1f}% of residual stall: address = {form(int(table['shift'][r]))} + {int(table['base'][r]):#x}, "
            f"{int(table['depth'][r])} micro-ops apart; reached: stepped {100 * reach['stepped']:.0f}%, "
            f"same line {100 * reach['same_line']:.0f}%, fill tap {100 * reach['fill_tap']:.0f}%, any {100 * reach['any']:.0f}%; "
            f"producer missed L1D {100 * fast.miss[1][producer].mean():.0f}%, producer is a same-PC delta {100 * window.classes['delta'][producer].mean():.0f}%"
        )
        print(f"      value loaded at {ppc:#x}: " + " | ".join(code_at(binary, ppc, 0, 12)[:2]))
        print(f"      used at       {cpc:#x}: " + " | ".join(code_at(binary, cpc, 0, 12)[:2]))

    for name in ("none", "same_pc_delta"):
        code = CHEAPEST_NAMES.index(name)
        chosen = residual & (label == code)
        share = float(stall[chosen].sum()) / total if total else 0.0
        print(f"\nclass {name}: {100 * share:.1f}% of residual stall")
        pcs, inverse = np.unique(load["pc"][chosen], return_inverse=True)
        by_pc = np.bincount(inverse, weights=stall[chosen])
        for i in np.argsort(-by_pc)[:3]:
            pc = int(pcs[i])
            print(f"  {100 * by_pc[i] / total:5.1f}% at {pc:#x}: " + " | ".join(code_at(binary, pc)))


if __name__ == "__main__":
    main()
