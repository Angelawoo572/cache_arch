"""Build a bounded, hindsight-only earlier-issue experiment from a pair of runs."""
import argparse
import bisect
import csv
import gzip
from collections import defaultdict, namedtuple
import json
from pathlib import Path
from statistics import median

Access = namedtuple("Access", "cpu instr operand pc line cycle")
Identity = namedtuple("Identity", "cpu instr operand")


def identity(access):
    return Identity(access.cpu, access.instr, access.operand)


def event_file(folder, name):
    path = folder / (name + ".csv")
    if path.is_file():
        return path.open(newline="")
    return gzip.open(str(path) + ".gz", "rt", newline="")


def read_llc(folder):
    arrivals, outcomes, dispatch = [], {}, {}
    latencies = []
    with event_file(folder, "loads") as stream:
        rows = csv.reader(stream)
        h = {name: i for i, name in enumerate(next(rows))}
        for row in rows:
            if row[h["level"]] != "LLC":
                continue
            stage = row[h["stage"]]
            if stage not in ("ingress", "lookup", "dispatch", "fill"):
                continue
            key = Identity(*(int(row[h[k]]) for k in ("cpu", "instr_id", "operand")))
            tick = int(row[h["cycle"]])
            if stage == "ingress":
                arrivals.append(Access(*key, int(row[h["pc"]], 0), int(row[h["virtual_line"]], 0), tick))
            elif stage == "lookup":
                if key in outcomes:
                    raise ValueError("duplicate completed LLC lookup for {}".format(key))
                outcomes[key] = int(row[h["hit"]])
            elif stage == "dispatch":
                dispatch[key] = tick
            elif stage == "fill" and key in dispatch:
                elapsed = tick - dispatch[key]
                if elapsed > 0:
                    latencies.append(elapsed)
    return arrivals, outcomes, latencies


def read_head(folder):
    waits = defaultdict(int)
    with event_file(folder, "head") as stream:
        for row in csv.DictReader(stream):
            key = Identity(*(int(row[k]) for k in ("cpu", "instr_id", "operand")))
            waits[key] += int(row["cycles"])
    return waits


def select(nopf, stride, out, cap):
    base_arrivals, base_outcomes, _ = read_llc(nopf)
    base_misses = {identity(a): (a.pc, a.line) for a in base_arrivals
                   if base_outcomes.get(identity(a)) == 0}
    arrivals, outcomes, services = read_llc(stride)
    if not services:
        raise ValueError("no positive measured LLC dispatch-to-demand-fill times")
    delay = int(median(services))
    waits = read_head(stride)
    by_page = defaultdict(list)
    ticks = defaultdict(list)
    last_same_line = {}
    opportunities = []
    for a in sorted(arrivals, key=lambda a: a.cycle):
        page = (a.cpu, a.line // 64)
        line_key = (a.cpu, a.line)
        key = identity(a)
        if key in base_misses and (a.pc, a.line) != base_misses[key]:
            raise ValueError("cross-run virtual-line or PC mismatch")
        if outcomes.get(key) == 0 and key in base_misses and waits.get(key, 0) > 0:
            # Stay within the current 4-KiB page. Once another demand for the target
            # has arrived, an even earlier prefetch may be consumed by that demand.
            low = max(a.cycle - 4 * delay, last_same_line.get(line_key, -1) + 1)
            start = bisect.bisect_left(ticks[page], low)
            end = bisect.bisect_right(ticks[page], a.cycle - delay)
            wanted = a.cycle - 2 * delay
            where = bisect.bisect_left(ticks[page], wanted, start, end) if start < end else start
            # Try a few nearby legal triggers so one trigger is not used twice.
            selected = sorted(range(max(start, where - 3), min(end, where + 4)),
                              key=lambda i: (abs(ticks[page][i] - wanted), i))
            anchors = [by_page[page][i] for i in selected if by_page[page][i].line != a.line]
            if anchors:
                opportunities.append((waits[key], a, anchors))
        ticks[page].append(a.cycle)
        by_page[page].append(a)
        last_same_line[line_key] = a.cycle

    used = set()
    picks = []
    for wait, target, anchors in sorted(opportunities, key=lambda item: (-item[0], item[1].cycle, item[1].cpu)):
        anchor = next((x for x in anchors if identity(x) not in used), None)
        if anchor is None:
            continue
        used.add(identity(anchor))
        picks.append((anchor, target, wait))
        if len(picks) == cap:
            break
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plan.csv").open("w", newline="") as fp, (out / "targets.csv").open("w", newline="") as ft:
        plan, targets = csv.writer(fp,lineterminator="\n"), csv.writer(ft,lineterminator="\n")
        plan.writerow(("cpu","instr_id","operand","delta"))
        targets.writerow(("cpu","anchor_instr","anchor_operand","target_instr","target_operand","virtual_line",
                          "stride_trigger_cycle","stride_target_cycle","head_wait_cycles","delta"))
        for a, t, wait in picks:
            plan.writerow((*identity(a), t.line - a.line))
            targets.writerow((*identity(a), t.instr, t.operand, t.line, a.cycle, t.cycle, wait, t.line - a.line))
    summary = dict(median_service=delay, measured_services=len(services), eligible_targets=len(opportunities),
                   selected=len(picks), cap=cap, lead_min=delay, lead_max=4 * delay)
    (out / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("nopf", "stride", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--cap", type=int, default=256)
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("cap must be positive")
    print(json.dumps(select(args.nopf, args.stride, args.out, args.cap)))
