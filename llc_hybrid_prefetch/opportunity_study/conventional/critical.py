"""Compare high and zero retirement-head wait targets at matched prefetch decision contexts."""

import bisect
import csv
import gzip
import json
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from run import BASE, HERE, RAW, comparison_cases, csvs, run, totals


def rows(d, name):
    p = d / (name + ".csv.gz")
    return gzip.open(p, "rt") if p.exists() else (d / (name + ".csv")).open()


def llc(d):
    arrivals = []
    hits = {}
    disp = {}
    fills = {}
    with rows(d, "loads") as f:
        for line in f:
            if ",LLC," not in line:
                continue
            a = line.rstrip().split(",")
            stage = a[7]
            if stage not in ("lookup", "ingress", "dispatch", "fill"):
                continue
            cpu, i, o, pc, va, pa = map(int, a[:6])
            key = (cpu, i, o)
            cycle = int(a[8])
            if stage == "lookup":
                hits[key] = int(a[9])
            if stage == "ingress":
                arrivals.append((cycle, key, pc, va, pa, int(a[10]), int(a[13])))
            elif stage == "dispatch":
                disp[key] = cycle
            elif stage == "fill":
                fills[key] = cycle
    arrivals.sort(key=lambda r: r[0])
    return arrivals, hits, disp, fills


def head(d):
    with rows(d, "head") as f:
        return {
            (int(r["cpu"]), int(r["instr_id"]), int(r["operand"])): int(r["cycles"])
            for r in csv.DictReader(f)
        }


def candidate_times(d):
    events = {}
    out = {}
    with rows(d, "candidates") as f:
        for line in f:
            if ",intervention," not in line and not events:
                continue
            a = line.rstrip().split(",")
            if a[0] == "id":
                continue
            id = int(a[0])
            stage = a[8]
            if stage == "proposed" and a[9] == "intervention":
                key = tuple(map(int, a[1:4]))
                out[key] = {}
                events[id] = out[key]
            if id in events:
                log = events[id]
                log[stage] = int(a[7])
                if stage in ("timely", "late"):
                    log["consumer"] = tuple(map(int, a[10:13]))
    return out


def dump(path, data):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)


def choose(base, folder, count):
    n_a, n_hits, *_ = llc(BASE / "raw/qualified" / f"{base}-nopf")
    identities = {e[1]: (e[2], e[3]) for e in n_a if n_hits.get(e[1]) == 0}
    d = BASE / "raw/qualified" / f"{base}-stride"
    a, h, dispatch, fill = llc(d)
    wait = head(d)
    service = int(statistics.median(fill[k] - t for k, t in dispatch.items() if k in fill and fill[k] > t))
    times = defaultdict(list)
    visits = defaultdict(list)
    last_line = {}
    opp = []
    for e in a:
        t, k, pc, va, pa, mshr, dram = e
        cpu = k[0]
        page = (cpu, pa // 64)
        line = (cpu, pa)
        if cpu == 0 and h.get(k) == 0 and identities.get(k) == (pc, va):
            lo = max(last_line.get(line, -1) + 1, t - 3 * service)
            first = bisect.bisect_left(times[page], lo)
            end = bisect.bisect_right(times[page], t - service)
            if end > first:
                ix = bisect.bisect_left(times[page], t - 2 * service, first, end)
                candidate = sorted(
                    visits[page][max(first, ix - 3) : min(end, ix + 4)],
                    key=lambda x: abs(x[0] - (t - 2 * service)),
                )
                anchor = next((x for x in candidate if x[4] != pa), None)
                if anchor:
                    lead = t - anchor[0]
                    delta = pa - anchor[4]
                    # Hold target PC, coarse pressure, lead, and distance sign/magnitude constant.
                    strata = (
                        pc,
                        (anchor[5] >= 16 or anchor[6] >= 16),
                        int(2 * lead / service),
                        int(abs(delta) > 8),
                        int(delta > 0),
                    )
                    opp.append((wait.get(k, 0), strata, anchor, e))
        last_line[line] = t
        times[page].append(t)
        visits[page].append(e)
    cut = max(1, int(statistics.quantiles([v[0] for v in opp if v[0]], n=4)[2]))
    low = defaultdict(list)
    for v in opp:
        if v[0] == 0:
            low[v[1]].append(v)
    used = set()
    pairs = []
    for hi in sorted((v for v in opp if v[0] >= cut), key=lambda x: (-x[0], x[3][0])):
        if hi[2][1] in used or hi[3][1] in used:
            continue
        options = low[hi[1]]
        # Prefer the closest lead among constrained alternatives with disjoint anchors/targets.
        lo = min(
            (
                v
                for v in options
                if v[2][1] not in used
                and v[3][1] not in used
                and len({hi[2][1], hi[3][1], v[2][1], v[3][1]}) == 4
            ),
            key=lambda v: abs((hi[3][0] - hi[2][0]) - (v[3][0] - v[2][0])),
            default=None,
        )
        if lo is None:
            continue
        pairs.append((hi, lo))
        used.update((hi[2][1], hi[3][1], lo[2][1], lo[3][1]))
        if len(pairs) == count:
            break
    if not pairs:
        raise ValueError("no matched high/zero-head targets")
    targets = []
    for group, idx in [("high", 0), ("zero", 1)]:
        plan = []
        for pair_id, p in enumerate(pairs, 1):
            w, st, anchor, target = p[idx]
            at, ak, ap, av, al, am, aq = anchor
            t, k, pc, va, pa, m, q = target
            plan.append(dict(cpu=ak[0], instr_id=ak[1], operand=ak[2], delta=pa - al))
            targets.append(
                dict(
                    case=base,
                    group=group,
                    pair=pair_id,
                    cpu=k[0],
                    anchor=ak[1],
                    anchor_operand=ak[2],
                    anchor_pc=ap,
                    target=k[1],
                    target_operand=k[2],
                    target_pc=pc,
                    physical_line=pa,
                    delta=pa - al,
                    baseline_anchor=at,
                    baseline_demand=t,
                    baseline_head=w,
                    baseline_dispatch=dispatch.get(k, ""),
                    baseline_fill=fill.get(k, ""),
                    baseline_mshr=am,
                    baseline_dram_rq=aq,
                )
            )
        dump(folder / (group + ".csv"), plan)
    dump(folder / "targets.csv", targets)
    (folder / "selection.json").write_text(
        json.dumps(
            dict(
                tuple_matches=len(pairs),
                eligible_targets=len(opp),
                nonzero_head_q75=cut,
                service_cycles=service,
            ),
            indent=2,
        )
        + "\n"
    )


def summarize(c, plans):
    original = BASE / "raw/qualified" / f"{c['case']}-stride"
    b = totals(original)
    baseline_cores = list(map(lambda r: int(r["cycles"]), csvs(original / "cores.csv")))
    targets = csvs(plans / "targets.csv")
    summary = []
    examples = []
    for group in ("high", "zero"):
        rp = RAW / "runs" / f"{c['case']}__stride__head-{group}"
        a, h, disp, fills = llc(rp)
        ingress = {r[1]: r[0] for r in a}
        cl = candidate_times(rp)
        wait = head(rp)
        t = totals(rp)
        core_cycles = [int(r["cycles"]) for r in csvs(rp / "cores.csv")]
        selected = [r for r in targets if r["group"] == group]
        detail = []
        for row in selected:
            c0 = int(row["cpu"])
            anchor = (c0, int(row["anchor"]), int(row["anchor_operand"]))
            key = (c0, int(row["target"]), int(row["target_operand"]))
            l = cl.get(anchor, {})
            detail.append(
                dict(
                    row,
                    roi_trigger=l.get("proposed", ""),
                    ready=l.get("ready", ""),
                    accepted=l.get("accepted", ""),
                    dispatch=l.get("dispatch", ""),
                    fill=l.get("fill", ""),
                    replay_demand=ingress.get(key, ""),
                    replay_target_hit=h.get(key, ""),
                    timely=int(l.get("consumer") == key and "timely" in l),
                    late=int(l.get("consumer") == key and "late" in l),
                    replay_head=wait.get(key, 0),
                )
            )
        examples.extend(detail)

        def mean(field):
            xs = [float(x[field]) for x in detail if x[field] != ""]
            return round(statistics.mean(xs), 2) if xs else ""

        def elapsed(a, b):
            xs = [float(v[b]) - float(v[a]) for v in detail if v[a] != "" and v[b] != ""]
            return round(statistics.median(xs), 2) if xs else ""

        summary.append(
            dict(
                case=c["case"],
                group=group,
                pairs=len(selected),
                actual_anchors=t["plan_matched"],
                accepted_overrides=sum(x["accepted"] != "" for x in detail),
                timely_intended=sum(x["timely"] for x in detail),
                late_intended=sum(x["late"] for x in detail),
                timely_target_hits=sum(x["replay_target_hit"] == 1 for x in detail),
                baseline_mean_head=mean("baseline_head"),
                replay_mean_head=mean("replay_head"),
                baseline_median_trigger_to_demand=elapsed("baseline_anchor", "baseline_demand"),
                replay_median_ready_to_dispatch=elapsed("ready", "dispatch"),
                replay_median_dispatch_to_fill=elapsed("dispatch", "fill"),
                replay_median_fill_slack=elapsed("fill", "replay_demand"),
                baseline_core0=baseline_cores[0],
                replay_core0=core_cycles[0],
                baseline_core1=baseline_cores[1],
                replay_core1=core_cycles[1],
                saved_makespan=max(baseline_cores) - max(core_cycles),
                dram_delta=sum(
                    t[k] - b[k] for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes")
                ),
                accepted_delta=t["prefetch_accepted"] - b["prefetch_accepted"],
            )
        )
    dump(HERE / "criticality.csv", summary)
    dump(HERE / "criticality_examples.csv", examples)


if __name__ == "__main__":
    case = comparison_cases()["mcf-cactus-r1"]
    folder = RAW / "plans"
    folder.mkdir(exist_ok=True, parents=True)
    choose(case["case"], folder, 96)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda g: run(case, "stride", "head-" + g, plan=folder / (g + ".csv"), detail="1"),
                ("high", "zero"),
            )
        )
    summarize(case, folder)
