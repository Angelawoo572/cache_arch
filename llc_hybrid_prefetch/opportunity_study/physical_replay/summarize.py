"""Measure timing, attributable coverage and original case-study example for each placement."""

import csv
import gzip
from collections import Counter
from run import CASES, HERE, RAW, ROOT

EXAMPLE = (10145357, 0)
ANCHOR = (10145041, 0)


def records(p, name):
    with gzip.open(p / (name + ".csv.gz"), "rt") as f:
        yield from csv.DictReader(f)


def tab(p, name):
    with (p / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def outcome(p):
    look = {}
    ingress = {}
    state = {}
    for r in records(p, "loads"):
        if r["level"] != "LLC" or r["stage"] not in ("ingress", "lookup"):
            continue
        k = (int(r["instr_id"]), int(r["operand"]))
        if r["stage"] == "lookup":
            look[k] = (int(r["pc"]), int(r["virtual_line"]), int(r["hit"]))
        elif k in (ANCHOR, EXAMPLE):
            ingress[k] = (int(r["cycle"]), int(r["physical_line"]))
    for r in records(p, "head"):
        k = (int(r["instr_id"]), int(r["operand"]))
        if k == EXAMPLE:
            state[k] = state.get(k, 0) + int(r["cycles"])
    return look, ingress, state.get(EXAMPLE, 0)


def save(name, items):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=items[0], lineterminator="\n")
        w.writeheader()
        w.writerows(items)


def main():
    results = []
    examples = []
    with (RAW / "history/cactus_live/r2_reference.csv").open() as f:
        reference = {
            (int(r["instr_id"]), int(r["operand"])): r for r in csv.DictReader(f)
        }
    for layout in ("original", "perm"):
        root = RAW / "physical_runs" / layout
        base, _, _ = outcome(root / "nopf")
        misses = {k for k, v in base.items() if v[-1] == 0}
        stride_cycles = int(tab(root / "stride", "cores")[0]["cycles"])
        for mode in CASES:
            p = root / mode
            dur = int(tab(p, "cores")[0]["cycles"])
            stats = {r["key"]: int(r["value"]) for r in tab(p, "stats")}
            if layout == "original" and mode in ("nopf", "stride"):
                native = ROOT / "raw/runs" / ("cactus-r2-" + mode)
                if dur != int(tab(native, "cores")[0]["cycles"]) or tab(
                    p, "stats"
                ) != tab(native, "stats"):
                    raise ValueError("baseline mismatch")
            out, ing, wait = outcome(p)
            if any(v[:2] != base[k][:2] for k, v in out.items() if k in base):
                raise ValueError("dynamic identity mismatch")
            cand = {}
            claims = {}
            count = Counter()
            for r in records(p, "candidates"):
                pid = int(r["id"])
                s = r["stage"]
                if s == "proposed":
                    cand[pid] = {
                        "proposed": int(r["cycle"]),
                        "line": int(r["target"]),
                        "aux": r["reason"] == "intervention",
                        "anchor": int(r["instr_id"]),
                    }
                if pid not in cand:
                    continue
                q = cand[pid]
                q[s] = int(r["cycle"])
                if q["aux"]:
                    count[s] += 1
                if s in ("timely", "late"):
                    k = (int(r["demand_instr_id"]), int(r["demand_operand"]))
                    claims[k] = (s, q)
            choices = tab(p, "wide") if (p / "wide.csv").is_file() else []

            def original_choice(r):
                k = (int(r["instr"]), int(r["operand"]))
                if k not in reference:
                    return False
                ref = reference[k]
                field = "plus128_target" if mode == "plus128" else "target_line"
                return (int(r["base"]), int(r["target"])) == (
                    int(ref["base_line"]),
                    int(ref[field]),
                )

            common = (
                sum(original_choice(r) for r in choices) if layout == "original" else ""
            )
            by = Counter(r["action"] for r in choices)
            if count["proposed"] != by["override"]:
                raise ValueError("override mismatch")
            for k, (s, q) in claims.items():
                if k in misses and s == "timely" and out.get(k, (0, 0, -1))[-1] == 1:
                    count["timely_matched"] += 1
                if k in misses and s == "late" and out.get(k, (0, 0, -1))[-1] == 0:
                    count["late_matched"] += 1
            results.append(
                dict(
                    layout=layout,
                    mode=mode,
                    cycles=dur,
                    cycles_saved_vs_stride=stride_cycles - dur,
                    time_saved_pct=round(
                        100 * (stride_cycles - dur) / stride_cycles, 5
                    ),
                    nopf_misses=len(misses),
                    llc_lookups=len(out),
                    llc_hits=sum(v[-1] for v in out.values()),
                    paired_miss_to_hit=sum(out[k][-1] for k in misses if k in out),
                    paired_hit_to_miss=sum(
                        1
                        for k, v in out.items()
                        if k in base and base[k][-1] == 1 and v[-1] == 0
                    ),
                    direct_timely_on_original_misses=count["timely_matched"],
                    direct_late_on_original_misses=count["late_matched"],
                    choices=len(choices),
                    identical_choice_at_original_anchor=common,
                    bounded_page_rejection=by["page_history"],
                    agrees=by["agree"],
                    override=by["override"],
                    aux_accepted=count["accepted"],
                    aux_timely=count["timely"],
                    aux_late=count["late"],
                    aux_unused_evictions=count["evicted"],
                    total_accepted=stats["prefetch_accepted"],
                    dram_bytes=sum(
                        stats["dram_" + t + "_bytes"]
                        for t in ("demand", "prefetch", "write")
                    ),
                    head_union=stats["head_unique_cycles_core_0"],
                )
            )
            selected = next(
                (r for r in choices if (int(r["instr"]), int(r["operand"])) == ANCHOR),
                {},
            )
            claim, life = claims.get(EXAMPLE, ("", {}))
            examples.append(
                dict(
                    layout=layout,
                    mode=mode,
                    anchor_instr=ANCHOR[0],
                    anchor_ingress=ing.get(ANCHOR, ("", ""))[0],
                    anchor_physical=ing.get(ANCHOR, ("", ""))[1],
                    anchor_choice=selected.get("target", ""),
                    anchor_action=selected.get("action", ""),
                    target_instr=EXAMPLE[0],
                    target_physical=ing.get(EXAMPLE, ("", ""))[1],
                    target_ingress=ing.get(EXAMPLE, ("", ""))[0],
                    target_lookup_hit=out.get(EXAMPLE, ("", "", ""))[-1],
                    target_head_wait=wait,
                    pf_firstuse=claim,
                    pf_anchor_instr=life.get("anchor", ""),
                    pf_proposed=life.get("proposed", ""),
                    pf_dispatch=life.get("dispatch", ""),
                    pf_fill=life.get("fill", ""),
                )
            )
    save("placement_results.csv", results)
    save("target_example.csv", examples)


if __name__ == "__main__":
    main()
