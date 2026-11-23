#!/usr/bin/env python3
"""Turn measurement/tables/*.csv into the Markdown tables of the findings.

    python analysis/np_markdown.py                  # print every table
    python analysis/np_markdown.py --splice FILE    # write them into FILE
    python analysis/np_markdown.py --ranges         # ranges quoted in the prose
    python analysis/np_markdown.py --tokens         # the same as ready-made phrases

--splice replaces whatever stands between <!-- table:NAME --> and
<!-- /table:NAME --> in FILE, so no number in a table is typed by hand.
"""

import argparse
import csv
import math
import os
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("GEM5_STUDY", HERE.parent))
TABLES = STUDY / "measurement" / "tables"

KERNEL = {"bfs": "BFS", "pr": "PageRank", "cc": "Components", "bc": "Betweenness", "sssp": "SSSP"}
ARMS = ("stride", "fast", "bop", "ampm")
ARM_NAME = {
    "stride": "Stride",
    "fast": "Stride + signature path",
    "bop": "Stride + best offset",
    "ampm": "Stride + access map",
}


def read(name):
    path = TABLES / name
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def key(row):
    return (row["workload"], int(row["offset"]))


def label(workload, offset):
    parts = workload.split("_")
    where = "start" if offset == 0 else f"+{offset // 1_000_000}M"
    if parts[0] == "gap":
        return f"{KERNEL.get(parts[1], parts[1])} {parts[2]}, {where}"
    return parts[1]


def num(row, name):
    value = row.get(name, "") if row else ""
    return float(value) if value not in ("", "None") else None


def fmt(value, digits=0, pct=False):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if pct:
        return f"{100 * value:.{digits}f}%"
    return f"{value:,.{digits}f}"


def form_name(code):
    code = int(code)
    return f"shift {code}" if code >= 0 else f"bit vector, {1 << -code}-bit words"


def ratio(a, b):
    return a / b if a is not None and b else None


def render(header, rows):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


class Data:
    def __init__(self, kind, arm):
        self.kind, self.arm = kind, arm

        def by_arm(name):
            table = {}
            for row in read(name):
                if row["workload"].startswith(kind):
                    table.setdefault(row["arm"], {})[key(row)] = row
            return table

        self.windows = by_arm("windows.csv")
        self.m1, self.m2, self.m3 = by_arm("m1_residual.csv"), by_arm("m2_mechanism.csv"), by_arm("m3_lead.csv")
        self.m4, self.m5, self.m6 = by_arm("m4_entropy.csv"), by_arm("m5_relation_reuse.csv"), by_arm("m6_trigger_rate.csv")
        self.m7, self.m8 = by_arm("m7_operand_age.csv"), by_arm("m8_translation.csv")
        self.walker = by_arm("m3_walker_bound.csv")
        self.top, self.relations = {}, {}
        for row in read("m5_top_relations.csv"):
            if row["workload"].startswith(kind) and row["arm"] == arm:
                self.relations.setdefault(key(row), []).append(row)
                if row["rank"] == "1":
                    self.top[key(row)] = row
        self.order = sorted(self.windows.get(arm, {}))

    def name(self, k):
        return label(*k)

    def relations_per_start(self, k):
        """Among the relations that carry 99% of the value-derived residual
        stall, the largest number that start at one producer PC."""
        covered, per_pc = 0.0, {}
        for row in sorted(self.relations.get(k, []), key=lambda r: int(r["rank"])):
            if covered >= 0.99:
                break
            covered += float(row["stall_share"])
            per_pc[row["producer_pc"]] = per_pc.get(row["producer_pc"], 0) + 1
        return max(per_pc.values()) if per_pc else None

    # ---- tables -------------------------------------------------------
    def arms(self):
        rows, logs = [], {arm: [] for arm in ARMS}
        for k in self.order:
            base = self.windows[self.arm][k]
            cells = [self.name(k), fmt(num(base, "ipc_nopf"), 3)]
            for arm in ARMS:
                row = self.windows.get(arm, {}).get(k)
                speed = ratio(num(row, "cycles_nopf"), num(row, "cycles_fast")) if row else None
                if speed:
                    logs[arm].append(math.log(speed))
                cells.append(fmt(speed, 2))
            rows.append(cells)
        mean = ["Geometric mean", ""]
        for arm in ARMS:
            mean.append(fmt(math.exp(statistics.fmean(logs[arm])), 2) if logs[arm] else "–")
        rows.append(mean)
        return render(["Window", "IPC, no prefetch"] + [ARM_NAME[a] for a in ARMS], rows)

    def residual(self):
        rows = []
        for k in self.order:
            r, w = self.m1[self.arm][k], self.windows[self.arm][k]
            before = num(r, "beyond_l2.stall_nopf")
            after = num(r, "beyond_l2.stall_fast_late") + num(r, "beyond_l2.stall_fast_miss")
            rows.append([
                self.name(k), fmt(num(r, "beyond_l2.original_misses")),
                fmt(num(r, "beyond_l2.timely")), fmt(num(r, "beyond_l2.other_found_in_l1")),
                fmt(num(r, "beyond_l2.other_found_in_l2")),
                fmt(num(r, "beyond_l2.late")), fmt(num(r, "beyond_l2.miss")),
                fmt(ratio(before, num(w, "cycles_nopf")), 0, True),
                fmt(ratio(after, num(w, "cycles_fast")), 0, True),
            ])
        return render(
            ["Window", "Original misses beyond L2", "First use of a prefetched line in L1", "Found in L1",
             "Found in L2", "Late", "Missed again",
             "Their stall without prefetching, share of run time", "Residual stall, share of run time"], rows)

    def residual_arms(self):
        rows = []
        for k in self.order:
            cells = [self.name(k)]
            for arm in ARMS:
                r, w = self.m1.get(arm, {}).get(k), self.windows.get(arm, {}).get(k)
                if r is None:
                    cells.append("–")
                    continue
                after = num(r, "beyond_l2.stall_fast_late") + num(r, "beyond_l2.stall_fast_miss")
                cells.append(fmt(ratio(after, num(w, "cycles_fast")), 0, True))
            rows.append(cells)
        return render(["Window"] + [ARM_NAME[a] for a in ARMS], rows)

    def mechanism(self):
        names = ("same_pc_delta", "value_depth_1", "value_depth_2_plus", "offset_on_known_page", "temporal_repeat", "none")
        rows = []
        for k in self.order:
            r = self.m2[self.arm][k]
            total = num(r, "stall")
            rows.append([self.name(k), fmt(num(r, "targets"))]
                        + [fmt(ratio(num(r, f"{n}.stall"), total), 1, True) for n in names]
                        + [fmt(ratio(num(r, "value_through_bit_vector.stall"), total), 1, True)])
        return render(["Window", "Residual targets", "Same-PC delta", "Loaded value, 1 link",
                       "Loaded value, chain of misses", "Offset on a known page", "Repeat of an earlier pair", "None",
                       "Of the value classes: bit-vector index"], rows)

    def lead(self):
        rows = []
        for k in self.order:
            r = self.m3[self.arm][k]
            rows.append([
                self.name(k), fmt(num(r, "targets")),
                fmt(num(r, "slack_cycles.p50")), fmt(num(r, "slack_cycles.p90")),
                fmt(num(r, "target_latency_cycles.p50")),
                fmt(num(r, "stall_share_slack_at_least_50pct_of_latency"), 1, True),
                fmt(num(r, "stall_share_consumer_fetched_after_value"), 0, True),
                fmt(num(r, "l1d_blocked_no_miss_register_share_of_cycles"), 1, True),
                fmt(num(r, "stall_share_removable_by_one_hop_bound"), 0, True),
            ])
        return render(["Window", "Value-derived residual targets", "Slack p50, cycles", "Slack p90",
                       "Miss latency p50, cycles", "Stall with slack of at least half the latency",
                       "Stall whose consumer was fetched after the value returned",
                       "Cycles with L1D blocked for lack of a miss register",
                       "Removable by one link at best"], rows)

    def reach(self):
        rows = []
        for k in self.order:
            r = self.walker.get(self.arm, {}).get(k)
            if r is None or num(r, "same_line.hops_1") is None:
                continue
            rows.append([
                self.name(k),
                fmt(num(r, "stepped.hops_1.distance_64"), 0, True),
                fmt(num(r, "same_line.hops_1"), 0, True),
                fmt(num(r, "fill_tap.hops_1"), 0, True),
                fmt(num(r, "any.hops_1"), 0, True),
                fmt(num(r, "any.hops_2"), 0, True),
                fmt(num(r, "any.hops_3"), 0, True),
                fmt(num(r, "share_of_all_residual_stall.any.hops_3"), 0, True),
                fmt(num(r, "targets_per_start.stepped.p99")),
                fmt(num(r, "targets_per_start.same_line.p99")),
            ])
        return render(["Window", "Stepped root, 1 link", "Same line, 1 link", "Fill tap, 1 link",
                       "Any of the three, 1 link", "Up to 2 links", "Up to 3 links",
                       "The same as a share of all residual stall",
                       "Targets per stepped start, p99", "Targets per line, p99"], rows)

    def reach_distance(self):
        rows = []
        for k in self.order:
            r = self.walker.get(self.arm, {}).get(k)
            if r is None or num(r, "stepped.hops_3.distance_1") is None:
                continue
            rows.append([self.name(k)] + [fmt(num(r, f"stepped.hops_3.distance_{d}"), 0, True) for d in (1, 2, 4, 8, 16, 32, 64)])
        return render(["Window"] + [f"{d} ahead" for d in (1, 2, 4, 8, 16, 32, 64)], rows)

    def gate(self):
        rows = []
        for k in self.order:
            r = self.m4[self.arm][k]
            rows.append([
                self.name(k), fmt(num(r, "relation_instances")),
                fmt(num(r, "relation_instance_positive_rate"), 1, True),
                fmt(num(r, "instance_gate_bits_prior.online"), 3),
                fmt(num(r, "instance_gate_bits_given_relation.online"), 3),
                fmt(num(r, "instance_gate_bits_given_relation_producer_level.online"), 3),
                fmt(num(r, "instance_gate_bits_given_relation_last_2.online"), 3),
                fmt(num(r, "instance_gate_bits_given_relation_producer_level_last_2.online"), 3),
                fmt(num(r, "instance_gate_bits_given_relation_producer_level_last_2.interleaved"), 3),
            ])
        return render(["Window", "Loads named by a relation", "Of them residual targets", "No context",
                       "Relation", "Relation, producer's cache level", "Relation, last 2 outcomes",
                       "All three", "All three, held out in phase"], rows)

    def choice(self):
        rows = []
        for k in self.order:
            r = self.m4[self.arm][k]
            rows.append([
                self.name(k),
                fmt(num(r, "relation_bits_prior.online"), 3),
                fmt(num(r, "relation_bits_given_origin_pc.online"), 3),
                fmt(num(r, "relation_bits_given_origin_pc.interleaved"), 3),
                fmt(num(r, "relation_bits_given_consumer_pc.online"), 3),
                fmt(num(r, "residual_stall_whose_origin_missed_l1d"), 0, True),
            ])
        return render(["Window", "No context", "PC of the starting load", "The same, held out in phase",
                       "PC of the target load", "Residual stall whose starting load missed L1D"], rows)

    def sizes(self):
        rows = []
        for k in self.order:
            a, b, c, d = self.m5[self.arm][k], self.m6[self.arm][k], self.m7[self.arm][k], self.m8[self.arm][k]
            top = self.top.get(k)
            rows.append([
                self.name(k),
                fmt(num(a, "relations_for_50pct_of_stall")), fmt(num(a, "relations_for_90pct_of_stall")),
                fmt(num(a, "relations_for_99pct_of_stall")),
                fmt(self.relations_per_start(k)),
                (f"{form_name(top['shift'])}, {100 * float(top['stall_share']):.0f}%" if top else "–"),
                fmt(num(c, "micro_ops_on_path.p99")),
                fmt(num(c, "loads_between_nearest_producer_and_consumer.p50")),
                fmt(num(c, "loads_between_nearest_producer_and_consumer.p99")),
                fmt(num(c, "l1d_misses_between_nearest.p99")),
                fmt(num(b, "loads_at_origin_pcs_per_kilocycle"), 1),
                fmt(num(b, "fast.l1d_miss_loads_per_kilocycle"), 1),
                fmt(num(b, "fast.share_gaps_under_4_cycles"), 0, True),
                fmt(num(d, "target_page_among_last_64_miss_pages"), 0, True),
            ])
        return render(["Window", "Relations for 50% of value-derived residual stall", "For 90%", "For 99%",
                       "Most of those at one starting PC",
                       "Largest relation", "Micro-ops from value to address, p99",
                       "Loads from nearest producer to target, p50", "The same, p99",
                       "L1D misses from nearest producer to target, p99",
                       "Loads at starting PCs per 1,000 cycles", "L1D-miss loads per 1,000 cycles",
                       "L1D-miss loads within 4 cycles of the previous",
                       "Target page among the last 64 miss pages"], rows)

    def validation_programs(self):
        rows = []
        for r in read("validation.csv"):
            rows.append([
                r["program"].split("_")[1], f"{int(r['committed_loads']):,}",
                r["V1_same_loads_in_3_arms"],
                f"{r['V2_non_intrusion']} ({r['V2_statistics_differing']} of {int(r['V2_statistics_compared']):,} differ)",
                f"{r['V3_known_structure']}: {r['V3_found_shift_base']}",
                f"{r['V4_reconciliation_3_arms']} ({r['V4_worst_count_difference_percent']}%)",
                r["V5_storage_3_arms"],
            ])
        return render(["Program", "Committed loads", "V1 same loads in 3 arms", "V2 capture changes nothing",
                       "V3 known relations found (shift, base)", "V4 counts match gem5 (worst difference)",
                       "V5 storage"], rows)

    def validation_windows(self):
        rows = []
        for r in read("validation_windows.csv"):
            if not r["workload"].startswith(self.kind):
                continue
            rows.append([
                label(r["workload"], int(r["offset"])), str(len(r["arms"].split())),
                f"{int(r['committed_loads']):,}", r["V1_same_loads_in_all_arms"],
                r["load_values_equal_in_all_arms"],
                f"{r['V4_reconciliation_all_arms']} ({r['V4_worst_count_difference_percent']}%)",
                r["V5_storage_all_arms"],
            ])
        return render(["Window", "Arms", "Committed loads", "V1 same loads in every arm",
                       "Load values equal in every arm", "V4 counts match gem5 (worst difference)", "V5 storage"], rows)

    def validation_bound(self):
        rows = []
        for r in read("validation_spp_bound.csv"):
            if not r["workload"].startswith(self.kind):
                continue
            rows.append([
                label(r["workload"], int(r["offset"])), r["class_in_fast"].replace("Prefetcher", ""),
                f"{int(r['statistics_compared']):,}", r["statistics_differing"],
                r["capture_window_equal"], r["lookaheads_stopped_in_fastcheck"], r["same"],
            ])
        return render(["Window", "Class in the fast capture", "Statistics compared", "Differing",
                       "Capture window equal", "Walks stopped by the bound", "Same"], rows)

    TABLES = ("arms", "residual", "residual_arms", "mechanism", "lead", "reach", "reach_distance",
              "gate", "choice", "sizes", "validation_programs", "validation_windows", "validation_bound")

    # ---- ranges for the prose ----------------------------------------
    def ranges(self):
        def span(values, digits=0, pct=False):
            values = [v for v in values if v is not None]
            if not values:
                return "–"
            return f"{fmt(min(values), digits, pct)} to {fmt(max(values), digits, pct)} (median {fmt(statistics.median(values), digits, pct)}, n={len(values)})"

        w, m1, m2, m3, m4, m5 = (t.get(self.arm, {}) for t in (self.windows, self.m1, self.m2, self.m3, self.m4, self.m5))
        walker = self.walker.get(self.arm, {})
        order = self.order
        out = []
        out.append(("speedup of the arm over no prefetching", span([ratio(num(w[k], "cycles_nopf"), num(w[k], "cycles_fast")) for k in order], 2)))
        out.append(("residual stall, share of the arm's run time", span([ratio(num(m1[k], "beyond_l2.stall_fast_late") + num(m1[k], "beyond_l2.stall_fast_miss"), num(w[k], "cycles_fast")) for k in order], 0, True)))
        value = [ratio(num(m2[k], "value_depth_1.stall") + num(m2[k], "value_depth_2_plus.stall"), num(m2[k], "stall")) for k in order]
        out.append(("value-derived share of residual stall", span(value, 0, True)))
        out.append(("same-PC delta share of residual stall", span([ratio(num(m2[k], "same_pc_delta.stall"), num(m2[k], "stall")) for k in order], 0, True)))
        out.append(("none share of residual stall", span([ratio(num(m2[k], "none.stall"), num(m2[k], "stall")) for k in order], 0, True)))
        out.append(("slack p50, cycles", span([num(m3[k], "slack_cycles.p50") for k in order])))
        out.append(("miss latency p50, cycles", span([num(m3[k], "target_latency_cycles.p50") for k in order])))
        out.append(("stall with slack >= half latency", span([num(m3[k], "stall_share_slack_at_least_50pct_of_latency") for k in order], 1, True)))
        out.append(("relations for 90% / 99%", span([num(m5[k], "relations_for_90pct_of_stall") for k in order]) + " / " + span([num(m5[k], "relations_for_99pct_of_stall") for k in order])))
        out.append(("relation choice bits given origin PC, online", span([num(m4[k], "relation_bits_given_origin_pc.online") for k in order], 3)))
        out.append(("instance gate bits: prior / relation / all three, online", " / ".join(span([num(m4[k], f"instance_gate_bits_{name}.online") for k in order], 3) for name in ("prior", "given_relation", "given_relation_producer_level_last_2"))))
        out.append(("residual stall whose origin missed L1D", span([num(m4[k], "residual_stall_whose_origin_missed_l1d") for k in order], 0, True)))
        for name in ("stepped.hops_1.distance_64", "same_line.hops_1", "fill_tap.hops_1", "any.hops_1", "any.hops_3", "share_of_all_residual_stall.any.hops_3"):
            out.append((f"walker {name}", span([num(walker[k], name) for k in order if k in walker], 0, True)))
        return out


def tokens(data):
    """Ranges across windows that the prose quotes, as ready-made phrases."""
    arm, order = data.arm, data.order
    w, m1, m2, m3, m4, m5, m6, m7, m8 = (
        t.get(arm, {}) for t in (data.windows, data.m1, data.m2, data.m3, data.m4, data.m5, data.m6, data.m7, data.m8)
    )
    walker = data.walker.get(arm, {})

    def column(table, name):
        return [v for v in (num(table[k], name) for k in order if k in table) if v is not None]

    def span(values, digits=0, pct=False):
        if not values:
            return "–"
        return f"{fmt(min(values), digits, pct)} to {fmt(max(values), digits, pct)}"

    def median(values, digits=0, pct=False):
        return fmt(statistics.median(values), digits, pct) if values else "–"

    residual = [ratio(num(m1[k], "beyond_l2.stall_fast_late") + num(m1[k], "beyond_l2.stall_fast_miss"), num(w[k], "cycles_fast")) for k in order]
    value = [ratio(num(m2[k], "value_depth_1.stall") + num(m2[k], "value_depth_2_plus.stall"), num(m2[k], "stall")) for k in order]
    none = [ratio(num(m2[k], "none.stall"), num(m2[k], "stall")) for k in order]
    choice = column(m4, "relation_bits_given_origin_pc.online")
    origin_miss = column(m4, "residual_stall_whose_origin_missed_l1d")
    any3 = column(walker, "any.hops_3")
    out = {
        "n_windows": str(len(order)),
        "residual_range": span(residual, 0, True), "residual_median": median(residual, 0, True),
        "value_range": span(value, 0, True), "value_median": median(value, 0, True),
        "none_max": fmt(max(none), 0, True),
        "rel99_max": fmt(max(column(m5, "relations_for_99pct_of_stall"))),
        "per_start_max": fmt(max(v for v in (data.relations_per_start(k) for k in order) if v)),
        "any1_range": span(column(walker, "any.hops_1"), 0, True),
        "all_residual_range": span(column(walker, "share_of_all_residual_stall.any.hops_3"), 0, True),
        "gate_relation_range": span(column(m4, "instance_gate_bits_given_relation.online"), 2),
        "delta_range": span([ratio(num(m2[k], "same_pc_delta.stall"), num(m2[k], "stall")) for k in order], 0, True),
        "bitvec_max": fmt(max(ratio(num(m2[k], "value_through_bit_vector.stall"), num(m2[k], "stall")) or 0 for k in order), 0, True),
        "nearest_misses_p99_range": span(column(m7, "l1d_misses_between_nearest.p99")),
        "choice_range": span(choice, 2), "choice_median": median(choice, 2),
        "gate_all_range": span(column(m4, "instance_gate_bits_given_relation_producer_level_last_2.online"), 2),
        "gate_prior_range": span(column(m4, "instance_gate_bits_prior.online"), 2),
        "half_range": span(column(m3, "stall_share_slack_at_least_50pct_of_latency"), 1, True),
        "other_page_range": span(column(m8, "target_on_other_page_than_producer"), 0, True),
        "slack_range": span(column(m3, "slack_cycles.p50")),
        "latency_range": span(column(m3, "target_latency_cycles.p50")),
        "origin_miss_range": span(origin_miss, 0, True), "origin_miss_median": median(origin_miss, 0, True),
        "static_relations_range": span(column(m5, "static_relations_all_loads")),
        "origin_rate_range": span(column(m6, "loads_at_origin_pcs_per_kilocycle"), 1),
        "burst_range": span(column(m6, "fast.share_gaps_under_4_cycles"), 0, True),
        "nearest_p50_range": span(column(m7, "loads_between_nearest_producer_and_consumer.p50")),
        "nearest_p99_range": span(column(m7, "loads_between_nearest_producer_and_consumer.p99")),
        "page_range": span(column(m8, "target_page_among_last_64_miss_pages"), 0, True),
        "uops_range": span(column(m7, "micro_ops_on_path.p99")),
        "memory_share_range": span(column(m7, "through_memory_share"), 0, True),
    }
    if any3:
        out["any3_range"] = span(any3, 0, True)
        out["any3_median"] = median(any3, 0, True)
    return out


def fill(paths, data):
    """Replace ⟦name⟧ placeholders with the phrases from tokens()."""
    values = tokens(data)
    for path in paths:
        text = path.read_text()
        for name, phrase in values.items():
            text = text.replace(f"⟦{name}⟧", phrase)
        path.write_text(text)
        left = sorted(set(re.findall(r"⟦(\w+)⟧", text)))
        print(f"{path}: placeholders left: {left or 'none'}")


def splice(path, data):
    text = path.read_text()
    for name in Data.TABLES:
        pattern = re.compile(rf"(<!-- table:{name} -->\n).*?(<!-- /table:{name} -->)", re.S)
        if pattern.search(text):
            table = getattr(data, name)()
            text = pattern.sub(lambda m: m.group(1) + table + "\n" + m.group(2), text)
    path.write_text(text)
    print(f"spliced tables into {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="fast")
    parser.add_argument("--kind", default="gap")
    parser.add_argument("--splice", type=Path, default=None)
    parser.add_argument("--ranges", action="store_true")
    parser.add_argument("--tokens", action="store_true", help="print the phrases --fill would insert")
    parser.add_argument("--fill", type=Path, nargs="+", default=None)
    args = parser.parse_args()
    data = Data(args.kind, args.arm)
    if args.fill:
        fill(args.fill, data)
    elif args.tokens:
        for name, phrase in tokens(data).items():
            print(f"{name}: {phrase}")
    elif args.splice:
        splice(args.splice, data)
    elif args.ranges:
        for name, text in data.ranges():
            print(f"{name}: {text}")
    else:
        for name in Data.TABLES:
            print(f"### {name}\n\n{getattr(data, name)()}\n")


if __name__ == "__main__":
    main()
