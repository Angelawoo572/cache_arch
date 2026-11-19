"""Compare prefetch deadlines at breaks in real, attributed timely-hit runs."""

import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OPPORTUNITY = PROJECT / "opportunity_study"
PAIRS = (
    ("gap_bfs8", 0, "gap_bfs8_lbm", 0),
    ("lbm_peer", 0, "gap_bfs8_lbm", 1),
    ("mcf-r1", 0, "mcf-cactus-r1", 0),
    ("lbm-r1", 0, "lbm-xalan-r1", 0),
)
PROJECTIONS = ("core", "pc", "page_4k", "region_64k")


@dataclass(frozen=True)
class Request:
    producer: tuple
    target: int
    proposed: int
    dispatched: int
    filled: int
    consumed: int
    first_use: str


def folder(case):
    for root in (OPPORTUNITY / "raw/runs", PROJECT / "raw/runs"):
        path = root / (case + "-stride")
        if (path / "complete.json").is_file():
            return path
    raise FileNotFoundError(case)


def claims(path, cpu):
    pending = {}
    consumers = {}
    with gzip.open(path / "candidates.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if int(r["cpu"]) != cpu:
                continue
            event, pid, cycle = r["stage"], int(r["id"]), int(r["cycle"])
            if event == "proposed":
                pending[pid] = dict(
                    producer=(int(r["instr_id"]), int(r["operand"])),
                    target=int(r["target"]),
                    proposed=cycle,
                )
            if pid not in pending:
                continue
            item = pending[pid]
            if event in ("dispatch", "fill"):
                item[{"dispatch": "dispatched", "fill": "filled"}[event]] = cycle
            if event in ("timely", "late") and int(r["demand_cpu"]) == cpu:
                key = int(r["demand_instr_id"]), int(r["demand_operand"])
                if key in consumers:
                    raise ValueError("two first uses for one demand")
                item.update(consumed=cycle, first_use=event)
                consumers[key] = item
    return {
        k: Request(**v)
        for k, v in consumers.items()
        if "filled" in v and "dispatched" in v
    }


class Run:
    def __init__(self, case, core):
        path = folder(case)
        self.case, self.core = case, core
        self.meta = json.loads((path / "run.json").read_text())
        self.claims = claims(path, core)
        with np.load(OPPORTUNITY / "raw/history" / f"{case}-c{core}.npz") as f:
            self.z = {
                k: f[k]
                for k in (
                    "instr",
                    "op",
                    "pc",
                    "pa",
                    "va",
                    "t",
                    "tag_t",
                    "hit",
                    "baseline",
                    "pf_kind",
                    "head",
                )
            }
        z = self.z
        self.keys = list(zip(z["instr"].tolist(), z["op"].tolist()))
        self.index = {k: i for i, k in enumerate(self.keys)}
        if len(self.index) != len(self.keys):
            raise ValueError("duplicate source load")
        self.timely = (z["baseline"] == 0) & (z["hit"] == 1) & (z["pf_kind"] == 1)
        self.late = (z["baseline"] == 0) & (z["hit"] == 0) & (z["pf_kind"] == 2)

    def project(self, name):
        if name == "core":
            return np.zeros(len(self.keys), dtype=int)
        if name == "pc":
            return self.z["pc"]
        return self.z["pa"] // (64 if name == "page_4k" else 1024)


def episodes(run, projection):
    streak = Counter()
    last = {}
    for i, group in enumerate(run.project(projection).tolist()):
        if run.timely[i]:
            streak[group] += 1
            last[group] = i
        else:
            if run.late[i] and streak[group] >= 2:
                yield last[group], i, streak[group]
            streak[group] = 0


def lead_metrics(arm, i, transaction):
    return dict(
        issue_to_ingress=int(arm.z["t"][i]) - transaction.proposed,
        issue_to_consumer=transaction.consumed - transaction.proposed,
        issue_to_dispatch=transaction.dispatched - transaction.proposed,
        dispatch_to_fill=transaction.filled - transaction.dispatched,
        slack=transaction.consumed - transaction.filled,
    )


def evaluate(arm, ref, projection):
    counts = Counter()
    values = defaultdict(list)
    comparisons = []
    examples = []
    meta = dict(
        late_arm=arm.case,
        core=arm.core,
        other_arm=ref.case,
        other_core=ref.core,
        projection=projection,
    )
    for prev, i, streak in episodes(arm, projection):
        counts["episodes"] += 1
        k = arm.keys[i]
        req = arm.claims.get(k)
        if req is None:
            continue
        if req.first_use != "late" or req.target != arm.z["pa"][i]:
            raise ValueError("late join mismatch")
        counts["complete"] += 1
        tag = int(arm.z["tag_t"][prev])
        if tag < 0:
            raise ValueError("timely label without lookup")
        due = req.consumed - (req.filled - req.proposed)
        counts["already_proposed_before_last_hit_lookup"] += int(req.proposed < tag)
        counts["latest_issue_before_last_hit_lookup"] += int(due < tag)
        counts["last_hit_lookup_not_yet_visible_at_target_ingress"] += int(
            tag >= arm.z["t"][i]
        )
        values["cycles_last_lookup_to_issue"].append(req.proposed - tag)
        values["cycles_last_lookup_to_estimated_latest_issue"].append(due - tag)
        values["remaining_to_fill_at_merge"].append(req.filled - req.consumed)
        source = ref.index.get(k)
        if source is None:
            continue
        if any(arm.z[a][i] != ref.z[a][source] for a in ("pc", "pa", "va")):
            raise ValueError("layout/identity mismatch")
        counts["target_ingress_in_both"] += 1
        if not ref.timely[source]:
            continue
        counts["same_target_timely_in_other"] += 1
        o = ref.claims.get(k)
        if o is None or o.producer != req.producer or o.target != req.target:
            continue
        counts["same_request_timely_in_other"] += 1
        x, y = lead_metrics(arm, i, req), lead_metrics(ref, source, o)
        for v in (x, y):
            if (
                v["slack"]
                != v["issue_to_consumer"]
                - v["issue_to_dispatch"]
                - v["dispatch_to_fill"]
            ):
                raise ValueError("pipeline balance")
        comparisons.append((x, y))
        if len(examples) < 2:
            examples.append(
                dict(
                    **meta,
                    timely_streak=streak,
                    last_hit_instr=arm.keys[prev][0],
                    last_hit_operand=arm.keys[prev][1],
                    last_hit_pc=int(arm.z["pc"][prev]),
                    last_hit_lookup=tag,
                    target_instr=k[0],
                    target_operand=k[1],
                    physical_line=req.target,
                    target_pc=int(arm.z["pc"][i]),
                    producer_instr=req.producer[0],
                    producer_operand=req.producer[1],
                    late_arm_proposed=req.proposed,
                    late_arm_dispatch=req.dispatched,
                    late_arm_ingress=int(arm.z["t"][i]),
                    late_arm_first_merge=req.consumed,
                    late_arm_fill=req.filled,
                    estimated_latest_issue=due,
                    late_arm_head=int(arm.z["head"][i]),
                    other_proposed=o.proposed,
                    other_dispatched=o.dispatched,
                    other_ingress=int(ref.z["t"][source]),
                    other_first_hit=o.consumed,
                    other_fill=o.filled,
                    other_head=int(ref.z["head"][source]),
                )
            )
    columns = (
        "episodes",
        "complete",
        "already_proposed_before_last_hit_lookup",
        "latest_issue_before_last_hit_lookup",
        "last_hit_lookup_not_yet_visible_at_target_ingress",
        "target_ingress_in_both",
        "same_target_timely_in_other",
        "same_request_timely_in_other",
    )
    metrics = (
        "cycles_last_lookup_to_issue",
        "cycles_last_lookup_to_estimated_latest_issue",
        "remaining_to_fill_at_merge",
    )
    result = dict(
        **meta,
        **{k: counts[k] for k in columns},
        **{"median_" + k: (median(values[k]) if values[k] else "") for k in metrics},
    )
    rows = []
    for label, side in (("late_arm", 0), ("timely_other_arm", 1)):
        rows.append(
            dict(
                **meta,
                role=label,
                n=len(comparisons),
                **{
                    "median_"
                    + k: (
                        median([p[side][k] for p in comparisons]) if comparisons else ""
                    )
                    for k in (
                        "issue_to_ingress",
                        "issue_to_consumer",
                        "issue_to_dispatch",
                        "dispatch_to_fill",
                        "slack",
                    )
                },
            )
        )
    if (
        result["same_request_timely_in_other"] > result["same_target_timely_in_other"]
        or result["complete"] > result["episodes"]
    ):
        raise AssertionError("invalid cohort counts")
    return result, rows, examples


def save(name, rows):
    with (HERE / name).open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def main():
    summary = []
    dist = []
    witness = []
    for a, core, b, peer in PAIRS:
        solo, mix = Run(a, core), Run(b, peer)
        if solo.meta["traces"][core] != mix.meta["traces"][peer] or any(
            solo.meta[f] != mix.meta[f] for f in ("warmup", "roi", "skip")
        ):
            raise ValueError("different traces or measured work")
        for run, ref in ((mix, solo), (solo, mix)):
            for projection in PROJECTIONS:
                result, rows, examples = evaluate(run, ref, projection)
                summary.append(result)
                dist.extend(rows)
                witness.extend(examples)
    save("last_hit_deadlines.csv", summary)
    save("matched_service.csv", dist)
    save("examples.csv", witness)
    print(
        f"Compared {len(PAIRS)} fixed-work pairs on {len(PROJECTIONS)} projections in both directions"
    )


if __name__ == "__main__":
    main()
