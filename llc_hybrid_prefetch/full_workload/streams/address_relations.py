"""Compare local, shared-increment and interleaved-continuation address arithmetic."""

import argparse
from collections import Counter, deque
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
sys.path.insert(0, str(FULL / "discovery"))
from streams import events, keys, label

FAMILIES = ("local", "shared_increment", "other_continuation")
OFFSETS = (1, -1, 2, -2, 4, -4)
FIXED = tuple(f"fixed_{delta:+d}" for delta in OFFSETS)
MODES = FAMILIES + FIXED


def analyze(
    z, strong, missed_all_four, start, stop, case, quarter, threshold, target_rows
):
    ix = [i for i, k in enumerate(z["instr"]) if start <= k <= stop]
    t = [int(z["t"][i]) for i in ix]
    pc = [int(z["pc"][i]) for i in ix]
    line = [int(z["pa"][i]) for i in ix]
    key = [(int(z["instr"][i]), int(z["op"][i])) for i in ix]
    orig = [int(z["baseline"][i]) == 0 for i in ix]
    stats = {m: Counter() for m in MODES}
    hits = {m: {} for m in MODES}
    seen = set()
    slots = {}
    recent = deque(maxlen=8)
    # Discard 16 ingresses on the left for scoring, while allowing causal state to warm.
    for i in range(len(ix) - 16):
        p, x = pc[i], line[i]
        slot = ((p >> 2) ^ (p >> 8)) & 63
        tag, old, _ = slots.get(slot, (None, 0, None))
        delta = x - old if tag == p else None
        slots[slot] = (p, x, delta)
        other = next((q for q in reversed(recent) if q != p), None)
        recent.append(p)
        prior, y, d = slots.get(
            (((other or 0) >> 2) ^ ((other or 0) >> 8)) & 63, (None, 0, None)
        )
        seen.add(x)
        actions = [
            (x, delta),
            (x, d if other is not None and prior == other else None),
            (y, d if other is not None and prior == other else None),
        ]
        actions.extend((x, offset) for offset in OFFSETS)
        if i < 16:
            continue
        first_future = {line[j]: j for j in range(i + 16, i, -1)}
        for mode, (base, stride) in zip(MODES, actions):
            s = stats[mode]
            s["anchors"] += 1
            if not stride:
                continue
            target = base + stride
            if target == x or target < 0 or (target >> 6) != (base >> 6):
                continue
            s["legal_proposals"] += 1
            j = first_future.get(target)
            if j is None or not orig[j]:
                continue
            lead = t[j] - t[i]
            s["matched_original_miss_offers"] += 1
            if lead < threshold:
                continue
            s["lead_qualified_offers"] += 1
            if j not in hits[mode]:
                # First matching causal anchor has the largest available lead.
                hits[mode][j] = (i, lead, target not in seen, other, base, stride)
    rows, examples = [], []
    for mode in MODES:
        s = stats[mode]
        matches = hits[mode]
        leads = sorted(v[1] for v in matches.values())
        s["unique_targets"] = len(matches)
        s["unseen_in_island"] = sum(v[2] for v in matches.values())
        s["anchor_and_target_different_pc"] = sum(
            pc[v[0]] != pc[j] for j, v in matches.items()
        )
        s["absent_from_local"] = sum(j not in hits["local"] for j in matches)
        s["tag_miss_in_all_four"] = sum(key[j] in missed_all_four for j in matches)
        s["uncovered_by_four"] = sum(
            all(key[j] not in a for a in strong.values()) for j in matches
        )
        distinct = [
            j
            for j in matches
            if j not in hits["local"] and all(key[j] not in a for a in strong.values())
        ]
        beyond_fixed = [j for j in distinct if not any(j in hits[k] for k in FIXED)]
        s["also_beyond_six_fixed_offsets"] = len(beyond_fixed)
        displacements = Counter(matches[j][5] for j in beyond_fixed)
        if mode == "shared_increment" and case.startswith("mcf17_"):
            for j in beyond_fixed:
                i, lead, _, other, base, delta = matches[j]
                target_rows.append(
                    dict(
                        case=case,
                        quarter=quarter,
                        target_instr=key[j][0],
                        target_op=key[j][1],
                        target_line=line[j],
                        cue_instr=key[i][0],
                        cue_op=key[i][1],
                        cue_line=base,
                        other_pc=other,
                        increment=delta,
                        lead=lead,
                        intervening_ingresses=j - i - 1,
                    )
                )
        s["beyond_local_and_four"] = len(distinct)
        s["beyond_local_and_four_unseen"] = sum(matches[j][2] for j in distinct)
        rows.append(
            dict(
                case=case,
                quarter=quarter,
                mode=mode,
                **{
                    k: s[k]
                    for k in (
                        "anchors",
                        "legal_proposals",
                        "matched_original_miss_offers",
                        "lead_qualified_offers",
                        "unique_targets",
                        "unseen_in_island",
                        "anchor_and_target_different_pc",
                        "absent_from_local",
                        "uncovered_by_four",
                        "tag_miss_in_all_four",
                        "beyond_local_and_four",
                        "beyond_local_and_four_unseen",
                        "also_beyond_six_fixed_offsets",
                    )
                },
                survivor_increments=json.dumps(
                    dict(sorted(displacements.items())), separators=(",", ":")
                ),
                median_lead=leads[(len(leads) - 1) // 2] if leads else "",
            )
        )
        if distinct and mode in FAMILIES:
            j = (beyond_fixed or distinct)[0]
            i, lead, unseen, other, base, stride = matches[j]
            examples.append(
                dict(
                    case=case,
                    quarter=quarter,
                    mode=mode,
                    anchor_instr=key[i][0],
                    anchor_operand=key[i][1],
                    anchor_pc=hex(pc[i]),
                    anchor_line=hex(line[i]),
                    prior_other_pc=hex(other) if other is not None else "",
                    beyond_fixed_offsets=int(j in beyond_fixed),
                    base=hex(base),
                    signed_increment=stride,
                    target_instr=key[j][0],
                    target_operand=key[j][1],
                    target_pc=hex(pc[j]),
                    target_line=hex(line[j]),
                    anchor_cycle=t[i],
                    demand_ingress_cycle=t[j],
                    lead_cycles=lead,
                    intervening_ingresses=j - i - 1,
                    unseen_in_island=int(unseen),
                )
            )
    return rows, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    with (FULL / "intake/selection.csv").open() as f:
        cases = args.cases or [r["case"] for r in csv.DictReader(f)] + [
            "bfs_discovery_far",
            "mcf17_discovery_far",
        ]
    threshold = json.loads((FULL / "discovery/service.json").read_text())["external"]
    rows, examples = [], []
    target_rows = []
    for case in cases:
        arms = {
            arm: events(case, arm, "full")
            for arm in ("nopf", "stride", "multi", "spp", "temporal")
        }
        z = label(arms["stride"], arms["nopf"])
        timely = {
            arm: {k for k, v in zip(keys(arm_data), arm_data["kind"]) if v == 1}
            for arm, arm_data in arms.items()
            if arm != "nopf"
        }
        missed_all_four = set.intersection(
            *(
                {k for k, v in zip(keys(data), data["hit"]) if v == 0}
                for arm, data in arms.items()
                if arm != "nopf"
            )
        )
        phases_path = FULL / "raw/native/runs" / (case + "__stride") / "phases.csv"
        with phases_path.open() as f:
            for ph in csv.DictReader(f):
                data, evidence = analyze(
                    z,
                    timely,
                    missed_all_four,
                    int(ph["first_ordinal"]),
                    int(ph["last_ordinal"]),
                    case,
                    ph["quarter"],
                    threshold,
                    target_rows,
                )
                rows += data
                examples += evidence
    for name, data in (
        ("arithmetic.csv", rows),
        ("arithmetic_examples.csv", examples),
        ("mcf_arithmetic_targets.csv", target_rows),
    ):
        if data:
            with (HERE / name).open("w") as f:
                w = csv.DictWriter(f, fieldnames=data[0], lineterminator="\n")
                w.writeheader()
                w.writerows(data)


if __name__ == "__main__":
    main()
