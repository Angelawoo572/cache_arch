"""Match the preidentified offline opportunities to exact native first uses and producers."""

import csv
import gzip
from collections import Counter, defaultdict
from run_mcf import CASES, FULL, HERE, POLICIES, WARMUP, native
from score import ARMS, folder, write_table


def operand_states(case, arm, targets):
    origin = int(
        native.rows(FULL / "intake" / (case + ".validation.csv"))[0][
            "materialized_start"
        ]
    )
    result = {}
    with gzip.open(folder(case, arm) / "outcomes_core0.u16le.gz", "rb") as f:
        last = -1
        for inst, slot in sorted(targets):
            if inst != last:
                offset = (inst - origin - WARMUP - 1) * 2
                if offset < 0:
                    raise ValueError("Targets outside the measured ROI")
                f.seek(offset)
                item = f.read(2)
                if len(item) != 2:
                    raise ValueError("Target after EOF")
                word = int.from_bytes(item, "little")
                last = inst
            result[inst, slot] = (word >> (int(slot) * 4)) & 15
    return result


def provenance(case, arm, examples):
    target_keys = {(int(x["target_instr"]), int(x["target_op"])) for x in examples}
    cue_keys = {(int(x["cue_instr"]), int(x["cue_op"])) for x in examples}
    uses, cues, accepted = {}, {}, {}
    relevant_lines = {int(x["target_line"]) for x in examples}
    lifecycle = defaultdict(dict)
    ingress = {}
    with gzip.open(folder(case, arm) / "loads.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            if (
                row["stage"] == "ingress"
                and row["level"] == "LLC"
                and row["cpu"] == "0"
            ):
                key = (int(row["instr_id"]), int(row["operand"]))
                if key in target_keys:
                    ingress[key] = int(row["cycle"])
    with gzip.open(folder(case, arm) / "candidates.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            cue = (int(r["instr_id"]), int(r["operand"]))
            stage = r["stage"]
            target = int(r["target"])
            ident = int(r["id"])
            if target in relevant_lines and stage in ("proposed", "dispatch", "fill"):
                lifecycle[ident][stage] = int(r["cycle"])
            if stage == "accepted":
                accepted[ident] = r["reason"]
            if cue in cue_keys and stage in ("accepted", "rejected", "silent"):
                cues[cue] = (target, stage, r["reason"])
            if stage in ("timely", "late"):
                key = (int(r["demand_instr_id"]), int(r["demand_operand"]))
                if key in target_keys:
                    if key in uses:
                        raise ValueError("Multiple first uses on the same demand")
                    uses[key] = (
                        target,
                        stage,
                        accepted.get(ident, "unlogged_producer"),
                        cue,
                        ident,
                        int(r["pc"]),
                        int(r["base"]),
                    )
    return uses, cues, lifecycle, ingress


def analyze(case, examples):
    keys = {(int(x["target_instr"]), int(x["target_op"])) for x in examples}
    if len(keys) != len(examples):
        raise ValueError("Duplicate original target")
    states = {a: operand_states(case, a, keys) for a in ARMS}
    records = []
    summaries = []
    for arm in POLICIES:
        uses, cues, lifecycle, ingress = provenance(case, arm, examples)
        totals = Counter()
        for x in examples:
            key = (int(x["target_instr"]), int(x["target_op"]))
            anchor = (int(x["cue_instr"]), int(x["cue_op"]))
            line = int(x["target_line"])
            if not states["nopf"][key] & 2 or any(
                states[a][key] & 8 for a in ("stride", "multi", "spp", "temporal")
            ):
                raise ValueError(
                    f"{case} {key}: target outside frozen unmatched NoPF set"
                )
            bits = states[arm][key]
            stage = (
                "timely"
                if bits & 8
                else ("late" if bits & 4 else "no_attributed_first_use")
            )
            actual = uses.get(key)
            if bool(actual) != (stage != "no_attributed_first_use"):
                raise ValueError(
                    f"{case}/{arm} {key}: detailed provenance and retirement masks disagree"
                )
            source = ""
            actual_cue = ("", "")
            history = {}
            actual_pc = ""
            actual_base = ""
            if actual:
                (
                    actual_line,
                    actual_stage,
                    source,
                    actual_cue,
                    ident,
                    actual_pc,
                    actual_base,
                ) = actual
                history = lifecycle[ident]
                if (actual_line, actual_stage) != (line, stage):
                    raise ValueError(
                        "First use maps to a different physical line or stage"
                    )
            issue = cues.get(anchor)
            if issue:
                offered, outcome, reason = issue
                cue_result = (
                    ("same_line_" + outcome)
                    if offered == line
                    else ("no_offer" if outcome == "silent" else "different_line")
                )
            else:
                cue_result = "no_logged_decision"
                reason = ""
            totals[stage] += 1
            totals[(stage, source)] += 1
            totals[cue_result] += 1
            records.append(
                dict(
                    case=case,
                    arm=arm,
                    quarter=x["quarter"],
                    target_instr=key[0],
                    target_op=key[1],
                    target_line=line,
                    offline_increment=x["increment"],
                    offline_cue_instr=anchor[0],
                    offline_cue_op=anchor[1],
                    offline_lead_cycles=x["lead"],
                    actual_mask=bits,
                    actual_use=stage,
                    actual_producer=source,
                    actual_cue_instr=actual_cue[0],
                    actual_cue_op=actual_cue[1],
                    actual_cue_pc=actual_pc,
                    actual_cue_line=actual_base,
                    actual_generate_cycle=history.get("proposed", ""),
                    actual_dispatch_cycle=history.get("dispatch", ""),
                    actual_fill_cycle=history.get("fill", ""),
                    actual_target_ingress_cycle=ingress.get(key, ""),
                    outcome_at_original_cue=cue_result,
                    reason_at_original_cue=reason,
                )
            )
        summaries.append(
            dict(
                case=case,
                arm=arm,
                original_sampled_targets=len(keys),
                timely=totals["timely"],
                late=totals["late"],
                no_attributed_first_use=totals["no_attributed_first_use"],
                timely_arithmetic=totals["timely", "arithmetic"],
                timely_stride=totals["timely", "stride"],
                timely_unlogged_producer=totals["timely", "unlogged_producer"],
                same_line_accepted_at_original_cue=totals["same_line_accepted"],
                same_line_rejected_at_original_cue=totals["same_line_rejected"],
                different_line_at_original_cue=totals["different_line"],
                native_silent_at_original_cue=totals["no_offer"],
                no_logged_decision_at_original_cue=totals["no_logged_decision"],
            )
        )
    return summaries, records


if __name__ == "__main__":
    samples = defaultdict(list)
    for r in native.rows(HERE.parent / "mcf_arithmetic_targets.csv"):
        samples[r["case"]].append(r)
    out, records = [], []
    for case in CASES:
        summary, items = analyze(case, samples[case])
        out.extend(summary)
        records.extend(items)
    write_table("target_outcomes.csv", out)
    write_table("target_details.csv", records)
    print(
        f"Matched {len(records) // len(POLICIES)} frozen target operands under both live alternatives."
    )
