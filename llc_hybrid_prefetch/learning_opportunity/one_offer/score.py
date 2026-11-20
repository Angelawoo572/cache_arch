"""Verify the exact pre-decision history and measure whole-run effects of each forecast."""

import numpy as np
from replay import BASE, HERE, OFFERS, RAW, STUDY, dump, read
import sys

sys.path.insert(0, str(STUDY / "capture"))
from access import IndexedStream
from score_helpers import turnover


def validate_prefix(folder, key, cue, baseline=BASE):
    base = IndexedStream(baseline, "arrival")
    arm = IndexedStream(folder, "arrival")
    start = 0
    for a, b in zip(base.chunks(), arm.chunks()):
        found = np.flatnonzero(a[:, 0] == key)
        stop = int(found[0]) + 1 if len(found) else len(a)
        if len(b) < stop or not np.array_equal(a[:stop], b[:stop]):
            raise ValueError("Baseline and treatment histories differ before the offer")
        if len(found):
            if int(a[stop - 1, 2]) != cue:
                raise ValueError("Physical cue differs from frozen forecast")
            return start + stop - 1, int(a[stop - 1, 3])
        start += len(a)
    raise ValueError("Missing forecast anchor")


def stages(directory, key):
    vocab = {int(r["id"]): r["value"] for r in read(directory / "symbols.csv")}
    matches = []
    for block in IndexedStream(directory, "offer").chunks():
        for _, _, request, line, cycle, code, consumer in block[
            block[:, 1] == key
        ].tolist():
            matches.append(
                dict(
                    stage=vocab[code >> 32],
                    reason=vocab[code & 0xFFFFFFFF],
                    line=line,
                    cycle=cycle,
                    consumer=consumer,
                    request=request,
                )
            )
    return matches


def measure(r, baseline=BASE, destination=RAW, case="bfs_discovery", policy="stride"):
    folder = destination / r["scenario"]
    key = int(r["instr_id"]) * 4 + int(r["operand"])
    cue, pred = int(r["cue_line"]), int(r["predicted_line"])
    rank, cycle = validate_prefix(folder, key, cue, baseline)
    base = IndexedStream(baseline, "arrival").read(rank + 8, rank + 9)[0]
    replay_eighth = IndexedStream(folder, "arrival").read(rank + 8, rank + 9)[0]
    target = int(base[0])
    hs, now = stages(baseline, key), stages(folder, key)
    offered = [x for x in now if x["stage"] == "proposed"]
    if len(offered) != 1 or offered[0]["line"] != pred:
        raise ValueError("Actual offer differs from published forecast")
    gains, losses, masks = turnover(case, folder, {(target >> 2, target & 3)}, policy)
    original, prior, treated = masks[target >> 2, target & 3]

    def stats(path):
        return {r["key"]: int(r["value"]) for r in read(path / "stats.csv")}

    a, b = stats(baseline), stats(folder)

    def cycles(directory):
        return int(next(read(directory / "cores.csv"))["cycles"])

    return dict(
        scenario=r["scenario"],
        anchor_rank=rank,
        anchor_cycle=cycle,
        cue_line=cue,
        predicted_line=pred,
        actual_eighth_line=int(base[2]),
        actual_eighth_lead_cycles=int(base[3]) - cycle,
        actual_eighth_operand_key=target,
        exact_eighth_target=int(pred == base[2]),
        replay_eighth_line=int(replay_eighth[2]),
        replay_eighth_lead_cycles=int(replay_eighth[3]) - cycle,
        replay_eighth_operand_key=int(replay_eighth[0]),
        offered_at=offered[0]["cycle"],
        dispatched_at=next((x["cycle"] for x in now if x["stage"] == "dispatch"), ""),
        filled_at=next((x["cycle"] for x in now if x["stage"] == "fill"), ""),
        first_use_at=next(
            (x["cycle"] for x in now if x["stage"] in ("late", "timely")), ""
        ),
        baseline_offer=next((x["line"] for x in hs if x["stage"] == "proposed"), ""),
        original_stages=" ".join(x["stage"] for x in hs),
        offer_stages=" ".join(x["stage"] for x in now),
        terminal_reason=now[-1]["reason"],
        mask_nopf=original,
        **{f"mask_{policy}": prior},
        mask_replay=treated,
        accepted=int(any(x["stage"] == "accepted" for x in now)),
        first_claimed_operand=next(
            (x["consumer"] for x in now if x["stage"] in ("late", "timely")), ""
        ),
        timely_gained=gains,
        timely_displaced=losses,
        cycles_saved=cycles(baseline) - cycles(folder),
        fixed_work_cycles=cycles(folder),
        change_dram_bytes=sum(
            b[k] - a[k]
            for k in ("dram_demand_bytes", "dram_prefetch_bytes", "dram_write_bytes")
        ),
    )


if __name__ == "__main__":
    dump(HERE / "effects.csv", [measure(r) for r in read(OFFERS)])
