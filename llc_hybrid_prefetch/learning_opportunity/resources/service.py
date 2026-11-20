"""Join sampled baseline misses to the lifecycle of the physically matching conventional offer."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path[:0] = [str(STUDY / "capture"), str(STUDY / "labels")]
from access import IndexedStream
from sequence import Sequence

CASES = ("mcf17_discovery", "bfs_discovery", "soplex06_heldout")
POLICIES = ("stride", "multi", "spp")
N = 8192


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def selected(ids, block):
    if not len(ids):
        return np.zeros(len(block), dtype=bool)
    pos = np.searchsorted(ids, block)
    return (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == block)


def summarize(case, policy):
    folder = STUDY / "raw/capture/runs" / f"{case}__{policy}"
    sequence = Sequence(STUDY / "raw/labels" / folder.name)
    # Select population quantiles before examining address, outcome or service.
    rank = (np.arange(N, dtype="int64") * sequence.size + sequence.size // 2) // N
    anchors = sequence.observations(rank)
    masks = sequence.outcomes(rank)
    which = ("nopf", *POLICIES).index(policy)
    sub = (masks[:, 0] == 2) & np.isin(masks[:, which], [6, 9])
    ranks = rank[sub]
    targets = anchors[sub]
    expected = masks[sub, which]
    keys = np.sort(targets[:, 0])
    lookup = {int(r[0]): i for i, r in enumerate(targets)}
    if len(lookup) != len(targets):
        raise ValueError("Duplicate sampled target operand")
    symbols = {int(r["id"]): r["value"] for r in read(folder / "symbols.csv")}
    codes = {v: k for k, v in symbols.items()}
    offers = IndexedStream(folder, "offer")
    request = {}
    for block in offers.chunks():
        kind = block[:, 5] >> 32
        selected_claims = np.isin(
            kind, [codes.get("late", -1), codes.get("timely", -1)]
        ) & selected(keys, block[:, 6])
        for _, anchor, identifier, line, cycle, code, consumer in block[
            selected_claims
        ].tolist():
            i = lookup[consumer]
            status = symbols[code >> 32]
            if (
                line != int(targets[i, 2])
                or (6 if status == "late" else 9) != expected[i]
            ):
                raise ValueError("Native claim and aligned target do not match")
            request[identifier] = (i, anchor, status)
    if len(request) != len(targets):
        raise ValueError(
            f"{case}/{policy} missing first-use claims: {len(request)} of {len(targets)}"
        )
    picked = np.array(sorted(request), dtype="uint64")
    stages = defaultdict(dict)
    for block in offers.chunks():
        for _, _, identifier, _, cycle, code, _ in block[
            selected(picked, block[:, 2])
        ].tolist():
            stages[identifier][symbols[code >> 32]] = cycle
    # Fetch only the relevant 64-event windows, one contiguous block at a time.
    earliest = {}
    for chunk in np.unique(ranks // sequence.chunk):
        on = np.flatnonzero(ranks // sequence.chunk == chunk)
        start = max(0, int(min(ranks[on])) - 64)
        end = int(max(ranks[on])) + 1
        events = sequence.observations(np.arange(start, end))
        for i in on:
            stop = int(ranks[i]) - start
            prior = events[max(0, stop - 64) : stop]
            match = np.flatnonzero(prior[:, 2] >> 6 == targets[i, 2] >> np.uint64(6))
            if len(match):
                k = int(match[0])
                earliest[i] = (int(prior[k, 3]), len(prior) - k)
    details = []
    for identifier, (i, anchor, status) in sorted(
        request.items(), key=lambda it: it[1][0]
    ):
        times = stages[identifier]
        issue = times["accepted"]
        admission = int(targets[i, 3])
        fill = times.get("fill")
        e, rank_gap = earliest.get(i, (None, None))
        details.append(
            dict(
                case=case,
                policy=policy,
                claim=status,
                target_key=int(targets[i, 0]),
                target_line=int(targets[i, 2]),
                admission_cycle=admission,
                anchor_key=anchor,
                request_id=identifier,
                proposed=times["proposed"],
                accepted=issue,
                dispatched=times.get("dispatch"),
                filled=fill,
                claim_cycle=times[status],
                early_legal_cycle=e,
                early_legal_event_gap=rank_gap,
                proposal_to_admission=admission - times["proposed"],
                queue_cycles=times["dispatch"] - issue if "dispatch" in times else None,
                dispatch_to_fill=fill - times["dispatch"]
                if fill is not None and "dispatch" in times
                else None,
                early_lead=admission - e if e is not None else None,
                early_slack_at_observed_service=admission - e - (fill - issue)
                if e is not None and fill is not None
                else None,
            )
        )
    result = []
    for status in ("timely", "late"):
        group = [r for r in details if r["claim"] == status]
        rec = dict(
            case=case,
            policy=policy,
            claim=status,
            sampled_admissions=len(rank),
            original_first_uses=len(group),
            same_page_witness=sum(r["early_lead"] is not None for r in group),
            witness_before_proposal=sum(
                r["early_legal_cycle"] is not None
                and r["early_legal_cycle"] < r["proposed"]
                for r in group
            ),
            comparable_service=sum(
                r["early_slack_at_observed_service"] is not None for r in group
            ),
        )
        for gap in (0, 8, 32):
            rec[f"earlier_slack_ge{gap}"] = sum(
                r["early_slack_at_observed_service"] is not None
                and r["early_slack_at_observed_service"] >= gap
                for r in group
            )
        for field in (
            "proposal_to_admission",
            "queue_cycles",
            "dispatch_to_fill",
            "early_lead",
        ):
            col = [r[field] for r in group if r[field] is not None]
            for q in (0.10, 0.50, 0.90):
                rec[f"{field}_p{int(q * 100)}"] = (
                    int(np.quantile(col, q, method="nearest")) if col else ""
                )
        result.append(rec)
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-only", action="store_true")
    args = parser.parse_args()
    out = STUDY / "raw/service"
    out.mkdir(exist_ok=True)
    aggregate = []
    examples = []
    for case in CASES:
        for policy in POLICIES:
            dest = out / f"{case}__{policy}.csv"
            metrics = out / f"{case}__{policy}_summary.csv"
            if not metrics.exists():
                pair = f"{case}__{policy}"
                if args.complete_only and not all(
                    p.exists()
                    for p in (
                        STUDY / "raw/labels" / pair / "index.json",
                        STUDY / "raw/capture/runs" / pair / "capture_validation.json",
                    )
                ):
                    continue
                counts, records = summarize(case, policy)
                if records:
                    save(dest, records)
                save(metrics, counts)
            aggregate.extend(read(metrics))
            if dest.exists():
                details = read(dest)
                for status in ("timely", "late"):
                    examples.extend([r for r in details if r["claim"] == status][:1])
    if aggregate:
        save(HERE / "request_service.csv", aggregate)
    if examples:
        save(HERE / "request_examples.csv", examples)


if __name__ == "__main__":
    main()
