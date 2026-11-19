"""Compare service and decision deadlines on the BFS-8 core, alone and with LBM."""

import csv
import gzip
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "raw/runs"


def measure(case):
    requests = {}
    with gzip.open(BASE / (case + "-stride") / "candidates.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["cpu"] != "0":
                continue
            stage = r["stage"]
            if stage == "proposed":
                requests[r["id"]] = {stage: int(r["cycle"])}
            elif r["id"] in requests:
                requests[r["id"]][stage] = int(r["cycle"])
    for kind in ("timely", "late"):
        complete = [
            q
            for q in requests.values()
            if all(k in q for k in (kind, "dispatch", "fill"))
        ]
        yield dict(
            case=case,
            first_use=kind,
            complete_requests=len(complete),
            median_proposal_to_dispatch=median(
                q["dispatch"] - q["proposed"] for q in complete
            ),
            median_dispatch_to_fill=median(q["fill"] - q["dispatch"] for q in complete),
            median_proposal_to_first_demand_event=median(
                q[kind] - q["proposed"] for q in complete
            ),
            median_first_demand_event_minus_fill=median(
                q[kind] - q["fill"] for q in complete
            ),
        )


if __name__ == "__main__":
    out = [row for case in ("gap_bfs8", "gap_bfs8_lbm") for row in measure(case)]
    with (HERE / "bfs_service.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0], lineterminator="\n")
        w.writeheader()
        w.writerows(out)
