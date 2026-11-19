"""Attribute LLC hit-to-miss reversals to the last observed eviction in native policy replays."""

import sys
from concurrent.futures import ThreadPoolExecutor

from analyze import save
from run import BASE, RAW, comparison_cases, run

sys.path.insert(0, str(BASE / "sim"))
from pollution import calculate


def probe(job):
    case, policy = job
    replay = run(comparison_cases()[case], policy, "victim_probe", detail="victims")
    before = BASE / "raw/qualified" / f"{case}-nopf"
    open_arm = RAW / "runs" / f"{case}__{policy}__open"
    for row in calculate(before, open_arm, replay):
        row["case"] = case
        row["policy"] = policy
        row["nopf_hit_policy_miss"] = row.pop("nopf_hit_stride_miss")
        yield row


if __name__ == "__main__":
    jobs = [("mcf-cactus-r1", p) for p in ("multi", "spp", "union")] + [
        ("lbm-xalan-r1", p) for p in ("multi", "union")
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        data = list(pool.map(lambda j: list(probe(j)), jobs))
    save("pollution.csv", [r for result in data for r in result])
