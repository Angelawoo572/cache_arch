"""Count temporal associations between a lost native Stride hit and the new policy's last LLC eviction."""

import importlib.util
from concurrent.futures import ProcessPoolExecutor

from run import BASE, HERE, MIXES, RUNS
from score import save

path = BASE / "sim/pollution.py"
spec = importlib.util.spec_from_file_location("native_victims", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def measure(job):
    case, policy = job
    base = RUNS / (case + "__stride")
    probe = RUNS / (case + "__" + policy)
    result = []
    for r in module.calculate(base, probe, probe):
        result.append(
            dict(
                case=case,
                policy=policy,
                core=r["cpu"],
                stride_hit_policy_miss=r["nopf_hit_stride_miss"],
                prefetch_last_evictor=r["last_evictor_prefetch"],
                distinct_events=r["distinct_prefetch_evictions"],
                same_core=r["same_core"],
                other_core=r["other_core"],
                other_last_evictor=r["last_evictor_other"],
                no_recorded_change=r["no_line_change_observed"],
                last_change_refill=r["last_change_refill"],
            )
        )
    return result


if __name__ == "__main__":
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(measure, [(case, p) for case in MIXES for p in ("look2", "paced", "turn")]))
    save(HERE / "pollution.csv", [r for group in rows for r in group])
