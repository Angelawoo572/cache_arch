"""Exercise previously observed cross-page successors using a recurring conflict-set traversal."""

import random
import sys

from analyze import save
from run import BASE, RAW, csvs, run, totals

sys.path.insert(0, str(BASE / "opportunity_study/workloads"))
from generate_controls import ins


def fixture():
    nlines = 19
    warm_pages = 64 * (nlines - 1) + 1
    warm_pattern = 19 * 16
    count = 19 * 96
    period = 196
    folder = RAW / "traces"
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / "region_conflict_cycle.champsimtrace"
    pattern = list(range(nlines))
    random.Random(1907).shuffle(pattern)
    pc = 0x400000
    prefix = ins(pc, d=1, s=1)
    suffix = (
        ins(pc + 8, d=1, s=14)
        + b"".join(ins(pc + 12 + 4 * j, d=1, s=1) for j in range(period - 4))
        + ins(pc + 0xD00, d=26, s=25, t=26, br=1)
    )
    with path.open("wb") as f:
        for page in [*range(warm_pages)] + [
            64 * pattern[n % nlines] for n in range(warm_pattern + count + 64)
        ]:
            f.write(prefix + ins(pc + 0xE00, d=14, s=1, va=0x20000000 + 4096 * page) + suffix)
    spec = dict(
        case="region_conflict_cycle",
        traces=["region_conflict_cycle"],
        slots=[0],
        skip=0,
        trace_dir=folder,
        warmup=period * (warm_pages + warm_pattern),
        roi=period * count,
    )
    data = []
    for p in ("nopf", "stride", "temporal_sp", "temporal"):
        dest = run(spec, p, detail="1")
        st = totals(dest)
        data.append(
            dict(
                case=spec["case"],
                policy=p,
                cycles=csvs(dest / "cores.csv")[0]["cycles"],
                degree=1,
                accepted=st["prefetch_accepted"],
                crosspage_accepted=st.get("study_accepted_crosspage", 0),
                rejected_page=st.get("study_rejected_page", 0),
                timely=st.get("study_pf_timely", 0),
                late=st.get("study_pf_late", 0),
                dram_bytes=sum(st["dram_" + k + "_bytes"] for k in ("demand", "prefetch", "write")),
            )
        )
    save("temporal_fixture.csv", data)
    assert (
        data[2]["rejected_page"]
        and not data[2]["accepted"]
        and data[3]["crosspage_accepted"]
        and data[3]["timely"]
    )


if __name__ == "__main__":
    fixture()
