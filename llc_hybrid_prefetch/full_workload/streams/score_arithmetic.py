"""Score generation and displacement with a common dynamic-load denominator."""

import csv
import gzip
import json
from pathlib import Path

from replay import AUX, CASES, NATIVE, RAW

HERE = Path(__file__).resolve().parent


def mask(folder):
    data = gzip.decompress((folder / "outcomes_core0.u16le.gz").read_bytes())
    if len(data) != 2_000_000:
        raise ValueError("Expected exactly one million measured instructions")
    return int.from_bytes(data, "little")


def main():
    out = []
    positions = int.from_bytes(b"\x11" * 2_000_000, "little")
    for case in CASES:
        arms = {a: RAW / "runs" / (case + "__" + a) for a in (*NATIVE, *AUX)}
        runs = {
            a: json.loads((p / "complete.json").read_text()) for a, p in arms.items()
        }
        for a, r in runs.items():
            expected = runs["nopf"]["config"]
            for k in ("traces", "warmup", "roi", "skip", "slots"):
                if expected[k] != r["config"][k]:
                    raise ValueError(f"Unmatched {case} {a}: {k}")
        masks = {a: mask(p) for a, p in arms.items()}
        orig = (masks["nopf"] >> 1) & positions
        timely = {a: (m >> 3) & orig for a, m in masks.items()}
        for a, folder in arms.items():
            with (folder / "stats.csv").open() as f:
                st = {r["key"]: int(r["value"]) for r in csv.DictReader(f)}
            cycles = int(runs[a]["core_results"][0]["cycles"])
            row = dict(
                case=case,
                policy=a,
                roi=1_000_000,
                cycles=cycles,
                ipc=1_000_000 / cycles,
                original_misses=orig.bit_count(),
                direct_timely=timely[a].bit_count(),
                grants=st.get("study_arithmetic_grants", 0),
                agrees=st.get("study_arithmetic_agrees", 0),
                displaced=st.get("study_arithmetic_displaces", 0),
                new_offer=st.get("study_arithmetic_extra", 0),
                accepted=st["prefetch_accepted"],
                dram_pf_bytes=st["dram_prefetch_bytes"],
            )
            for c in ("stride", "multi", "spp", "plus1", "plus3", "local"):
                row["gained_vs_" + c] = (timely[a] & ~timely[c]).bit_count()
                row["lost_vs_" + c] = (timely[c] & ~timely[a]).bit_count()
            out.append(row)
    with (HERE / "arithmetic_replay.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=out[0], lineterminator="\n")
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
