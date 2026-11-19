"""Verify little-endian packing for every memory operand of two simultaneously active cores."""

import gzip
import struct
from collections import Counter
from run import RAW, HERE, run
from timing import gzrows

RECORD = struct.Struct("<QBB2B4B2Q4Q")


def main():
    fixtures = RAW / "operand_check"
    fixtures.mkdir(exist_ok=True, parents=True)
    traces = []
    warm, roi = 4000, 16000
    for core in range(2):
        p = fixtures / f"four_operands_{core}.champsimtrace"
        with p.open("wb") as f:
            for i in range(warm + roi):
                # Four distinct physical source lines on each loading instruction.
                memory = (
                    [0x20000000 + (op * 100000 + i // 8) * 64 for op in range(4)]
                    if i % 8 == 2
                    else [0] * 4
                )
                f.write(RECORD.pack(0x40000 + (i % 8) * 4, *([0] * 10), *memory))
        traces.append(p)
    folder = run("operand_parity", "spp", traces, [1, 0], warm, roi, sample=roi)
    words = [[0] * roi for c in range(2)]
    counts = Counter()
    for r in gzrows(folder / "loads.csv.gz"):
        if r["level"] == "LLC" and r["stage"] == "lookup":
            c, pos, op = int(r["cpu"]), int(r["instr_id"]) - warm - 1, int(r["operand"])
            words[c][pos] |= (1 if r["hit"] == "1" else 2) << (op * 4)
            counts[c, op] += 1
    for r in gzrows(folder / "candidates.csv.gz"):
        if r["stage"] in ("late", "timely"):
            c, pos, op = (
                int(r["demand_cpu"]),
                int(r["demand_instr_id"]) - warm - 1,
                int(r["demand_operand"]),
            )
            words[c][pos] |= (4 if r["stage"] == "late" else 8) << (op * 4)
    for c in range(2):
        actual = gzip.decompress((folder / f"outcomes_core{c}.u16le.gz").read_bytes())
        if actual != struct.pack("<" + str(roi) + "H", *words[c]):
            raise ValueError("Incorrect per-core, per-operand packing")
        if not all(counts[c, op] for op in range(4)):
            raise ValueError("Fixture did not exercise every source slot")
    (HERE / "operand_validation.txt").write_text(
        "Paired SPP, warmup 4,000 / ROI 16,000 instructions per active core.\n"
        "All eight core/operand combinations matched the complete LLC tag and prefetch-first-use CSVs.\n"
        + "\n".join(
            f"core {c} operand {op}: {n} tag lookups"
            for (c, op), n in sorted(counts.items())
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
