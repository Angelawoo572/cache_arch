"""A physically seeded far view must reproduce a forward-scanned full-source skip exactly."""

import struct
from run import BASE, RAW, HERE, run

RECORD = struct.Struct("<Q8x6Q")  # IP, reg/branch bytes, two destinations, four sources


def main():
    # Actual trace ABI has Q + 8 bytes of branch/register fields + six Q.
    trace = BASE / "raw/traces/605.mcf_s-994B.champsimtrace"
    origin, warm, roi = 300000, 50000, 200000
    work = RAW / "seed_check"
    work.mkdir(exist_ok=True, parents=True)
    seed = work / "ordered.u64le"
    sliced = work / "sliced.champsimtrace"
    pages = {}
    with trace.open("rb") as f:
        for rec in RECORD.iter_unpack(f.read(origin * 64)):
            for addr in rec:
                if addr:
                    pages[addr >> 12] = None
        sliced.write_bytes(f.read((warm + roi) * 64))
    with seed.open("wb") as f:
        for vpn in pages:
            f.write(struct.pack("<Q", vpn))
    a = run("placement_direct", "spp", [trace], [0], warm, roi, skip=origin, sample=roi)
    b = run(
        "placement_seeded",
        "spp",
        [sliced],
        [0],
        warm,
        roi,
        origin=origin,
        prefix_pages=seed,
        sample=roi,
    )
    keys = (
        "cores.csv",
        "stats.csv",
        "phases.csv",
        "loads.csv.gz",
        "candidates.csv.gz",
        "head.csv.gz",
        "pages.csv.gz",
        "outcomes_core0.u16le.gz",
    )
    for name in keys:
        if (a / name).read_bytes() != (b / name).read_bytes():
            raise ValueError("Far-view seed/ordinal equivalence failed: " + name)
    (HERE / "placement_validation.txt").write_text(
        f"Native SPP physical placement equivalence\nSource records skipped: {origin}\n"
        f"Distinct prefix VPNs: {len(pages)}\nWarmup: {warm}\nROI: {roi}\n"
        f"Identical outputs for direct source skip and an independently seeded materialization:\n"
        + "\n".join(keys)
        + "\n"
    )


if __name__ == "__main__":
    main()
