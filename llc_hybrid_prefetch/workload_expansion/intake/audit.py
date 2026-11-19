"""Inspect both measured windows with the reader's exact memory and register slot layout."""
import argparse
import csv

import numpy as np

from fetch import RECORDS
from fetch_long import RECORDS as LONG_RECORDS
from register import HERE, STUDY, write_csv

DTYPE = np.dtype([("pc", "<u8"), ("branch", "u1"), ("taken", "u1"),
                  ("dst_reg", "u1", 2), ("src_reg", "u1", 4),
                  ("dst_mem", "<u8", 2), ("src_mem", "<u8", 4)])
WINDOWS = {"first": (100000, 350000), "later": (550000, 800000)}
assert DTYPE.itemsize == 64 and [x[1] for x in DTYPE.fields.values()] == [0, 8, 9, 10, 12, 16, 32]


def inputs(long_google):
    with (HERE / "registry.csv").open() as f:
        registry = list(csv.DictReader(f))
    discovery = []
    folder = STUDY / ("raw/intake/long" if long_google else "raw/intake/traces")
    for row in registry:
        if row["split"] == "heldout":
            assert not row["prefix_path"]
            continue
        if not long_google or row["collection"] == "gtrace_v2":
            discovery.append((row, folder / (row["case"] + ".champsimtrace")))
    assert {p for _, p in discovery} == set(folder.glob("*.champsimtrace"))
    if long_google:
        return discovery, {"deep": (3_000_000, 5_000_000)}, LONG_RECORDS, "long_"
    return discovery, WINDOWS, RECORDS, ""


def audit(long_google=False):
    discovery, windows, length, prefix = inputs(long_google)
    result = []
    for row, path in discovery:
        z = np.memmap(path, dtype=DTYPE, mode="r")
        assert len(z) == length
        assert z["pc"].min() > 0 and z["branch"].max() <= 1 and z["taken"].max() <= 1
        assert not ((z["taken"] == 1) & (z["branch"] == 0)).any()
        record = dict(case=row["case"], collection=row["collection"])
        for name, (low, high) in windows.items():
            a = z[low:high]
            load = (a["src_mem"] != 0); store = (a["dst_mem"] != 0)
            sr = (a["src_reg"] != 0); dr = (a["dst_reg"] != 0)
            no_branch = a["branch"] == 0
            loaded = load.any(axis=1)
            writers = dr.any(axis=1)
            sources = np.unique(a["src_reg"][sr])
            dests = np.unique(a["dst_reg"][dr])
            selected = dict(**record, window=name, records=len(a), pc_count=len(np.unique(a["pc"])),
                branch_count=int(a["branch"].sum()), taken_count=int(a["taken"].sum()),
                load_instructions=int(loaded.sum()), load_operands=int(load.sum()), store_operands=int(store.sum()),
                multi_load_instructions=int((load.sum(axis=1)>1).sum()), load_pcs=len(np.unique(a["pc"][loaded])),
                nonbranch_instructions=int(no_branch.sum()), nonbranch_with_source_regs=int(sr.any(axis=1)[no_branch].sum()),
                nonbranch_with_destination_regs=int(writers[no_branch].sum()), load_with_destination_regs=int(writers[loaded].sum()),
                distinct_nonzero_source_regs=len(sources), distinct_nonzero_dest_regs=len(dests),
                shared_source_and_dest_reg_ids=len(np.intersect1d(sources, dests)))
            for field in ("src_mem", "dst_mem", "src_reg", "dst_reg"):
                for slot in range(a[field].shape[1]):
                    selected[f"{field}_slot{slot}_nonzero"] = int(np.count_nonzero(a[field][:,slot]))
            result.append(selected)
    write_csv(prefix + "format_audit.csv", result)
    print(f"Validated {len(discovery)} input prefixes, {len(result)} input windows, {len(discovery)*length:,} complete records.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--long-google", action="store_true")
    audit(p.parse_args().long_google)
