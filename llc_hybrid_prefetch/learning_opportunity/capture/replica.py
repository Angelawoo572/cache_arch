"""Compare independently validated native captures across hosts by row block and operand mask."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from access import COLUMNS, IndexedStream


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def shutdown_inventory(folder, blocks, dictionary, statistics):
    count = int(statistics.get("study_pf_unresolved", 0))
    if not count:
        return {"rows": 0, "cycle": None}
    offers = IndexedStream(folder, "offer")
    start = offers.rows - count
    tail = offers.read(start, offers.rows)
    stage = int(dictionary["unresolved"])
    if (
        np.any((tail[:, 5] >> 32) != stage)
        or np.any(tail[:, 6] != 0)
        or np.any(np.diff(tail[:, 0]) != 1)
        or len(np.unique(tail[:, 4])) != 1
        or len(np.unique(tail[:, 2])) != count
    ):
        raise ValueError(
            "Expected a contiguous post-run inventory at a single shutdown cycle"
        )
    # hybrid_finish enumerates pending entries in hash-table iteration order.
    # Preserve the live prefix, then compare every (request ID, trigger, target,
    # shutdown cycle, stage/reason, consumer) tuple in order of its unique ID.
    canonical = tail[np.argsort(tail[:, 2], kind="stable")].copy()
    canonical[:, 0] = tail[:, 0]
    for i, (first, size, _, _) in enumerate(offers.members):
        end = first + size
        if end <= start:
            continue
        cut = max(start, first)
        block = offers.read(first, end)
        block[cut - first :] = canonical[cut - start : end - start]
        blocks[i] = (size, zlib.crc32(block.tobytes()))
    return {"rows": count, "cycle": int(tail[0, 4])}


def evidence(folder):
    marker = json.loads((folder / "capture_validation.json").read_text())
    if marker["baseline_counters_and_masks"] != "match":
        raise ValueError("Each replica needs a completed native parity audit")
    blocks = {}
    logical_bytes = 0
    for label in COLUMNS:
        stream = IndexedStream(folder, label)
        members = []
        with stream.path.open("rb") as file:
            for _, n, offset, size in stream.members:
                # The parity audit has already inflated and CRC-checked each member.
                file.seek(offset + size - 8)
                crc, raw_size = struct.unpack("<II", file.read(8))
                if raw_size != n * stream.width * 8:
                    raise ValueError(f"Member size mismatch: {label}")
                members.append((n, crc))
                logical_bytes += raw_size
        blocks[label] = members
    masks = []
    with gzip.open(folder / "outcomes_core0.u16le.gz", "rb") as file:
        while data := file.read(1 << 22):
            masks.append((len(data), zlib.crc32(data)))
    if sum(size for size, _ in masks) != 2 * marker["measured"]:
        raise ValueError("Wrong number of retirement outcomes")
    dictionary = sorted(rows(folder / "symbols.csv"), key=lambda r: r["value"])
    stats = rows(folder / "stats.csv")
    inventory = shutdown_inventory(
        folder,
        blocks["offer"],
        {r["value"]: r["id"] for r in dictionary},
        {r["key"]: r["value"] for r in stats},
    )
    return {
        "schema": 2,
        "postrun_unresolved": inventory,
        "config": json.loads((folder / "complete.json").read_text())["config"],
        "dictionary": dictionary,
        "cores": rows(folder / "cores.csv"),
        "phases": rows(folder / "phases.csv"),
        "statistics": stats,
        "members": blocks,
        "logical_rows": sum(sum(n for n, _ in members) for members in blocks.values()),
        "logical_bytes": logical_bytes,
        "outcomes": masks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--compare", type=Path, help="manifest from an audited second host"
    )
    args = parser.parse_args()
    output = args.folder / "replica_blocks.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(evidence(args.folder), separators=(",", ":")) + "\n"
    )
    temporary.replace(output)
    if args.compare:
        a, b = (json.loads(p.read_text()) for p in (output, args.compare))
        for key in a:
            if a[key] != b[key]:
                raise ValueError(f"Replica mismatch in {key}: {args.folder.name}")
        print(
            args.folder.name,
            a["logical_rows"],
            "verified rows",
            a["config"]["roi"],
            "instruction outcome words",
        )


if __name__ == "__main__":
    main()
