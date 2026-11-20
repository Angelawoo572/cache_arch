"""Index one complete LLC stream and join retired-operand outcomes across policies."""

import argparse
import csv
import errno
import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
BASE = STUDY.parent / "full_workload"
CACHE = STUDY.parents[1]
ARMS = ("nopf", "stride", "multi", "spp")
CHUNK = 65536


def sources(case, policy):
    name = f"{case}__{policy}"
    roots = [STUDY / "raw" / area / "runs" for area in ("capture", "expansion")]
    legacy = BASE / "raw/continuous/runs"
    run = next(
        (
            p / name
            for p in [legacy, *roots]
            if (p / name / "arrival.u64le.gz").is_file()
            and (p / name / "complete.json").is_file()
        ),
        None,
    )
    if run is None:
        raise FileNotFoundError(f"No completed continuous recording for {name}")
    # Choose a single reference experiment for all arms, even if some are still running.
    # External rich Stride replicas share the expansion stimulus and may finish first.
    reference_root = next(
        (
            folder
            for folder in (BASE / "raw/native/runs", roots[1], roots[0])
            if any((folder / f"{case}__{arm}").is_dir() for arm in ARMS)
        ),
        None,
    )
    if reference_root is None:
        raise FileNotFoundError(f"No outcome experiment for {case}")
    refs = [reference_root / f"{case}__{arm}" for arm in ARMS]
    for ref in refs:
        if not (ref / "complete.json").is_file():
            raise FileNotFoundError(f"Outcome reference is incomplete: {ref}")
    return run, refs


def quarters(folder):
    with (folder / "phases.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["cpu"] == "0" for r in rows)
    return rows


def key_values(events, key):
    if key == "pc":
        return events[:, 1]
    if key == "line":
        return events[:, 2]
    return events[:, 2] >> (6 if key == "page" else 12)


def write_postings(folder, events, key):
    values = key_values(events, key)
    keys, first, counts = np.unique(values, return_index=True, return_counts=True)
    np.save(folder / f"{key}_keys.npy", keys)
    np.save(folder / f"{key}_first.npy", first.astype("<u4"))
    if key != "line":
        ends = counts.cumsum(dtype=np.uint64)
        ranks = np.argsort(values, kind="stable").astype("<u4")
        np.save(folder / f"{key}_ends.npy", ends)
        np.save(folder / f"{key}_ranks.npy", ranks)


def read_into(stream, array):
    dest = memoryview(array).cast("B")
    offset = 0
    while offset < len(dest):
        count = stream.readinto(dest[offset : offset + 2**20])
        if not count:
            raise ValueError("Truncated compressed integer stream")
        offset += count


def scratch_array(folder, name, shape, dtype, memory):
    if memory:
        return np.empty(shape, dtype=dtype)
    return np.memmap(folder / name, dtype=dtype, mode="w+", shape=shape)


class ReferenceCache:
    """Retain four retirement arrays from at most one case, within a byte limit."""

    def __init__(self, limit_mib):
        if limit_mib < 0:
            raise ValueError("Reference cache bound must be nonnegative")
        self.limit = limit_mib * 2**20
        self.sources = ()
        self.arrays = {}

    def begin(self, refs, rows):
        sources = tuple(refs)
        if self.sources != sources:
            self.arrays.clear()
            self.sources = sources
        return self.arrays if 2 * rows * len(refs) <= self.limit else None


def build(case, policy, *, memory=False, mask_cache=None, output=None):
    if mask_cache is not None and not memory:
        raise ValueError("The reference cache requires memory-backed arrays")
    run, refs = sources(case, policy)
    out = (STUDY / "raw/labels" if output is None else output) / f"{case}__{policy}"
    if (out / "index.json").is_file():
        return out
    out.parent.mkdir(exist_ok=True, parents=True)
    phases = quarters(run)
    start = int(phases[0]["first_ordinal"])
    end = int(phases[-1]["last_ordinal"])
    num = sum(int(r["llc_ingress"]) for r in phases)
    if not 0 < num < 2**32:
        raise ValueError("Expected a nonempty stream addressable by uint32 event ranks")
    for ref in refs:
        other = quarters(ref)
        if [(r["first_ordinal"], r["last_ordinal"]) for r in other] != [
            (r["first_ordinal"], r["last_ordinal"]) for r in phases
        ]:
            raise ValueError("Incompatible retirement windows between policies")
    with TemporaryDirectory(prefix=out.name + ".", dir=out.parent) as tmp:
        work = Path(tmp)
        product = work / "indexed"
        product.mkdir()
        events = scratch_array(work, "events.bin", (num, 4), "<u8", memory)
        words = scratch_array(work, "labels.bin", num, "<u2", memory)
        words[:] = 0
        chunks = []
        with gzip.open(run / "arrival.u64le.gz", "rb") as stream:
            for lo in range(0, num, CHUNK):
                hi = min(num, lo + CHUNK)
                batch = events[lo:hi]
                read_into(stream, batch)
                cycle = batch[:, 3]
                if np.any(cycle[1:] < cycle[:-1]) or (
                    lo and events[lo - 1, 3] > cycle[0]
                ):
                    raise ValueError("Arrival order violates simulator time")
                ids = batch[:, 0] >> 2
                if ids.min() < start or ids.max() > end:
                    raise ValueError("Operand falls outside fixed-work interval")
                chunks.append((lo, hi, int(cycle[0]), int(cycle[-1])))
            if stream.read(1):
                raise ValueError("Unexpected extra LLC observations")
        retired = end - start + 1
        retained = mask_cache.begin(refs, retired) if mask_cache is not None else None
        for arm_index, ref in enumerate(refs):
            masks = retained.get(ref) if retained is not None else None
            if masks is None:
                masks = scratch_array(work, "reference.bin", retired, "<u2", memory)
                with gzip.open(ref / "outcomes_core0.u16le.gz", "rb") as stream:
                    read_into(stream, masks)
                    if stream.read(1):
                        raise ValueError("Unexpected extra retired-operand labels")
                if retained is not None:
                    retained[ref] = masks
            for lo, hi, _, _ in chunks:
                ids = events[lo:hi, 0]
                shifted = masks[(ids >> 2) - start] >> (4 * (ids & 3)).astype("<u2")
                codes = shifted & 15
                if not np.isin(codes, [0, 1, 2, 6, 9]).all():
                    raise ValueError("Unknown outcome code")
                words[lo:hi] |= codes << (4 * arm_index)
            del masks
            if not memory:
                (work / "reference.bin").unlink()
        for chunk_id, (lo, hi, _, _) in enumerate(chunks):
            batch = events[lo:hi]
            np.savez_compressed(
                product / f"events_{chunk_id:04d}.npz",
                idop=batch[:, 0],
                pc=batch[:, 1],
                line=batch[:, 2],
                cycle=batch[:, 3],
            )
            np.savez_compressed(
                product / f"outcomes_{chunk_id:04d}.npz", masks=words[lo:hi]
            )
        for key in ("pc", "line", "page", "region"):
            write_postings(product, events, key)
        meta = {
            "case": case,
            "policy": policy,
            "observations": num,
            "first_ordinal": start,
            "last_ordinal": end,
            "chunk_rows": CHUNK,
            "chunks": chunks,
            "source": str(run.relative_to(CACHE)),
            "outcomes": [str(ref.relative_to(CACHE)) for ref in refs],
            "arms": ARMS,
        }
        (product / "index.json").write_text(json.dumps(meta, indent=1) + "\n")
        try:
            os.rename(product, out)
        except OSError as error:
            if (
                error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                or not (out / "index.json").is_file()
            ):
                raise
            existing = json.loads((out / "index.json").read_text())
            proposed = json.loads((product / "index.json").read_text())
            if existing != proposed:
                raise ValueError(
                    f"Concurrent builders selected different sources for {out.name}"
                ) from error
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--policy", default="stride", choices=ARMS)
    parser.add_argument(
        "--memory", action="store_true", help="Use RAM for transient arrays"
    )
    args = parser.parse_args()
    for case in args.cases:
        print(build(case, args.policy, memory=args.memory), flush=True)
