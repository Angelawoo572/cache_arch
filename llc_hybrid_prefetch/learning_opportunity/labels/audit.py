"""Check source operands, physical placement and outcome labels on every indexed stream."""

import argparse
import csv
import gzip
import json

import numpy as np

from build import CACHE, HERE
from sequence import ROOT, Sequence


def table(path):
    with gzip.open(path, "rt", newline="") as f:
        return {
            int(r["virtual_page"]): int(r["physical_page"]) for r in csv.DictReader(f)
        }


def check_case(folder):
    seq = Sequence(folder)
    source = CACHE / seq.meta["source"]
    native = json.loads((source / "complete.json").read_text())["config"]
    config_keys = ("traces", "slots", "warmup", "roi", "skip", "source_ordinal_base")
    signature = [native.get(k) for k in config_keys]
    mapping = table(source / "pages.csv.gz")
    samples = np.unique(np.linspace(0, seq.size - 1, min(65, seq.size), dtype=np.int64))
    obs = seq.observations(samples)
    keys = obs[:, 0]
    record = (keys >> 2) - native.get("source_ordinal_base", 0) - 1
    raw = np.memmap(CACHE / native["traces"][0], mode="r", dtype="u1").reshape(-1, 64)
    selected = raw[record]
    pc = selected[:, :8].copy().view("<u8").ravel()
    va = selected[:, 32:].copy().view("<u8")[np.arange(len(keys)), keys & 3]
    if not np.all(va):
        raise ValueError("Sample joined to a non-load source operand")
    physical = (
        np.array([mapping[int(addr >> np.uint64(12))] for addr in va], dtype="<u8") << 6
    ) | ((va >> 6) & 63)
    np.testing.assert_array_equal(pc, obs[:, 1])
    np.testing.assert_array_equal(physical, obs[:, 2])
    # Check every complete reference map before matching retired outcomes from other arms.
    for ref in seq.meta["outcomes"]:
        ref = CACHE / ref
        config = json.loads((ref / "complete.json").read_text())["config"]
        if signature != [config.get(k) for k in config_keys]:
            raise ValueError(
                f"Outcome reference uses a different dynamic program: {ref}"
            )
        if mapping != table(ref / "pages.csv.gz"):
            raise ValueError(f"Computed physical mappings differ across arms: {ref}")
    # Validate sampled mask bytes directly in the uncompressed retirement position.
    expected = seq.outcomes(samples)
    for col, ref in enumerate(seq.meta["outcomes"]):
        with gzip.open(CACHE / ref / "outcomes_core0.u16le.gz", "rb") as f:
            for j in np.argsort(keys):
                key = int(keys[j])
                f.seek(2 * ((key >> 2) - seq.meta["first_ordinal"]))
                word = int.from_bytes(f.read(2), "little")
                if (word >> (4 * (key & 3))) & 15 != int(expected[j, col]):
                    raise ValueError("Retrospective labels disagree with native source")
    return {
        "case": seq.meta["case"],
        "policy": seq.meta["policy"],
        "arrivals": seq.size,
        "measured_instructions": seq.meta["last_ordinal"]
        - seq.meta["first_ordinal"]
        + 1,
        "source_operand_checks": len(samples),
        "matched_reference_maps": 4,
        "dynamic_label_checks": 4 * len(samples),
        "event_chunks": len(seq.meta["chunks"]),
        "index_bytes": sum(f.stat().st_size for f in folder.iterdir()),
        "source_run": seq.meta["source"],
        "indexed_dataset": str(folder.relative_to(CACHE)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checked immutable dataset rows while capturing new cases",
    )
    args = parser.parse_args()
    path = HERE / "datasets.csv"
    previous = {}
    if args.resume and path.is_file():
        with path.open() as f:
            previous = {(r["case"], r["policy"]): r for r in csv.DictReader(f)}
    out = []
    for stamp in sorted(ROOT.glob("*/index.json")):
        meta = json.loads(stamp.read_text())
        key = (meta["case"], meta["policy"])
        if key not in previous:
            previous[key] = check_case(stamp.parent)
            print(*key, flush=True)
        out.append(previous[key])
    with path.with_suffix(".csv.tmp").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0], lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    path.with_suffix(".csv.tmp").replace(path)


if __name__ == "__main__":
    main()
