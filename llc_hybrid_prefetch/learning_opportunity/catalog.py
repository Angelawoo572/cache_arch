"""Publish a path from every measured policy to its exact input, native data and labels."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1]
FULL = HERE.parent / "full_workload"
POLICIES = ("nopf", "stride", "multi", "spp")


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def rel(path):
    return str(path.relative_to(CACHE))


def register():
    original = {
        r["view"]: r
        for r in rows(FULL / "data/trace_files.csv")
        if r["view"] != "pr_heldout_entire"
    }
    cases = list(original)
    ext = {r["case"]: r for r in rows(HERE / "expansion/sources.csv")}
    domains = {r["case"]: r for r in rows(HERE / "domains/sources.csv")}
    cases += (
        list(ext)
        + list(domains)
        + ["gcc734_slot1", "cc5_slot1", "snap_stanford", "snap_google"]
    )
    if len(cases) != 26 or len(set(cases)) != 26:
        raise ValueError(f"Expected 26 distinct views, found {cases}")
    for case in cases:
        stem = case.removesuffix("_slot1")
        if case in original:
            r = original[case]
            recorded = int(r["source_records"])
            url, archive, validation = (
                r["source_url"],
                r["compressed_archive"],
                r["validation"],
            )
            certified = HERE / "raw/capture/runs"
            tier = "full cache paths"
        elif stem in ext or stem in domains:
            group = "expansion" if stem in ext else "domains"
            r = (ext if stem in ext else domains)[stem]
            check = HERE / group / "validation" / (stem + ".validation.csv")
            v = rows(check)[0]
            assert int(v["compressed_bytes"]) == int(r["bytes"])
            recorded = int(v["source_records"])
            if int(v["materialized_records"]) != 250_001_024 or recorded < 250_001_024:
                raise ValueError(f"Incomplete decoded recording for {stem}")
            url = r["url"]
            archive = rel(HERE / "raw/expansion/traces" / r["source"])
            validation = rel(check)
            certified = HERE / "raw/expansion/runs"
            tier = "LLC continuous; lower paths sampled"
        else:
            check = HERE / "graph_inputs" / (case + ".csv")
            v = rows(check)[0]
            recorded = int(v["recorded_instructions"])
            url = v["graph_url"]
            archive = rel(HERE / "raw/graph_inputs" / url.rsplit("/", 1)[-1])
            validation = rel(check)
            certified = HERE / "raw/expansion/runs"
            tier = "LLC continuous; lower paths sampled"
        for arm in POLICIES:
            run = certified / (case + "__" + arm)
            meta = json.loads((run / "complete.json").read_text())["config"]
            label = HERE / "raw/labels" / (case + "__" + arm)
            idx = json.loads((label / "index.json").read_text())
            ingress = sum(int(r["llc_ingress"]) for r in rows(run / "phases.csv"))
            assert meta["policy"] == arm and meta["roi"] == int(
                rows(run / "cores.csv")[0]["instructions"]
            )
            assert idx["observations"] == ingress and all(
                (run / k).is_file()
                for k in ("outcomes_core0.u16le.gz", "arrival.u64le.gz")
            )
            rich = HERE / "raw/capture/runs" / (case + "__" + arm)
            detailed = case in original or (case in ext and arm == "stride")
            if detailed and not (rich / "capture_validation.json").is_file():
                raise ValueError(f"Missing full-path validation for {case}/{arm}")
            yield dict(
                case=case,
                policy=arm,
                capture="full cache paths" if detailed else tier,
                source_url=url,
                source_archive=archive,
                full_source_instructions=recorded,
                warmup=meta["warmup"],
                measured=meta["roi"],
                skipped_source=meta.get("source_ordinal_base", 0) + meta.get("skip", 0),
                physical_slot=meta["slots"][0],
                llc_ingress=ingress,
                native_input=meta["traces"][0],
                source_validation=validation,
                raw_run=rel(run),
                rich_run=rel(rich) if detailed else "",
                label_source_run=idx["source"],
                aligned_index=rel(label / "index.json"),
            )


if __name__ == "__main__":
    data = list(register())
    with (HERE / "data_registry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)
