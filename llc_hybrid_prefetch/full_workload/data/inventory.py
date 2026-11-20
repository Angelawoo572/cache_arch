"""Index the retained trace inputs and completed native run folders."""

import csv
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
CACHE = STUDY.parents[1]
DATA = STUDY / "data"
TRACE_ROOT = STUDY / "raw/traces"
RUN_GROUPS = {
    "native": "raw/native/runs",
    "continuous ingress": "raw/continuous/runs",
    "short discovery": "raw/discovery/runs",
    "full discovery": "raw/discovery/full/runs",
    "short arithmetic": "raw/arithmetic/runs",
    "MCF arithmetic": "raw/mcf_arithmetic/runs",
    "single address": "raw/single_address",
    "independent Linux copy": "raw/linux_repeat",
}
RUN_FILES = (
    "complete.json",
    "run.log",
    "stdout.log",
    "run.json",
    "config.json",
    "cores.csv",
    "phases.csv",
    "stats.csv",
    "loads.csv.gz",
    "candidates.csv.gz",
    "head.csv.gz",
    "pages.csv.gz",
    "victims.csv.gz",
    "choices.csv.gz",
    "arrival.u64le.gz",
)


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return path.relative_to(CACHE).as_posix()


def trace_files():
    sources = {row["case"]: row for row in read_csv(STUDY / "intake/selection.csv")}
    files = []
    for view in read_csv(STUDY / "intake/views.csv"):
        name = view["view"]
        source = sources[view["case"]]
        archive = TRACE_ROOT / Path(source["manifest_path"]).name
        trace = TRACE_ROOT / f"{name}.champsimtrace"
        seed = TRACE_ROOT / f"{name}.prefix_vpns.u64le"
        start, count = int(view["start"]), int(view["count"])
        if archive.stat().st_size != int(source["compressed_bytes"]):
            raise ValueError(f"Wrong compressed archive size for {archive.name}")
        if trace.stat().st_size != 64 * count:
            raise ValueError(f"Wrong native view size for {name}")
        if start and not seed.is_file():
            raise ValueError(f"Missing physical-placement seed for {name}")
        files.append({
            "view": name,
            "role": view["role"],
            "source_url": source["url"],
            "source_records": source["verified_source_records"],
            "compressed_archive": relative(archive),
            "compressed_bytes": source["compressed_bytes"],
            "native_trace": relative(trace),
            "skipped_source_records": start,
            "native_records": count,
            "physical_placement_seed": relative(seed) if start else "",
            "validation": relative(STUDY / "intake" / f"{name}.validation.csv"),
        })
    return files


def run_files():
    files = []
    for group, path in RUN_GROUPS.items():
        pattern = "*/*/complete.json" if group == "short discovery" else "*/complete.json"
        for marker in sorted((STUDY / path).glob(pattern)):
            folder = marker.parent
            metadata = json.loads(marker.read_text())
            config = metadata.get("config", metadata)
            if group == "short discovery":
                case, policy = folder.parent.name, config["mode"]
                trace_paths = [
                    TRACE_ROOT / f"{item}.champsimtrace" for item in config["traces"]
                ]
            else:
                case, policy = config["case"], config["policy"]
                trace_paths = [CACHE / item for item in config["traces"]]
            if any(not trace.is_file() for trace in trace_paths):
                raise ValueError(f"Retained input missing for {marker.relative_to(CACHE)}")
            traces = [relative(trace) for trace in trace_paths]
            artifacts = [name for name in RUN_FILES if (folder / name).is_file()]
            artifacts += sorted(
                file.name for file in folder.glob("outcomes_core*.u16le.gz")
            )
            files.append({
                "group": group,
                "case": case,
                "policy": policy,
                "warmup_per_active_core": config["warmup"],
                "measured_per_active_core": config["roi"],
                "raw_directory": relative(folder),
                "trace_paths": "|".join(traces),
                "artifacts": "|".join(artifacts),
            })
    return files


def main():
    write_csv(DATA / "trace_files.csv", trace_files())
    write_csv(DATA / "run_files.csv", run_files())


if __name__ == "__main__":
    main()
