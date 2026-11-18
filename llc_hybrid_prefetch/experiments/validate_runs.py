"""Validate complete run counts and fixed physical page assignment across policies, co-runners and warmup lengths."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from matrix import ROOT, run_specs
from run import checked_completion


def page_map(run):
    result = {}
    with gzip.open(str(run / "pages.csv.gz"), "rt") as source:
        for row in csv.DictReader(source):
            core, slot, vpn, ppn = (int(row[k]) for k in ("cpu", "page_slot","virtual_page","physical_page"))
            entry = result.setdefault(core, {})
            if vpn in entry or not ppn:
                raise RuntimeError("invalid page map at " + str(run))
            entry[vpn] = (slot, ppn)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "raw/runs")
    parser.add_argument("--out", type=Path, default=ROOT / "raw/page_audit.csv")
    args = parser.parse_args()
    reference, rows = {}, []
    for spec in run_specs():
        folder = args.runs / spec["run_id"]
        if not (folder / "complete.json").is_file() or not checked_completion(folder, spec):
            raise ValueError("incomplete run " + folder.name)
        saved = json.loads((folder / "run.json").read_text())
        if any(saved[k] != spec[k] for k in spec):
            raise ValueError("unexpected parameters: " + folder.name)
        for name in ("loads", "candidates", "head", "victims", "pages"):
            if not (folder / (name + ".csv.gz")).is_file():
                raise ValueError("missing " + name + ": " + folder.name)
        maps = page_map(folder)
        for cpu, trace in enumerate(spec["traces"]):
            observed = maps[cpu]
            expected = reference.setdefault(trace, (folder.name, observed))
            if observed != expected[1]:
                raise ValueError("physical mapping differs: {} CPU {} vs {}".format(folder.name, cpu, expected[0]))
            rows.append((folder.name,cpu,trace,next(iter(observed.values()))[0],len(observed),expected[0]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as sink:
        w=csv.writer(sink); w.writerow(("run_id","cpu","trace","slot","mapped_pages","reference_run")); w.writerows(rows)
    print("{} runs, {} active-core page maps matched.".format(len(list(run_specs())), len(rows)))
