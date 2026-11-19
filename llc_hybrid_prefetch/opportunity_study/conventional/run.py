"""Run live matched conventional-policy comparisons on the shared A–C machine."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
BASE = STUDY.parent
RAW = STUDY / "raw/conventional"
sys.path.insert(0, str(BASE / "experiments"))
from matrix import run_specs


def comparison_cases():
    result = {}
    for r in run_specs():
        if r["group"] != "spec" or r["policy"] != "stride":
            continue
        key = r["run_id"].removesuffix("-stride")
        result[key] = dict(
            r,
            case=key,
            slots=([0, 1] if len(r["workloads"]) == 2 else [int(r["workloads"][0] in ("cactus", "xalan"))]),
            trace_dir=BASE / "raw/traces",
        )
    for r in json.loads((STUDY / "workloads/cases.json").read_text()):
        result[r["case"]] = dict(r, trace_dir=STUDY / "raw/traces")
    return result


def csvs(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def totals(folder):
    return {r["key"]: int(r["value"]) for r in csvs(folder / "stats.csv")}


def run(case, policy, label="open", limit=0, latency=0, caps=None, plan=None, detail="0", choices=False):
    name = f"{case['case']}__{policy}__{label}"
    folder = RAW / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "complete.json"
    expected = dict(
        case=case["case"],
        policy=policy,
        label=label,
        warmup=case["warmup"],
        roi=case["roi"],
        inputs=case["traces"],
        decision_delay=latency,
        epoch_issues=limit,
        per_core_acceptance_cap=caps or [],
    )
    if manifest.is_file():
        prior = json.loads(manifest.read_text())
        if any(prior.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing run uses different settings: " + str(folder))
        return folder
    n = len(case["traces"])
    env = dict(
        LLC_PF_POLICY="nopf" if policy == "nopf" else "stride",
        STUDY_POLICY="stride" if policy == "nopf" else policy,
        STUDY_EPOCH_LIMIT=str(limit),
        STUDY_LATENCY=str(latency),
        STUDY_FAST_NATIVE=str(int(label.startswith("async"))),
        STUDY_CHOICES=str(int(choices)),
        HYBRID_ACTIVE_CORES=str(n),
        HYBRID_COMPRESS="1",
        HYBRID_DETAIL=detail,
        HYBRID_OUT=str(folder),
        HYBRID_PAGE_SLOTS=",".join(map(str, case["slots"])),
        HYBRID_SKIPS=",".join([str(case["skip"])] * n),
    )
    if caps:
        env["STUDY_CAPS"] = ",".join(map(str, caps))
    if plan:
        env["HYBRID_PLAN"] = str(plan.resolve())
    args = [
        str(RAW / "build/pythia/bin/champsim"),
        f"--warmup_instructions={case['warmup']}",
        f"--simulation_instructions={case['roi']}",
        "-traces",
    ] + [str(case["trace_dir"] / (s + ".champsimtrace")) for s in case["traces"]]
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("STUDY_", "HYBRID_", "LLC_PF_"))}
    if not all(Path(x).exists() for x in args[4:]):
        raise ValueError("Missing trace " + name)
    start = time.monotonic()
    with (folder / "run.log").open("w") as out:
        subprocess.run(args, env=dict(environ, **env), stdout=out, stderr=subprocess.STDOUT, check=True)
    cores = csvs(folder / "cores.csv")
    if len(cores) != n or any(
        int(r["instructions"]) != case["roi"] or int(r["warmup"]) != case["warmup"] for r in cores
    ):
        raise ValueError("incomplete " + name)
    (folder / "complete.json").write_text(
        json.dumps(
            dict(
                case=case["case"],
                policy=policy,
                label=label,
                warmup=case["warmup"],
                roi=case["roi"],
                instructions=case["roi"] * n,
                per_core_acceptance_cap=caps or [],
                epoch_size=512,
                epoch_issues=limit,
                decision_delay=latency,
                wall_seconds=round(time.monotonic() - start, 2),
                inputs=case["traces"],
            ),
            indent=2,
        )
        + "\n"
    )
    print("completed", name, flush=True)
    return folder


def grid(group):
    real = ["mcf-cactus-r1", "mcf-r2", "lbm-xalan-r1"]
    new = ["gap_bfs3", "gap_bfs8", "mcf994_adjacent"]
    pfull = [
        "nopf",
        "stride",
        "look2",
        "look4",
        "multi",
        "boundary",
        "xpc",
        "temporal_sp",
        "temporal",
        "spp",
        "union",
        "confirm",
        "priority",
        "pressure",
        "confidence",
        "next",
    ]
    psyn = ["nopf", "stride", "look2", "multi", "boundary", "xpc", "temporal", "spp", "union"]
    if group == "followup":
        cases = (
            "row_hierarchy_paced",
            "streams_three_switch_paced",
            "streams_three_identified_paced",
            "xor_context_paced",
            "xor_maskfirst_paced",
            "row_previous_paced",
            "phase_flip_paced",
            "compiled_rows",
        )
        return [
            (c, p, "open", 0, 0)
            for c in cases
            for p in ("nopf", "stride", "look2", "multi", "boundary", "xpc", "temporal", "spp", "union")
        ]
    if group == "lategraph":
        return [
            (c, p, "open", 0, 0)
            for c in ("gap_bfs3_late", "gap_bfs8_late", "gap_bfs8_lbm")
            for p in ("nopf", "stride", "look2", "multi", "xpc", "temporal", "spp", "union", "priority")
        ]
    if group == "promotion":
        return [
            (c, "promote", "open", 0, 0)
            for c in real + ["gap_bfs8", "streams_onepc_paced", "crosspc_cue_paced"]
        ]
    if group == "open":
        return [(c, p, "open", 0, 0) for c in real for p in pfull]
    if group == "newreal":
        return [
            (c, p, "open", 0, 0)
            for c in new
            for p in ("nopf", "stride", "look2", "multi", "xpc", "temporal", "spp", "union", "priority")
        ]
    if group == "synthetic":
        return [
            (c, p, "open", 0, 0)
            for c in (
                "row_period2_paced",
                "row_period5_paced",
                "row_independent_paced",
                "streams_onepc_paced",
                "streams_pc_paced",
                "crosspc_cue_paced",
                "crosspc_mask_paced",
                "motif_noise_paced",
                "ramp_mlp4",
            )
            for p in psyn
        ]
    if group == "budget":
        return [
            (c, p, "paced8", 8, 0)
            for c in real
            for p in ("stride", "look2", "multi", "xpc", "temporal", "spp", "union", "confirm", "priority")
        ]
    if group == "async":
        return [
            (c, "union", f"async{lat}", 0, lat)
            for c in ("mcf-cactus-r1", "lbm-xalan-r1", "gap_bfs3_late")
            for lat in (8, 32)
        ]
    if group == "strong_timing":
        return [
            (c, p, f"lat{lat}", 0, lat)
            for c in ("mcf-cactus-r1", "lbm-xalan-r1", "gap_bfs3_late")
            for p in ("multi", "spp")
            for lat in (8, 32)
        ]
    if group == "timing":
        return [
            (c, p, f"lat{l}", 0, l)
            for c in ("mcf-cactus-r1", "lbm-xalan-r1", "crosspc_cue_paced", "gap_bfs3_late")
            for p in ("stride", "union", "priority")
            for l in (8, 32)
        ]
    raise ValueError(group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        choices=(
            "open",
            "newreal",
            "synthetic",
            "budget",
            "timing",
            "strong_timing",
            "async",
            "exact",
            "promotion",
            "followup",
            "lategraph",
        ),
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    spec = comparison_cases()
    if args.group == "exact":
        jobs = []
        for c in ("mcf-cactus-r1", "lbm-xalan-r1", "mcf-r2"):
            modes = ("stride", "look2", "multi", "spp", "union", "priority")
            caps = [
                min(
                    totals(RAW / "runs" / f"{c}__{p}__open").get("study_accepted_core_" + str(i), 0)
                    for p in modes
                )
                for i in range(len(spec[c]["traces"]))
            ]
            for p in modes:
                jobs.append((spec[c], p, "exact", 0, 0, caps))
    else:
        jobs = [(spec[c], p, label, limit, lat) for c, p, label, limit, lat in grid(args.group)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda j: run(*j), jobs))
