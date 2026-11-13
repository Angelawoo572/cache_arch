#!/usr/bin/env python3
"""Run matched LLC policies in the patched ChampSim-Pythia."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


POLICIES = ("nopf", "next_line", "stride", "streamer", "ampm", "spp")
RUNS = Path(__file__).resolve().parent / "runs"
SETTING = re.compile(r"^([A-Za-z0-9_]+) (\S+)$")
WARMUP = re.compile(r"^Warmup complete CPU\s+(\d+) instructions:\s+(\d+) cycles:\s+(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="directory under conventional/runs")
    parser.add_argument("--warmup", type=int, default=200000, help="warmup instructions per core")
    parser.add_argument("--instructions", type=int, default=1000000, help="ROI instructions per core")
    parser.add_argument("--skip", type=int, default=0, help="initial trace records to skip per core")
    parser.add_argument("--degree", type=int, default=1, help="maximum issued requests per demand event")
    parser.add_argument("--policy", choices=POLICIES, action="append", dest="policies",
                        help="run only this policy; repeat to select several")
    parser.add_argument("--events", action="store_true", help="save demand lookup events per policy")
    parser.add_argument("traces", nargs="+", type=Path, help="one gzip or xz trace per configured core")
    args = parser.parse_args()
    args.policies = args.policies or POLICIES

    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    args.traces = [trace.resolve() for trace in args.traces]
    if RUNS.resolve() not in args.output.parents:
        parser.error("--output must be a directory inside {}".format(RUNS))
    if not args.binary.is_file() or not os.access(str(args.binary), os.X_OK):
        parser.error("--binary must be an executable file")
    for trace in args.traces:
        if not trace.is_file() or not os.access(str(trace), os.R_OK) or trace.suffix not in (".gz", ".xz"):
            parser.error("trace must be a readable .gz or .xz file: {}".format(trace))
    if args.warmup < 0 or args.skip < 0 or args.instructions < 1:
        parser.error("warmup and skip must be nonnegative; instructions must be positive")
    if not 1 <= args.degree <= 64:
        parser.error("--degree must be between 1 and 64")
    if len(args.policies) != len(set(args.policies)):
        parser.error("each policy can appear only once")
    return args


def file_identity(path):
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def write_json(path, value):
    staging = path.with_suffix(".tmp")
    with staging.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    staging.replace(path)


def read_json(path):
    with path.open() as source:
        return json.load(source)


def trace_links(traces, directory):
    # Pythia sends these names to a shell and uses the name to seed address translation.
    names = []
    for core, trace in enumerate(traces):
        name = trace.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", name):
            name = "workload-{}.champsimtrace{}".format(core, trace.suffix)
        alias = Path("traces") / "core{}".format(core) / name
        (directory / alias.parent).mkdir(parents=True)
        (directory / alias).symlink_to(trace)
        names.append(str(alias))
    return names


def read_log(path):
    values = {}
    warmup = {}
    saw_completion = False
    with path.open(errors="replace") as log:
        for raw_line in log:
            line = raw_line.strip()
            if line == "ChampSim completed all CPUs":
                saw_completion = True
            if "Repeating trace:" in line:
                raise ValueError("the trace wrapped before all cores finished")
            warmup_match = WARMUP.match(line)
            if warmup_match:
                core, instructions, cycles = map(int, warmup_match.groups())
                warmup[core] = (instructions, cycles)
            match = SETTING.fullmatch(line)
            if match:
                key, value = match.groups()
                values[key] = value
    if not saw_completion:
        raise ValueError("ChampSim did not finish all cores")
    return values, warmup


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def optional_count(counters, key):
    value = counters.get(key)
    return None if value is None else int(value)


def parse_result(path, args, policy, aliases, wall_seconds):
    values, warmup = read_log(path)
    counters = {key[len("LLC_PF_"):]: value for key, value in values.items() if key.startswith("LLC_PF_")}
    expected = {
        "num_cpus": len(aliases),
        "warmup_instructions": args.warmup,
        "simulation_instructions": args.instructions,
        "LLC_PF_cores": len(aliases),
        "LLC_PF_policy": policy,
        "LLC_PF_degree": args.degree,
        "LLC_PF_skip": args.skip,
    }
    expected.update({"trace_{}".format(core): alias for core, alias in enumerate(aliases)})
    for key, value in expected.items():
        if values.get(key) != str(value):
            raise ValueError("expected {} {}; got {}".format(key, value, values.get(key)))

    cores = []
    for core in range(len(aliases)):
        counter_prefix = "core_{}_".format(core)
        instructions = int(values["Core_{}_instructions".format(core)])
        cycles = int(values["Core_{}_cycles".format(core)])
        warmup_instructions, warmup_cycles = warmup[core]
        if instructions < args.instructions or cycles <= 0:
            raise ValueError("core {} did not complete its ROI".format(core))
        ingress = int(counters[counter_prefix + "ingress"])
        hits = int(counters[counter_prefix + "hit"])
        misses = int(counters[counter_prefix + "miss"])
        accesses = hits + misses
        if accesses > ingress:
            raise ValueError("core {} has more LLC demand lookups than ingress events".format(core))
        cores.append({
            "core": core,
            "instructions": instructions,
            "cycles": cycles,
            "warmup_retired_instructions": warmup_instructions,
            "warmup_cycles": warmup_cycles,
            "ipc": instructions / cycles,
            "llc_demand_ingress": ingress,
            "llc_rq_forwarded": optional_count(counters, counter_prefix + "rq_forwarded"),
            "llc_rq_merged": optional_count(counters, counter_prefix + "rq_merged"),
            "llc_rq_full": optional_count(counters, counter_prefix + "rq_full"),
            "llc_demand_accesses": accesses,
            "llc_demand_hits": hits,
            "llc_demand_misses": misses,
            "llc_hit_rate": ratio(hits, accesses),
            "llc_miss_rate": ratio(misses, accesses),
            "llc_mpki": 1000 * misses / instructions,
        })

    frequency = int(values["cpu_freq"])
    cycles = max(core["cycles"] for core in cores)
    instructions = sum(core["instructions"] for core in cores)
    hits = sum(core["llc_demand_hits"] for core in cores)
    misses = sum(core["llc_demand_misses"] for core in cores)
    issued = int(counters["prefetch_issued"])
    timely = int(counters["prefetch_timely"])
    late = int(counters["prefetch_late"])
    unused = int(counters["prefetch_unused"])
    redundant = int(counters["prefetch_redundant"])
    unresolved = int(counters["prefetch_unresolved"])
    if timely + late + unused + redundant + unresolved != issued:
        raise ValueError("LLC prefetch outcomes do not account for every issued request")
    if policy == "nopf" and issued:
        raise ValueError("NoPF issued LLC prefetch requests")
    demand_bytes = int(counters["dram_demand_read_bytes"])
    prefetch_bytes = int(counters["dram_prefetch_read_bytes"])
    write_bytes = int(counters["dram_write_bytes"])
    dram_bytes = demand_bytes + prefetch_bytes + write_bytes
    slack_sum = optional_count(counters, "prefetch_slack_cycles_sum")
    accesses = hits + misses
    return {
        "policy": policy,
        "trace_seed": int(values["champsim_seed"]),
        "cpu_mhz": frequency,
        "cores": cores,
        "instructions": instructions,
        "llc_demand_ingress": sum(core["llc_demand_ingress"] for core in cores),
        "llc_demand_accesses": accesses,
        "llc_demand_hits": hits,
        "llc_demand_misses": misses,
        "llc_hit_rate": ratio(hits, accesses),
        "llc_miss_rate": ratio(misses, accesses),
        "llc_mpki": 1000 * misses / instructions,
        "prefetch_candidates": int(counters["prefetch_candidates"]),
        "prefetch_issued": issued,
        "prefetch_fills": int(counters["prefetch_fills"]),
        "prefetch_timely": timely,
        "prefetch_late": late,
        "prefetch_unused": unused,
        "prefetch_redundant": redundant,
        "prefetch_unresolved": unresolved,
        "prefetch_accuracy": ratio(timely + late, timely + late + unused),
        "prefetch_timely_accuracy": ratio(timely, timely + late + unused),
        "prefetch_usefulness_per_issued": ratio(timely + late, issued),
        "prefetch_timeliness": ratio(timely, timely + late),
        "prefetch_slack_cycles_sum": slack_sum,
        "prefetch_slack_cycles_min": optional_count(counters, "prefetch_slack_cycles_min") if timely else None,
        "prefetch_slack_cycles_max": optional_count(counters, "prefetch_slack_cycles_max") if timely else None,
        "prefetch_slack_cycles_mean": ratio(slack_sum, timely) if slack_sum is not None else None,
        "roi_cycles": cycles,
        "roi_seconds": cycles / (frequency * 1000000),
        "aggregate_ipc": instructions / cycles,
        "mean_core_ipc": sum(core["ipc"] for core in cores) / len(cores),
        "dram_demand_read_bytes": demand_bytes,
        "dram_prefetch_read_bytes": prefetch_bytes,
        "dram_write_bytes": write_bytes,
        "dram_total_bytes": dram_bytes,
        "dram_bytes_per_cycle": dram_bytes / cycles,
        "instructions_per_dram_byte": ratio(instructions, dram_bytes),
        "simulator_wall_seconds": wall_seconds,
        "counters": {key: int(value) if value.isdigit() else value for key, value in counters.items()},
    }


def run_policy(args, policy, directory):
    directory.mkdir(exist_ok=True)
    index = 1
    while (directory / "attempt-{}".format(index)).exists():
        index += 1
    attempt = directory / "attempt-{}".format(index)
    attempt.mkdir()
    aliases = trace_links(args.traces, attempt)
    command = [
        str(args.binary),
        "--warmup_instructions={}".format(args.warmup),
        "--simulation_instructions={}".format(args.instructions),
        "-traces",
    ] + aliases
    settings = {
        "LLC_PF_POLICY": policy,
        "LLC_PF_DEGREE": str(args.degree),
        "LLC_PF_SKIP": str(args.skip),
    }
    env = os.environ.copy()
    env.pop("LLC_PF_EVENTS", None)
    if args.events:
        settings["LLC_PF_EVENTS"] = "demand.csv"
    env.update(settings)
    write_json(attempt / "command.json", {"argv": command, "environment": settings})
    log_path = attempt / "run.log"
    print("Running {} ({})".format(policy, log_path), flush=True)
    start = time.monotonic()
    with log_path.open("wb") as log:
        subprocess.run(command, cwd=str(attempt), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = parse_result(log_path, args, policy, aliases, time.monotonic() - start)
    if args.events and not (attempt / "demand.csv").is_file():
        raise ValueError("the simulator did not produce a demand event file")
    result["log"] = str(log_path.relative_to(args.output))
    return result


def write_summary(directory, results):
    ordered = [results[policy] for policy in POLICIES if policy in results]
    baseline = results.get("nopf")
    if baseline is not None:
        for result in ordered:
            mpki_ratio = ratio(result["llc_mpki"], baseline["llc_mpki"])
            result["net_miss_reduction"] = None if mpki_ratio is None else 1 - mpki_ratio
            result["roi_speedup"] = baseline["roi_cycles"] / result["roi_cycles"]
            for core, baseline_core in zip(result["cores"], baseline["cores"]):
                core["ipc_speedup"] = core["ipc"] / baseline_core["ipc"]
    write_json(directory / "summary.json", ordered)
    with (directory / "summary.tmp").open("w") as output:
        rows = [{key: value for key, value in result.items() if key not in ("cores", "counters")}
                for result in ordered]
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.tmp").replace(directory / "summary.csv")


def matched_start(result):
    warmup = [(core["warmup_retired_instructions"], core["warmup_cycles"]) for core in result["cores"]]
    return result["trace_seed"], result["cpu_mhz"], warmup


def run(args):
    config = {
        "binary": file_identity(args.binary),
        "traces": [file_identity(trace) for trace in args.traces],
        "warmup": args.warmup,
        "instructions": args.instructions,
        "skip": args.skip,
        "degree": args.degree,
        "events": args.events,
    }
    config_path = args.output / "config.json"
    if config_path.is_file():
        if read_json(config_path) != config:
            raise ValueError("the output directory already belongs to a different run configuration")
    else:
        write_json(config_path, config)
    results = {}
    for policy in POLICIES:
        path = args.output / policy / "result.json"
        if path.is_file():
            results[policy] = read_json(path)
    if results:
        write_summary(args.output, results)
    for policy in args.policies:
        if policy in results:
            continue
        directory = args.output / policy
        result = run_policy(args, policy, directory)
        start = matched_start(result)
        if any(matched_start(other) != start for other in results.values()):
            raise ValueError("trace seed, CPU frequency, or actual ROI start changed between policies")
        write_json(directory / "result.json", result)
        results[policy] = result
        write_summary(args.output, results)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("LLC run failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
