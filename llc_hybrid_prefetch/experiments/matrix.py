"""Explicit workload, placement, and phase registration for this characterization."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "raw" / "traces"
WARMUP = 10_000_000
ROI = 5_000_000
PREFIX = WARMUP + ROI + 250_000
PROGRAMS = {
    "mcf": ("605.mcf_s", [(665, 0.440506), (994, 0.142785)]),
    "cactus": ("607.cactuBSSN_s", [(2421, 0.902072), (3477, 0.0392205)]),
    "xalan": ("623.xalancbmk_s", [(700, 0.413137), (592, 0.257563)]),
    "lbm": ("619.lbm_s", [(4268, 0.418157), (3766, 0.214938)]),
}
MIXES = (("mcf", "cactus"), ("mcf", "xalan"), ("lbm", "xalan"))
SYNTHETIC = ("steady", "nested", "nested_samepage", "nested_crosspage", "nested_variable",
             "interleaved", "phase", "recurring", "paced_nested_samepage")
SYN_WARMUP = 100_000
SYN_ROI = 500_000
POLICIES = ("nopf", "stride")


def trace_name(key, region):
    stem, points = PROGRAMS[key]
    return f"{stem}-{points[region][0]}B"


def workloads():
    for key in PROGRAMS:
        for region in (0, 1):
            yield trace_name(key, region)


def run_specs():
    for keys in [(k,) for k in PROGRAMS] + list(MIXES):
        for region in (0, 1):
            traces = [trace_name(k, region) for k in keys]
            run_group = "-".join(keys) + f"-r{region + 1}"
            for policy in POLICIES:
                yield dict(run_id=run_group + "-" + policy, group="spec", policy=policy,
                           workloads=list(keys), traces=traces, region_rank=region + 1,
                           warmup=WARMUP, roi=ROI, skip=0)
    for policy in POLICIES:
        yield dict(run_id="warm5-mcf-cactus-r1-" + policy, group="warmup", policy=policy,
                   workloads=["mcf", "cactus"], traces=[trace_name(k, 0) for k in MIXES[0]],
                   region_rank=1, warmup=5_000_000, roi=ROI, skip=5_000_000)
    for case in SYNTHETIC:
        for policy in POLICIES:
            yield dict(run_id="syn-" + case + "-" + policy, group="synthetic", policy=policy,
                       workloads=[case], traces=["synthetic-" + case], region_rank=None,
                       warmup=SYN_WARMUP, roi=SYN_ROI, skip=0)


if __name__ == "__main__":
    import json
    print(json.dumps(list(run_specs()), indent=2))
