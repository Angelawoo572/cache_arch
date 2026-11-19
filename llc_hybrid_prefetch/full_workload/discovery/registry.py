"""Emit exact portable commands, fixed work, fitting provenance and completion for these intervention arms."""

from streams import location
from replay import CASES, POLICIES
from characterize import publish


def describe(scope, case, arm):
    root = location(case, arm, scope)
    import json

    record = root / "complete.json"
    state = "complete" if record.exists() else "pending"
    data = json.loads(record.read_text()) if record.exists() else {}
    if scope == "full":
        fit = "shape.rules" if arm == "shape" else ""
        cmd = f"python3 full_replay.py {'--policy sparse3 ' if arm == 'sparse3' else ''}{case}"
        warm, roi, skip = (
            50_000_000,
            (450_000_001 if case == "pr_heldout_entire" else 200_000_000),
            0,
        )
        if data:
            spec = data["config"]
            assert (spec["warmup"], spec["roi"], spec["skip"]) == (warm, roi, skip)
    else:
        fit = arm + ".rules" if arm in ("shape", "ordered", "local") else ""
        cmd = f"python3 replay.py --cases {case} --policies {arm}"
        warm, roi, skip = 1_000_000, 500_000, 2_000_000
        if data:
            assert (data["warmup"], data["roi"], data["skip"]) == (warm, roi, skip)
    return dict(
        scope=scope,
        case=case,
        arm=arm,
        source=case + ".champsimtrace",
        skip=skip,
        warmup=warm,
        measured=roi,
        physical_slot=0,
        geometry_cores=2,
        active_cores=1,
        measured_first_ordinal=skip + warm + 1,
        measured_last_ordinal=skip + warm + roi,
        independent_rule_file=fit,
        command=cmd,
        status=state,
    )


if __name__ == "__main__":
    rows = [describe("diagnostic", c, p) for c in CASES for p in POLICIES]
    rows.extend(describe("full", c, "shape") for c in (*CASES, "pr_heldout_entire"))
    rows.append(describe("full", "bfs_discovery", "sparse3"))
    publish("run_registry", rows)
