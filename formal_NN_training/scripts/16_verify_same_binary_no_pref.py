#!/usr/bin/env python3
"""Verify frozen same-binary/no-prefetch IPC references before replay.

This is a Python 3.6 and standard-library-only utility for Sacramento.
"""
import argparse
import json
import re
from pathlib import Path

IPC_RE = re.compile(r"(?:Core_0_IPC|CPU 0 cumulative IPC)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")


def parse_ipc(path):
    matches = IPC_RE.findall(path.read_text(errors="replace"))
    if not matches:
        raise ValueError("missing Core_0_IPC in {}".format(path))
    return float(matches[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--traces", nargs="*", default=None)
    args = ap.parse_args()

    reference = json.loads(args.reference.read_text())
    expected_by_trace = reference.get("references", reference)
    if not isinstance(expected_by_trace, dict) or not expected_by_trace:
        raise SystemExit("reference has no nonempty references map")
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = float(reference.get("run_identity", {}).get("tolerance_abs_ipc", 0.0005))
    if tolerance < 0:
        raise SystemExit("tolerance must be nonnegative")

    rows = []
    failures = []
    for trace in args.traces or sorted(expected_by_trace):
        if trace not in expected_by_trace:
            failures.append("{}: absent from reference".format(trace))
            continue
        log = args.log_root / (trace + ".same_binary_no_pref.log")
        expected = float(expected_by_trace[trace])
        try:
            observed = parse_ipc(log)
        except Exception as exc:
            rows.append({"trace": trace, "reference_ipc": expected,
                         "status": "missing_or_unparseable", "log": str(log)})
            failures.append(str(exc))
            continue
        delta = observed - expected
        status = "PASS" if abs(delta) <= tolerance else "DRIFT"
        rows.append({
            "trace": trace,
            "reference_ipc": expected,
            "observed_ipc": observed,
            "delta_ipc": delta,
            "tolerance_abs_ipc": tolerance,
            "status": status,
            "log": str(log),
        })
        if status != "PASS":
            failures.append(
                "{}: observed={:.6f} reference={:.6f} delta={:+.6f} tol={:.6f}".format(
                    trace, observed, expected, delta, tolerance
                )
            )

    payload = {
        "reference": str(args.reference),
        "log_root": str(args.log_root),
        "tolerance_abs_ipc": tolerance,
        "rows": rows,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print("[baseline guard] {}: {}".format(row["trace"], row["status"]))
    if failures:
        raise SystemExit("[baseline guard FAIL] " + " | ".join(failures))
    print("[baseline guard PASS] {} traces within +/-{:.6f} IPC".format(len(rows), tolerance))


if __name__ == "__main__":
    main()
