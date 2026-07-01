#!/usr/bin/env python3
"""Fail replay preflight when frozen same-binary no-pref IPC references drift.

This guard intentionally checks the logs produced by the exact standalone
ListReplayer binary with L2 prefetching disabled.  It does not compare against
any separate normal-prefetcher binary, so candidate IPC deltas remain anchored
to the replay mechanism actually being evaluated.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

IPC_RE = re.compile(r"(?:Core_0_IPC|CPU 0 cumulative IPC)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")


def parse_ipc(path: Path) -> float:
    text = path.read_text(errors="replace")
    matches = IPC_RE.findall(text)
    if not matches:
        raise ValueError(f"missing Core_0_IPC in {path}")
    return float(matches[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--traces", nargs="*", default=None)
    args = ap.parse_args()

    ref = json.loads(args.reference.read_text())
    refs = ref.get("references", ref)
    if not isinstance(refs, dict) or not refs:
        raise SystemExit("[baseline guard] reference has no nonempty 'references' map")
    tol = args.tolerance
    if tol is None:
        tol = float(ref.get("run_identity", {}).get("tolerance_abs_ipc", 0.0005))
    if tol < 0:
        raise SystemExit("[baseline guard] tolerance must be nonnegative")

    traces = args.traces or sorted(refs)
    rows = []
    failed = []
    for trace in traces:
        if trace not in refs:
            failed.append(f"{trace}: absent from reference")
            continue
        log = args.log_root / f"{trace}.same_binary_no_pref.log"
        try:
            observed = parse_ipc(log)
        except Exception as exc:  # keep all failures in the proof artifact
            failed.append(str(exc))
            rows.append({"trace": trace, "reference_ipc": float(refs[trace]), "status": "missing_or_unparseable", "log": str(log)})
            continue
        expected = float(refs[trace])
        delta = observed - expected
        ok = abs(delta) <= tol
        rows.append({
            "trace": trace,
            "reference_ipc": expected,
            "observed_ipc": observed,
            "delta_ipc": delta,
            "tolerance_abs_ipc": tol,
            "status": "PASS" if ok else "DRIFT",
            "log": str(log),
        })
        if not ok:
            failed.append(f"{trace}: observed={observed:.6f} reference={expected:.6f} delta={delta:+.6f} tol={tol:.6f}")

    payload = {
        "reference": str(args.reference),
        "log_root": str(args.log_root),
        "tolerance_abs_ipc": tol,
        "rows": rows,
        "status": "PASS" if not failed else "FAIL",
        "failures": failed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print("[baseline guard] {trace}: {status}".format(**row))
    if failed:
        raise SystemExit("[baseline guard FAIL] " + " | ".join(failed))
    print(f"[baseline guard PASS] {len(rows)} traces within ±{tol:.6f} IPC")


if __name__ == "__main__":
    main()
