#!/usr/bin/env python3
"""Report the root status of one 623 matched comparison.

Recursive text search is unsafe because matched_comparison.json embeds child
manifests that have their own status fields.  This helper reads only the root
status and prints every root failure.  It is Python 3.6/standard-library only.
"""
import argparse
import json
import sys
from pathlib import Path


DEFAULT_RUN_ID = "623_offline_lstm_spp_factorized_fill_v17_seed7"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir or (EXPERIMENT_DIR / "runs" / args.run_id)
    result_path = run_dir / "matched_comparison.json"
    if not result_path.is_file():
        print("[NOT READY] {}".format(result_path))
        return 2

    try:
        payload = json.loads(result_path.read_text())
    except (OSError, ValueError) as error:
        print("[INVALID] {}: {}".format(result_path, error))
        return 3

    status = payload.get("status")
    failures = payload.get("failures")
    if not isinstance(failures, list):
        print("[INVALID] {} has no root failures list".format(result_path))
        return 3

    print("[{}] {}".format(status, result_path))
    for failure in failures:
        print("  - {}".format(failure))

    if status == "PASS" and not failures:
        return 0
    if status == "FAIL" and failures:
        return 1
    print(
        "[INVALID] inconsistent root status/failures: status={!r}, count={}"
        .format(status, len(failures))
    )
    return 3


if __name__ == "__main__":
    sys.exit(main())

