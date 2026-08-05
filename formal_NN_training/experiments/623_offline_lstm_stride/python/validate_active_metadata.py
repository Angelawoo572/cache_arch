#!/usr/bin/env python3
"""Torch-free, fail-closed validator for one active Stride v22 output."""
import argparse
import json
import re
from pathlib import Path

from analyze_replay import stream_hashes, validate_active_metadata
from model_contract import POLICY, TRACE, model_points_description


SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.metadata.is_file():
        raise SystemExit("missing metadata {}".format(args.metadata))
    try:
        metadata = json.loads(args.metadata.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit("invalid metadata {}: {}".format(args.metadata, exc))
    tag = metadata.get("model_tag")
    expected_tags = {
        point["model_tag"] for point in model_points_description()["points"]
    }
    if (
        not isinstance(tag, str) or SAFE_TAG.fullmatch(tag) is None
        or tag not in expected_tags
    ):
        raise SystemExit("metadata has an unconfigured model tag {!r}".format(tag))

    inputs = {POLICY: {}}
    for role in ("train", "guard", "eval"):
        inputs[POLICY][role] = {}
        for kind in ("stream", "candidate"):
            suffix = "stream" if kind == "stream" else "candidates"
            path = args.input_dir / "{}.{}.{}_{}.csv.gz".format(
                TRACE, POLICY, role, suffix
            )
            if not path.is_file():
                raise SystemExit("missing input {}".format(path))
            inputs[POLICY][role][kind] = stream_hashes(path)

    failures = []
    validate_active_metadata(metadata, tag, inputs, failures)
    if failures:
        raise SystemExit(
            "invalid active Stride metadata {}:\n{}".format(
                args.metadata, json.dumps(failures, indent=2, sort_keys=True)
            )
        )
    print("[PASS] {}".format(args.metadata))


if __name__ == "__main__":
    main()
