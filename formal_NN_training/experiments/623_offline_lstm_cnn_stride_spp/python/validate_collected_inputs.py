#!/usr/bin/env python3
"""Fail-closed post-collection validation for all 623 policy/window files."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


TRACE = "623.xalancbmk_s-700B"
POLICIES = ("stride", "spp")
ROLES = ("train", "guard", "eval")
LOGGER_SCHEMA = "623_causal_trigger_v4"
ATTACHMENT_MODE = "explicit_trigger_event_id"
EXPERIMENT_REVISION = "stride_spp_sliding_cnn_v4"


def as_int(value):
    text = str(value).strip()
    return int(text, 16) if text.lower().startswith("0x") else int(float(text))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_content_sha256(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("{},{},{},{}\n".format(*row)).encode())
    return digest.hexdigest()


def read_stream(path):
    rows = []
    occurrences = defaultdict(int)
    last_cycle = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "demand_idx", "cycle", "pc", "line", "pc_line_occ",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing stream columns {}".format(path, sorted(missing)))
        for index, row in enumerate(reader):
            cycle = as_int(row["cycle"])
            pc = as_int(row["pc"])
            line = as_int(row["line"])
            occ = as_int(row["pc_line_occ"])
            pair = (pc, line)
            expected_occ = occurrences[pair]
            occurrences[pair] += 1
            if row["trace"] != TRACE or as_int(row["demand_idx"]) != index:
                raise RuntimeError("{} identity/order failure at demand {}".format(path, index))
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError("{} contains stale logger schema".format(path))
            if cycle < last_cycle:
                raise RuntimeError("{} cycle order regressed at demand {}".format(path, index))
            if occ != expected_occ:
                raise RuntimeError("{} occurrence mismatch at demand {}".format(path, index))
            rows.append((index, pc, line, occ))
            last_cycle = cycle
    if not rows:
        raise RuntimeError("empty demand stream {}".format(path))
    return rows


def read_candidates(path, policy, stream_rows):
    counts = defaultdict(int)
    total = 0
    last_pf_event_id = -1
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "trace", "policy", "demand_idx", "pc", "line", "pc_line_occ",
            "candidate_rank", "pf_line", "accepted", "duplicate",
            "trigger_event_id", "pf_event_id", "event_distance", "match_mode",
            "logger_schema",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError("{} missing candidate columns {}".format(path, sorted(missing)))
        for row in reader:
            demand_idx = as_int(row["demand_idx"])
            if demand_idx < 0 or demand_idx >= len(stream_rows):
                raise RuntimeError("{} demand_idx out of range".format(path))
            index, pc, line, occ = stream_rows[demand_idx]
            observed = (
                as_int(row["demand_idx"]),
                as_int(row["pc"]),
                as_int(row["line"]),
                as_int(row["pc_line_occ"]),
            )
            if observed != (index, pc, line, occ):
                raise RuntimeError("{} transport identity mismatch".format(path))
            if row["trace"] != TRACE or row["policy"] != policy:
                raise RuntimeError("{} trace/policy mismatch".format(path))
            if row["logger_schema"] != LOGGER_SCHEMA:
                raise RuntimeError("{} contains stale logger schema".format(path))
            if row["match_mode"] != ATTACHMENT_MODE:
                raise RuntimeError("{} contains non-explicit candidate attachment".format(path))

            counts[demand_idx] += 1
            if as_int(row["candidate_rank"]) != counts[demand_idx]:
                raise RuntimeError("{} candidate ranks are not contiguous".format(path))
            trigger_id = as_int(row["trigger_event_id"])
            pf_event_id = as_int(row["pf_event_id"])
            distance = as_int(row["event_distance"])
            if pf_event_id <= last_pf_event_id:
                raise RuntimeError("{} PF event IDs are not increasing".format(path))
            if trigger_id >= pf_event_id or distance != pf_event_id - trigger_id:
                raise RuntimeError("{} explicit trigger distance mismatch".format(path))
            if distance < 1 or distance > 256:
                raise RuntimeError("{} trigger distance outside validated bound".format(path))
            accepted = as_int(row["accepted"])
            duplicate = as_int(row["duplicate"])
            if accepted not in (0, 1) or duplicate not in (0, 1) or (duplicate and not accepted):
                raise RuntimeError("{} invalid candidate outcome bits".format(path))
            last_pf_event_id = pf_event_id
            total += 1
    if total == 0:
        raise RuntimeError("empty candidate bank {}".format(path))
    return total, max(counts.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = {
        "status": "PASS",
        "trace": TRACE,
        "experiment_revision": EXPERIMENT_REVISION,
        "event_logger_schema": LOGGER_SCHEMA,
        "candidate_attachment_mode": ATTACHMENT_MODE,
        "cross_policy_identity_required_for_fair_claim": False,
        "cross_policy_demand_identity_diagnostic": {},
        "tracks": {},
    }
    streams = {}
    for policy in POLICIES:
        manifest["tracks"][policy] = {}
        for role in ROLES:
            stream_path = args.input_dir / "{}.{}.{}_stream.csv.gz".format(
                TRACE, policy, role
            )
            candidate_path = args.input_dir / "{}.{}.{}_candidates.csv.gz".format(
                TRACE, policy, role
            )
            if not stream_path.is_file() or not candidate_path.is_file():
                raise RuntimeError("missing normalized {} {} inputs".format(policy, role))
            stream_rows = read_stream(stream_path)
            candidate_count, max_candidates = read_candidates(
                candidate_path, policy, stream_rows
            )
            streams[(policy, role)] = stream_rows
            manifest["tracks"][policy][role] = {
                "demand_callbacks": len(stream_rows),
                "candidate_requests": candidate_count,
                "max_candidates_per_demand": max_candidates,
                "demand_identity_sha256": identity_sha256(stream_rows),
                "stream_gzip_sha256": sha256(stream_path),
                "stream_content_sha256": gzip_content_sha256(stream_path),
                "candidate_gzip_sha256": sha256(candidate_path),
                "candidate_content_sha256": gzip_content_sha256(candidate_path),
            }

    for role in ROLES:
        stride_rows = streams[("stride", role)]
        spp_rows = streams[("spp", role)]
        manifest["cross_policy_demand_identity_diagnostic"][role] = {
            "identical": stride_rows == spp_rows,
            "stride_callbacks": len(stride_rows),
            "spp_callbacks": len(spp_rows),
            "stride_identity_sha256": identity_sha256(stride_rows),
            "spp_identity_sha256": identity_sha256(spp_rows),
            "interpretation": (
                "diagnostic only; stride and SPP are independent matched tracks"
            ),
        }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[PASS] {}".format(args.manifest_out))


if __name__ == "__main__":
    main()
