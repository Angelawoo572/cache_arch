#!/usr/bin/env python3
"""Repair the v25 Stride Colab sweep-manifest reference omission.

The affected archive already contains a freshly generated
validated_collection_manifest.json and all five trained model artifacts.  This
tool adds the missing reference in sweep_manifest.json, verifies run identity,
and proves every non-manifest archive payload is unchanged.
"""
from __future__ import print_function

import argparse
import copy
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


SWEEP_NAME = "sweep_manifest.json"
VALIDATION_NAME = "validated_collection_manifest.json"


def fail(message):
    raise RuntimeError(message)


def safe_members(handle):
    members = handle.getmembers()
    seen = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not member.name
            or path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or (not member.isdir() and not member.isfile())
        ):
            fail("unsafe or unsupported archive member {!r}".format(member.name))
        canonical = path.as_posix()
        if canonical in seen:
            fail("duplicate archive member {}".format(canonical))
        seen.add(canonical)
    for required in (SWEEP_NAME, VALIDATION_NAME):
        matches = [member for member in members if member.name == required]
        if len(matches) != 1 or not matches[0].isfile():
            fail("archive must contain exactly one top-level {}".format(required))
    return members


def read_member(handle, members, name):
    member = next(item for item in members if item.name == name)
    stream = handle.extractfile(member)
    if stream is None:
        fail("cannot read archive member {}".format(name))
    with stream:
        return stream.read()


def json_object(data, label):
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        fail("invalid {}: {}".format(label, error))
    if not isinstance(value, dict):
        fail("{} must be one JSON object".format(label))
    return value


def payload_hashes(path):
    result = {}
    with tarfile.open(str(path), "r:gz") as handle:
        for member in safe_members(handle):
            if not member.isfile() or member.name == SWEEP_NAME:
                continue
            stream = handle.extractfile(member)
            if stream is None:
                fail("cannot read {}".format(member.name))
            digest = hashlib.sha256()
            with stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[member.name] = (member.size, digest.hexdigest())
    return result


def load_manifests(archive):
    with tarfile.open(str(archive), "r:gz") as source:
        members = safe_members(source)
        sweep_bytes = read_member(source, members, SWEEP_NAME)
        validation_bytes = read_member(source, members, VALIDATION_NAME)
    return (
        sweep_bytes,
        json_object(sweep_bytes, SWEEP_NAME),
        json_object(validation_bytes, VALIDATION_NAME),
    )


def verify_identity(sweep, validation, run_id, trace, policy):
    for label, payload in (("sweep", sweep), ("validation", validation)):
        expected = {"trace": trace, "policy": policy}
        if label == "sweep":
            expected["run_id"] = run_id
        bad = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if bad:
            fail("{} manifest identity mismatch: {}".format(label, bad))
    if validation.get("status") != "PASS":
        fail("validated collection manifest status is not PASS")


def rewrite_archive(archive, corrected_sweep_bytes):
    before = payload_hashes(archive)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stride_v25_manifest_fix.",
        suffix=".tar.gz",
        dir=str(archive.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(str(archive), "r:gz") as source:
            members = safe_members(source)
            with tarfile.open(str(temporary), "w:gz") as target:
                for member in members:
                    if member.name == SWEEP_NAME:
                        updated = copy.copy(member)
                        updated.size = len(corrected_sweep_bytes)
                        target.addfile(updated, io.BytesIO(corrected_sweep_bytes))
                    elif member.isfile():
                        stream = source.extractfile(member)
                        if stream is None:
                            fail("cannot copy {}".format(member.name))
                        with stream:
                            target.addfile(member, stream)
                    else:
                        target.addfile(member)
        if before != payload_hashes(temporary):
            fail("non-manifest model/replay payload changed during repair")
        _, repaired, _ = load_manifests(temporary)
        if (
            repaired.get("status") != "PASS"
            or repaired.get("fresh_input_validation_manifest")
            != VALIDATION_NAME
        ):
            fail("repaired sweep manifest failed verification")
        os.replace(str(temporary), str(archive))
    finally:
        if temporary.exists():
            temporary.unlink()


def synchronize_installed_manifest(path, payload):
    if not path.parent.is_dir():
        return
    current = path.read_bytes() if path.is_file() else None
    if current == payload:
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sweep_manifest.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print("[PASS] synchronized extracted Stride sweep metadata")


def repair(archive, installed_manifest, run_id, trace, policy):
    if not archive.is_file() or archive.stat().st_size <= 0:
        fail("missing or empty Stride Colab output {}".format(archive))
    original_bytes, sweep, validation = load_manifests(archive)
    verify_identity(sweep, validation, run_id, trace, policy)

    reference = sweep.get("fresh_input_validation_manifest")
    if reference not in (None, VALIDATION_NAME):
        fail("unexpected fresh input validation reference {!r}".format(reference))
    status = sweep.get("status")
    if status not in (None, "PASS"):
        fail("unexpected sweep status {!r}".format(status))

    changed = reference is None or status is None
    sweep["status"] = "PASS"
    sweep["fresh_input_validation_manifest"] = VALIDATION_NAME
    corrected_bytes = (
        json.dumps(sweep, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    if changed:
        rewrite_archive(archive, corrected_bytes)
        installed_bytes = corrected_bytes
        print("[PASS] repaired Stride sweep metadata only")
    else:
        installed_bytes = original_bytes
        print("[PASS] Stride sweep metadata was already correct")

    synchronize_installed_manifest(installed_manifest, installed_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--installed-manifest", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    repair(
        args.archive,
        args.installed_manifest,
        args.run_id,
        args.trace,
        args.policy,
    )


if __name__ == "__main__":
    main()
