#!/usr/bin/env python3
"""Safely install one Colab output archive into its canonical run directory.

This helper is intentionally Python 3.6 and standard-library only.  The active
623 replay launchers call it automatically, so users never need to paste Python
heredocs into an interactive Sacramento shell.
"""
import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath


COMMON_REQUIRED_FILES = (
    "run_metadata.json",
    "offline_nn.replay.csv",
    "model.pt",
    "training_history.csv",
)
ROOT_REQUIRED_FILES = ("sweep_manifest.json",)


def fail(message):
    raise RuntimeError(message)


def parse_tags(value):
    tags = [item.strip() for item in value.split(",") if item.strip()]
    if not tags or len(tags) != len(set(tags)):
        fail("model tags must be a nonempty unique comma-separated list")
    for tag in tags:
        path = PurePosixPath(tag)
        if path.is_absolute() or len(path.parts) != 1 or tag in (".", ".."):
            fail("unsafe model tag {!r}".format(tag))
    return tags


def visible_children(path):
    return sorted(item.name for item in path.iterdir()) if path.exists() else []


def verify_installed(output_dir, tags):
    observed = visible_children(output_dir)
    expected = sorted(tags + list(ROOT_REQUIRED_FILES))
    if observed != expected:
        fail(
            "Colab output top-level entries {} do not match expected entries {}"
            .format(observed, expected)
        )
    for name in ROOT_REQUIRED_FILES:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            fail("missing or empty Colab output {}".format(path))
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            fail("invalid Colab sweep manifest {}: {}".format(path, error))
        points = manifest.get("points") if isinstance(manifest, dict) else None
        manifest_tags = (
            [point.get("model_tag") for point in points]
            if isinstance(points, list)
            and all(isinstance(point, dict) for point in points)
            else None
        )
        if manifest_tags is None or sorted(manifest_tags) != sorted(tags):
            fail(
                "sweep manifest model tags {} do not match expected {}"
                .format(manifest_tags, sorted(tags))
            )
    for tag in tags:
        tag_dir = output_dir / tag
        if not tag_dir.is_dir():
            fail("missing model output directory {}".format(tag_dir))
        for name in COMMON_REQUIRED_FILES:
            path = tag_dir / name
            if not path.is_file() or path.stat().st_size <= 0:
                fail("missing or empty Colab output {}".format(path))


def validate_members(members, tags):
    expected = set(tags).union(ROOT_REQUIRED_FILES)
    observed = set()
    for member in members:
        path = PurePosixPath(member.name)
        if not member.name or path.is_absolute() or ".." in path.parts:
            fail("unsafe archive path {!r}".format(member.name))
        if member.issym() or member.islnk():
            fail("archive links are not allowed: {}".format(member.name))
        if not member.isdir() and not member.isfile():
            fail("unsupported archive entry: {}".format(member.name))
        if not path.parts:
            fail("archive contains an empty path")
        observed.add(path.parts[0])
    if observed != expected:
        fail(
            "archive top-level entries {} do not match expected tags {}"
            .format(sorted(observed), sorted(expected))
        )


def install(archive, output_dir, tags):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = visible_children(output_dir)
    if existing:
        verify_installed(output_dir, tags)
        print("[PASS] Colab output already installed in {}".format(output_dir))
        return
    if not archive.is_file() or archive.stat().st_size <= 0:
        fail("missing Colab archive {}".format(archive))
    with tarfile.open(str(archive), "r:gz") as handle:
        members = handle.getmembers()
        if not members:
            fail("empty Colab archive {}".format(archive))
        validate_members(members, tags)
        handle.extractall(str(output_dir), members=members)
    verify_installed(output_dir, tags)
    print("[PASS] installed {} into {}".format(archive, output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-tags", required=True)
    args = parser.parse_args()
    install(args.archive, args.output_dir, parse_tags(args.model_tags))


if __name__ == "__main__":
    main()
