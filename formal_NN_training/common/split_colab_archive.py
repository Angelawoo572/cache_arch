#!/usr/bin/env python3
"""Split, verify, reassemble, and safely extract Colab transfer archives.

The multipart manifest is the authority for both the complete ``.tar.gz``
archive and every numbered part.  This module intentionally uses only the
Python standard library so the same file can run on the Sacramento host, a
Mac, or Colab.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


MIB = 1024 * 1024
MAX_PART_MIB = 90
MAX_PART_BYTES = MAX_PART_MIB * MIB
SCHEMA = "cache_arch.colab_archive_parts.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArchiveContractError(RuntimeError):
    """Raised when a transfer archive or manifest violates the contract."""


def sha256_file(path, chunk_bytes=8 * MIB):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_basename(value, field):
    if not isinstance(value, str) or not value:
        raise ArchiveContractError(f"{field} must be a nonempty string")
    if value != Path(value).name or value in {".", ".."} or "\\" in value:
        raise ArchiveContractError(f"unsafe {field}: {value!r}")
    return value


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArchiveContractError(f"{field} must be a positive integer")
    return value


def _sha256(value, field):
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ArchiveContractError(f"{field} must be a lowercase SHA256 digest")
    return value


def validate_manifest(payload):
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ArchiveContractError(f"manifest schema must be {SCHEMA!r}")
    if set(payload) != {"schema", "archive", "max_part_bytes", "parts"}:
        raise ArchiveContractError("manifest has missing or unexpected top-level fields")

    archive = payload["archive"]
    if not isinstance(archive, dict) or set(archive) != {
        "name", "size_bytes", "sha256", "format"
    }:
        raise ArchiveContractError("manifest archive record has invalid fields")
    archive_name = _safe_basename(archive["name"], "archive.name")
    if not archive_name.endswith(".tar.gz") or archive["format"] != "tar+gzip":
        raise ArchiveContractError("archive must be a gzip-compressed tar named *.tar.gz")
    archive_size = _positive_int(archive["size_bytes"], "archive.size_bytes")
    _sha256(archive["sha256"], "archive.sha256")

    max_part_bytes = _positive_int(payload["max_part_bytes"], "max_part_bytes")
    if max_part_bytes > MAX_PART_BYTES:
        raise ArchiveContractError(
            f"max_part_bytes exceeds the {MAX_PART_MIB} MiB transfer limit"
        )
    parts = payload["parts"]
    if not isinstance(parts, list) or not parts:
        raise ArchiveContractError("manifest parts must be a nonempty list")

    total = 0
    seen_names = set()
    for expected_index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != {
            "index", "name", "size_bytes", "sha256"
        }:
            raise ArchiveContractError(f"invalid part record at index {expected_index}")
        if part["index"] != expected_index:
            raise ArchiveContractError("part indices must be contiguous and zero-based")
        expected_name = f"{archive_name}.part-{expected_index:05d}"
        name = _safe_basename(part["name"], f"parts[{expected_index}].name")
        if name != expected_name or name in seen_names:
            raise ArchiveContractError(
                f"part {expected_index} must be named {expected_name!r} exactly"
            )
        seen_names.add(name)
        size = _positive_int(part["size_bytes"], f"parts[{expected_index}].size_bytes")
        if size > max_part_bytes or size > MAX_PART_BYTES:
            raise ArchiveContractError(f"part {name!r} exceeds the size limit")
        _sha256(part["sha256"], f"parts[{expected_index}].sha256")
        total += size
    if total != archive_size:
        raise ArchiveContractError(
            f"part byte total {total} does not match archive size {archive_size}"
        )
    return payload


def load_manifest(path):
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveContractError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return validate_manifest(payload)


def split_archive(archive_path, output_dir, max_part_bytes=MAX_PART_BYTES,
                  overwrite=False):
    archive = Path(archive_path).resolve()
    if not archive.is_file() or archive.is_symlink():
        raise ArchiveContractError(f"archive is not a regular file: {archive}")
    if not archive.name.endswith(".tar.gz"):
        raise ArchiveContractError("input archive name must end in .tar.gz")
    if not isinstance(max_part_bytes, int) or not 0 < max_part_bytes <= MAX_PART_BYTES:
        raise ArchiveContractError(
            f"part size must be between 1 byte and {MAX_PART_MIB} MiB"
        )
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            next(iter(handle), None)
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveContractError(f"input is not a readable gzip tar archive: {exc}") from exc

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / f"{archive.name}.parts.json"
    if manifest_path.exists() and not overwrite:
        raise ArchiveContractError(f"refusing to overwrite {manifest_path}")

    staging = Path(tempfile.mkdtemp(prefix=f".{archive.name}.split-", dir=destination))
    archive_digest = hashlib.sha256()
    part_records = []
    total_bytes = 0
    try:
        with archive.open("rb") as source:
            index = 0
            while True:
                chunk = source.read(max_part_bytes)
                if not chunk:
                    break
                archive_digest.update(chunk)
                total_bytes += len(chunk)
                name = f"{archive.name}.part-{index:05d}"
                part_path = staging / name
                part_path.write_bytes(chunk)
                part_records.append({
                    "index": index,
                    "name": name,
                    "size_bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                })
                index += 1
        if not part_records:
            raise ArchiveContractError("cannot split an empty archive")
        if total_bytes != archive.stat().st_size:
            raise ArchiveContractError("archive size changed while it was being split")

        payload = validate_manifest({
            "schema": SCHEMA,
            "archive": {
                "name": archive.name,
                "size_bytes": total_bytes,
                "sha256": archive_digest.hexdigest(),
                "format": "tar+gzip",
            },
            "max_part_bytes": max_part_bytes,
            "parts": part_records,
        })
        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        targets = [destination / record["name"] for record in part_records]
        targets.append(manifest_path)
        existing = [path for path in targets if path.exists()]
        if existing and not overwrite:
            raise ArchiveContractError(
                "refusing to overwrite existing transfer files: "
                + ", ".join(str(path) for path in existing)
            )
        for record in part_records:
            os.replace(staging / record["name"], destination / record["name"])
        os.replace(staged_manifest, manifest_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest_path


def validate_parts(manifest, parts_dir):
    payload = load_manifest(manifest) if not isinstance(manifest, dict) else validate_manifest(manifest)
    directory = Path(parts_dir).resolve()
    if not directory.is_dir():
        raise ArchiveContractError(f"parts directory does not exist: {directory}")
    for part in payload["parts"]:
        path = directory / part["name"]
        if not path.is_file() or path.is_symlink():
            raise ArchiveContractError(f"missing regular part file: {path}")
        observed_size = path.stat().st_size
        if observed_size != part["size_bytes"]:
            raise ArchiveContractError(
                f"size mismatch for {path.name}: {observed_size} != {part['size_bytes']}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != part["sha256"]:
            raise ArchiveContractError(f"SHA256 mismatch for {path.name}")
    return payload


def reassemble_archive(manifest_path, parts_dir, output_path, overwrite=False):
    payload = validate_parts(manifest_path, parts_dir)
    output = Path(output_path).resolve()
    if output.name != payload["archive"]["name"]:
        raise ArchiveContractError(
            f"output must be named {payload['archive']['name']!r} exactly"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise ArchiveContractError(f"refusing to overwrite {output}")

    digest = hashlib.sha256()
    size = 0
    temporary = output.with_name(f".{output.name}.reassembling-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as destination:
            for part in payload["parts"]:
                with (Path(parts_dir).resolve() / part["name"]).open("rb") as source:
                    while True:
                        chunk = source.read(8 * MIB)
                        if not chunk:
                            break
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if size != payload["archive"]["size_bytes"]:
            raise ArchiveContractError("reassembled archive size mismatch")
        if digest.hexdigest() != payload["archive"]["sha256"]:
            raise ArchiveContractError("reassembled archive SHA256 mismatch")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _normalized_member_name(name):
    if not name or "\x00" in name or "\\" in name:
        raise ArchiveContractError(f"unsafe tar member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveContractError(f"unsafe tar member path: {name!r}")
    normalized = PurePosixPath(*[part for part in path.parts if part != "."])
    if not normalized.parts:
        raise ArchiveContractError(f"empty tar member path: {name!r}")
    return normalized


def safe_extract_tar_gz(archive_path, output_dir):
    """Extract only unique regular files/directories from a gzip tar archive."""

    archive = Path(archive_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ArchiveContractError(f"safe extraction destination already exists: {destination}")
    if not archive.is_file() or archive.is_symlink():
        raise ArchiveContractError(f"archive is not a regular file: {archive}")

    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members = handle.getmembers()
            records = []
            kinds = {}
            for member in members:
                relative = _normalized_member_name(member.name)
                if relative in kinds:
                    raise ArchiveContractError(f"duplicate tar member: {relative}")
                if member.issym() or member.islnk():
                    raise ArchiveContractError(f"tar links are forbidden: {member.name!r}")
                if member.isdir():
                    kind = "directory"
                elif member.isfile():
                    kind = "file"
                else:
                    raise ArchiveContractError(
                        f"special tar entry is forbidden: {member.name!r}"
                    )
                kinds[relative] = kind
                records.append((member, relative, kind))

            for relative, kind in kinds.items():
                for parent in relative.parents:
                    if not parent.parts:
                        continue
                    if kinds.get(parent) == "file":
                        raise ArchiveContractError(
                            f"tar file {parent} is an ancestor of {relative}"
                        )
                if kind == "file" and any(
                    other != relative and relative in other.parents for other in kinds
                ):
                    raise ArchiveContractError(
                        f"tar file {relative} conflicts with a child entry"
                    )

            destination.mkdir(parents=True, exist_ok=False)
            try:
                for _, relative, kind in sorted(
                    records, key=lambda item: (len(item[1].parts), item[2] != "directory")
                ):
                    target = destination.joinpath(*relative.parts)
                    if kind == "directory":
                        target.mkdir(parents=True, exist_ok=True)
                for member, relative, kind in records:
                    if kind != "file":
                        continue
                    target = destination.joinpath(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = handle.extractfile(member)
                    if source is None:
                        raise ArchiveContractError(f"cannot read tar member {member.name!r}")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=8 * MIB)
            except Exception:
                shutil.rmtree(destination, ignore_errors=True)
                raise
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveContractError(f"cannot safely extract {archive}: {exc}") from exc
    return destination


def validate_sha256sums(root_dir, sums_name="SHA256SUMS"):
    root = Path(root_dir).resolve()
    sums_path = root / _safe_basename(sums_name, "SHA256SUMS filename")
    if not sums_path.is_file() or sums_path.is_symlink():
        raise ArchiveContractError(f"missing regular {sums_name}: {sums_path}")
    verified = []
    seen = set()
    for line_number, raw_line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, raw_name = raw_line.split(maxsplit=1)
        except ValueError as exc:
            raise ArchiveContractError(
                f"malformed {sums_name} line {line_number}"
            ) from exc
        expected = _sha256(expected, f"{sums_name} line {line_number}")
        name = raw_name[1:] if raw_name.startswith("*") else raw_name
        relative = _normalized_member_name(name)
        if relative in seen:
            raise ArchiveContractError(f"duplicate {sums_name} entry: {relative}")
        seen.add(relative)
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ArchiveContractError(f"missing regular checksummed file: {relative}")
        if sha256_file(path) != expected:
            raise ArchiveContractError(f"SHA256SUMS mismatch: {relative}")
        verified.append(relative.as_posix())
    if not verified:
        raise ArchiveContractError(f"{sums_name} contains no file records")
    observed_payloads = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArchiveContractError(f"symlink found below checksum root: {path}")
        if path.is_file() and path != sums_path:
            observed_payloads.add(path.relative_to(root).as_posix())
    if set(verified) != observed_payloads:
        missing = sorted(observed_payloads - set(verified))
        unexpected = sorted(set(verified) - observed_payloads)
        raise ArchiveContractError(
            "SHA256SUMS coverage mismatch; unlisted files={!r}, "
            "non-payload entries={!r}".format(missing, unexpected)
        )
    return verified


def _summary(payload, manifest_path):
    return json.dumps({
        "status": "PASS",
        "manifest": str(manifest_path),
        "archive": payload["archive"],
        "part_count": len(payload["parts"]),
        "max_part_bytes": payload["max_part_bytes"],
    }, indent=2, sort_keys=True)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    split = subparsers.add_parser("split", help="split a .tar.gz and write its manifest")
    split.add_argument("archive")
    split.add_argument("--output-dir", required=True)
    split.add_argument("--max-part-mib", type=int, default=MAX_PART_MIB)
    split.add_argument("--overwrite", action="store_true")

    verify = subparsers.add_parser("verify", help="verify all parts against a manifest")
    verify.add_argument("manifest")
    verify.add_argument("--parts-dir", required=True)

    join = subparsers.add_parser("join", help="verify and reassemble an archive")
    join.add_argument("manifest")
    join.add_argument("--parts-dir", required=True)
    join.add_argument("--output", required=True)
    join.add_argument("--overwrite", action="store_true")

    extract = subparsers.add_parser("extract", help="safely extract and verify SHA256SUMS")
    extract.add_argument("archive")
    extract.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    if args.command == "split":
        if not 0 < args.max_part_mib <= MAX_PART_MIB:
            raise ArchiveContractError(
                f"--max-part-mib must be between 1 and {MAX_PART_MIB}"
            )
        manifest_path = split_archive(
            args.archive,
            args.output_dir,
            max_part_bytes=args.max_part_mib * MIB,
            overwrite=args.overwrite,
        )
        print(_summary(load_manifest(manifest_path), manifest_path))
    elif args.command == "verify":
        payload = validate_parts(args.manifest, args.parts_dir)
        print(_summary(payload, Path(args.manifest).resolve()))
    elif args.command == "join":
        output = reassemble_archive(
            args.manifest, args.parts_dir, args.output, overwrite=args.overwrite
        )
        payload = load_manifest(args.manifest)
        print(_summary(payload, Path(args.manifest).resolve()))
        print(f"reassembled={output}")
    elif args.command == "extract":
        output = safe_extract_tar_gz(args.archive, args.output_dir)
        verified = validate_sha256sums(output)
        print(json.dumps({
            "status": "PASS",
            "output_dir": str(output),
            "sha256sums_verified": verified,
        }, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse makes this unreachable
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveContractError as exc:
        raise SystemExit(f"ERROR: {exc}")
