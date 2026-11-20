"""Move a large validated capture with a reversible column transform. Restore the native row format."""

import argparse
import csv
from contextlib import ExitStack
import io
from pathlib import Path
import subprocess
import tarfile
import zlib

import numpy as np

from access import COLUMNS, IndexedStream


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def pack(folder, target):
    if not (folder / "capture_validation.json").is_file():
        raise ValueError("Pack only verified, closed runs")
    temp = target.with_suffix(".building")
    with (
        temp.open("wb") as dest,
        subprocess.Popen(
            ["zstd", "-q", "-3", "-T1", "-c"], stdin=subprocess.PIPE, stdout=dest
        ) as codec,
    ):
        with tarfile.open(fileobj=codec.stdin, mode="w|") as tar:
            for f in sorted(folder.iterdir()):
                if (
                    f.is_file()
                    and not f.name.endswith((".index.csv", ".u64le.gz", ".lock"))
                    and f.name not in {"capture_validation.json", "replica_blocks.json"}
                ):
                    tar.add(f, arcname=f.name, recursive=False)
            for name in COLUMNS:
                stream = IndexedStream(folder, name)
                for (first, count, _, _), block in zip(
                    stream.members, stream.chunks(), strict=True
                ):
                    columns = block.T.copy()
                    columns[:, 1:] -= columns[:, :-1].copy()
                    planes = (
                        columns.view("u1")
                        .reshape(stream.width, count, 8)
                        .transpose(0, 2, 1)
                    )
                    crc = zlib.crc32(block.tobytes())
                    add(tar, f"{name}/{first}/{crc}", planes.tobytes())
        codec.stdin.close()
        if codec.wait():
            raise ValueError("Transfer encoding failed")
    temp.replace(target)


def native_member(data):
    codec = zlib.compressobj(1, zlib.DEFLATED, 31)
    return codec.compress(data) + codec.flush()


def restore(source, folder):
    if (folder / "complete.json").exists():
        raise FileExistsError("Use a fresh staging folder for a restore")
    folder.mkdir(parents=True, exist_ok=True)
    complete = None
    with ExitStack() as stack:
        outputs, indexes, counts = {}, {}, {}
        for name in COLUMNS:
            outputs[name] = stack.enter_context(
                (folder / (name + ".u64le.gz")).open("wb")
            )
            index = stack.enter_context(
                (folder / (name + ".index.csv")).open("w", newline="")
            )
            indexes[name] = csv.writer(index, lineterminator="\n")
            indexes[name].writerow(
                ("first_row", "rows", "byte_offset", "compressed_bytes")
            )
            counts[name] = 0
        with subprocess.Popen(
            ["zstd", "-q", "-dc", source], stdout=subprocess.PIPE
        ) as codec:
            with tarfile.open(fileobj=codec.stdout, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        raise ValueError("Unexpected transfer member")
                    data = tar.extractfile(member).read()
                    parts = member.name.split("/")
                    if len(parts) == 1:
                        if parts[0] in (".", "..") or not parts[0].endswith(
                            (".csv", ".csv.gz", ".json", ".log", ".u16le.gz")
                        ):
                            raise ValueError("Unexpected capture metadata")
                        if parts[0] == "complete.json":
                            complete = data
                        else:
                            (folder / parts[0]).write_bytes(data)
                        continue
                    if len(parts) != 3 or parts[0] not in COLUMNS:
                        raise ValueError("Unknown native stream")
                    name, first, crc = parts
                    width = len(COLUMNS[name])
                    n, remainder = divmod(len(data), 8 * width)
                    if remainder or not 0 < n <= 32768 or int(first) != counts[name]:
                        raise ValueError("Incomplete or unordered transfer")
                    delta = (
                        np.frombuffer(data, dtype="u1")
                        .reshape(width, 8, n)
                        .transpose(0, 2, 1)
                        .copy()
                        .view("<u8")
                        .reshape(width, n)
                    )
                    raw = np.cumsum(delta, axis=1, dtype="<u8").T.copy().tobytes()
                    if zlib.crc32(raw) != int(crc):
                        raise ValueError("Restored row block differs from source")
                    chunk = native_member(raw)
                    out = outputs[name]
                    indexes[name].writerow((counts[name], n, out.tell(), len(chunk)))
                    out.write(chunk)
                    counts[name] += n
            # Consume tar's final padding so the decoder exits normally.
            while codec.stdout.read(1 << 20):
                pass
            if codec.wait():
                raise ValueError("Transfer decompression failed")
        for name in COLUMNS:
            if counts[name] == 0:
                outputs[name].write(native_member(b""))
    if complete is None:
        raise ValueError("Missing run metadata")
    (folder / "complete.json").write_bytes(complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pack", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    (pack if args.action == "pack" else restore)(args.source, args.destination)
