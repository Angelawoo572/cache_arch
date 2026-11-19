"""Fetch complete bounded prefixes from the positive discovery allowlist."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import lzma
import subprocess
import zlib

from register import HERE, STUDY, write_csv

RECORDS = 801_024
TARGET = RECORDS * 64
CHUNK = 1024 * 1024
RAW = STUDY / "raw/intake"


def remote_prefix(row, dest, target=TARGET, scratch=RAW):
    """Validate byte ranges against the manifest before exposing a decoded prefix."""
    spec = int(row["compressed_bytes"])
    assert spec > CHUNK
    received = 0
    produced = 0
    codec = lzma.LZMADecompressor if row["manifest_path"].endswith(".xz") else lambda: zlib.decompressobj(31)
    decoder = codec()
    head = scratch / (row["case"] + ".headers")
    body = scratch / (row["case"] + ".range")
    with dest.open("wb") as out:
        while produced < target:
            end = min(spec - 1, received + CHUNK - 1)
            if end < received:
                raise ValueError("Trace ended early")
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                            "--max-time", "60", "--retry", "3", "--max-filesize", str(CHUNK),
                            "--range", f"{received}-{end}", "--dump-header", str(head),
                            "--output", str(body), row["source_url"]], check=True)
            headers = head.read_text().lower()
            expected = f"content-range: bytes {received}-{end}/{spec}"
            if expected not in headers.splitlines() or body.stat().st_size != end - received + 1:
                raise ValueError(f"{row['case']}: range mismatch, expected {expected}")
            data = body.read_bytes()
            body.unlink()
            received += len(data)
            while data:
                decoded = decoder.decompress(data, target - produced)
                out.write(decoded)
                produced += len(decoded)
                if produced == target:
                    break
                if decoder.eof:
                    data = decoder.unused_data
                    decoder = codec()
                else:
                    data = b""
    return received


def local_prefix(row, dest):
    src = STUDY / row["legacy_path"]
    provenance = json.loads(src.with_suffix(".source.json").read_text())
    assert row["source_url"] in (provenance.get("url"), provenance.get("source"))
    with src.open("rb") as f, dest.open("wb") as out:
        remaining = TARGET
        while remaining:
            block = f.read(min(CHUNK, remaining))
            if not block:
                raise ValueError(f"Local source too short: {src}")
            out.write(block)
            remaining -= len(block)


def get(row):
    if row["split"] != "discovery":
        raise ValueError("Input is outside the frozen discovery allowlist")
    dest = STUDY / row["prefix_path"]
    with dest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return install(row, dest)


def install(row, dest):
    meta = dest.with_suffix(".json")
    if dest.is_file() and meta.is_file():
        existing = json.loads(meta.read_text())
        if existing["source_url"] == row["source_url"] and dest.stat().st_size == TARGET and existing["records"] == RECORDS:
            return existing
        raise ValueError("Existing discovery prefix differs from registered input")
    tmp = dest.with_suffix(".partial")
    local = bool(row["legacy_path"] and (STUDY / row["legacy_path"]).is_file())
    if local:
        local_prefix(row, tmp)
        n = 0
    else:
        n = remote_prefix(row, tmp)
    if tmp.stat().st_size != TARGET:
        raise ValueError("Incomplete prefix")
    item = dict(case=row["case"], collection=row["collection"], source_url=row["source_url"],
                method="existing DPC4 prefix" if local else "bounded HTTP ranges",
                compressed_bytes_received=n, records=RECORDS, record_bytes=64)
    meta.write_text(json.dumps(item, indent=2) + "\n")
    tmp.replace(dest)
    print(row["case"], item["method"], n, flush=True)
    return item


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    with (HERE / "registry.csv").open() as f:
        selected = [r for r in csv.DictReader(f) if r["split"] == "discovery"]
    (RAW / "traces").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(get, sorted(selected, key=lambda r: not bool(r["legacy_path"]))))
    write_csv("acquisition.csv", sorted(results, key=lambda r: r["case"]))
