"""Extract exact instruction-record prefixes from a cached original or a public HTTP stream."""
import argparse
import fcntl
import json
import lzma
import os
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from matrix import PREFIX, ROOT, TRACE_DIR, workloads

BASE = "https://traces.rbera.com/dpc4/SPEC17/"
MIRROR = "https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/"
RECORD_BYTES = 64
CHUNK = 4 << 20
HEADERS = {"User-Agent": "Mozilla/5.0 (public DPC4 trace research)"}


def download(filename):
    """Request only the leading bytes needed to decode the instruction prefix."""
    offset = 0
    while True:
        last = None
        for attempt in range(4):
            base = BASE if attempt < 2 else MIRROR
            url = base + filename + f"?prefix_offset={offset}"
            request = urllib.request.Request(url, headers={
                **HEADERS, "Range": f"bytes={offset}-{offset + CHUNK - 1}"})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    span = response.headers.get("Content-Range", "")
                    if response.status != 206 or not span.startswith(f"bytes {offset}-"):
                        raise ValueError(f"expected byte range starting at {offset}, got {span}")
                    payload = response.read(CHUNK)
                break
            except (OSError, ValueError) as exc:
                last = exc
                time.sleep(1 + attempt)
        else:
            raise OSError(f"failed to fetch {filename} at {offset}: {last}")
        if not payload:
            return
        offset += len(payload)
        yield payload


def extract(name, archives, records, transfers):
    with (TRACE_DIR / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return extract_locked(name, archives, records, transfers)


def extract_locked(name, archives, records, transfers):
    target = TRACE_DIR / (name + ".champsimtrace")
    meta = target.with_suffix(".source.json")
    size = records * RECORD_BYTES
    if target.exists() and meta.exists() and target.stat().st_size == size:
        return
    filename = name + ".champsimtrace.xz"
    cached = archives / filename
    saved = transfers / (filename + ".prefix")
    source = cached if cached.is_file() else saved if saved.is_file() else None
    if source is not None:
        fileobj = source.open("rb")
        compressed = iter(lambda: fileobj.read(CHUNK), b"")
    else:
        fileobj = None
        compressed = download(filename)
    packet = target.with_suffix(".part")
    transfer_tmp = saved.with_suffix(".part")
    decoder = lzma.LZMADecompressor()
    remaining = size
    feed = None
    try:
        if source is None:
            feed = transfer_tmp.open("wb")
        with packet.open("wb") as output:
            for block in compressed:
                if feed:
                    feed.write(block)
                while remaining:
                    out = decoder.decompress(block, max_length=min(4 << 20, remaining))
                    block = b""
                    output.write(out)
                    remaining -= len(out)
                    if decoder.eof or decoder.needs_input:
                        break
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"{filename}: {remaining} uncompressed bytes missing")
        with packet.open("rb") as f:
            records_test = f.read(RECORD_BYTES * 4096)
        data = list(struct.iter_unpack("<Q8B6Q", records_test))
        if not any(rec[0] != 0 for rec in data) or any(rec[1] > 1 or rec[2] > 1 for rec in data):
            raise ValueError(f"prefix of {filename} fails instruction record sanity checks")
        if feed:
            feed.close()
            os.replace(transfer_tmp, saved)
        os.replace(packet, target)
        meta.write_text(json.dumps(dict(original=filename, url=BASE + filename,
                         records=records, record_bytes=RECORD_BYTES,
                         used_local_archive=source == cached,
                         already_saved_transport=source == saved), indent=2) + "\n")
        print(f"{name}: ready ({records:,} records)", flush=True)
    finally:
        if fileobj:
            fileobj.close()
        if feed:
            feed.close()
        for partial in (packet, transfer_tmp):
            if partial.exists():
                partial.unlink()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--archives", type=Path, default=ROOT.parent / "traces")
    args.add_argument("--jobs", type=int, default=4)
    opts = args.parse_args()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    transfers = ROOT / "raw" / "transport"
    transfers.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=opts.jobs) as executor:
        list(executor.map(lambda name: extract(name, opts.archives, PREFIX, transfers), workloads()))
