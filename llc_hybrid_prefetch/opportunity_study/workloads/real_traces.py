"""Acquire canonical DPC4 graph recordings and an adjacent region from an existing DPC3 SPEC archive."""

import gzip, io, json, lzma, struct, os, time
import urllib.request as u
from concurrent.futures import ThreadPoolExecutor
from cases import ROOT, TRACES

BASE = "https://traces.rbera.com/dpc4/Graph/GAP/"


class Ranges(io.RawIOBase):
    """Read leading public bytes on demand without retrieving entire recordings."""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.start = 0
        self.hold = memoryview(b"")
        self.exhausted = False

    def readable(self):
        return True

    def readinto(self, buffer):
        if not self.hold and not self.exhausted:
            end = self.start + (8 << 20) - 1
            r = u.Request(
                self.url + f"?workload_range={self.start}",
                headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={self.start}-{end}"},
            )
            for attempt in range(3):
                try:
                    with u.urlopen(r, timeout=60) as response:
                        bounds = response.headers.get("Content-Range", "")
                        if response.status != 206 or not bounds.startswith(f"bytes {self.start}-"):
                            raise ValueError("unexpected response range " + bounds)
                        payload = response.read()
                    if not payload:
                        raise EOFError("short public trace range")
                    break
                except (OSError, EOFError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            self.start += len(payload)
            self.exhausted = self.start == int(bounds.split("/")[1])
            self.hold = memoryview(payload)
        n = min(len(buffer), len(self.hold))
        buffer[:n] = self.hold[:n]
        self.hold = self.hold[n:]
        return n


def prepare(name):
    records = 20_100_000 if name == "605.mcf_s-994B" else 3_100_000
    if name.endswith("-long"):
        records = 52_100_000
    out = TRACES / (name + ".champsimtrace")
    meta = TRACES / (name + ".source.json")
    if out.is_file() and out.stat().st_size == records * 64 and meta.is_file():
        return
    if name.startswith("bfs"):
        url = BASE + name.removesuffix("-long") + ".trace.gz"
        remote = io.BufferedReader(Ranges(url), buffer_size=1 << 20)
        source = gzip.GzipFile(fileobj=remote)
    else:
        url = "https://traces.rbera.com/dpc4/SPEC17/" + name + ".champsimtrace.xz"
        archive = ROOT.parents[1] / "traces" / (name + ".champsimtrace.xz")
        remote = None
        if archive.is_file():
            source = lzma.open(archive)
        else:
            remote = io.BufferedReader(Ranges(url), buffer_size=1 << 20)
            source = lzma.LZMAFile(remote)
    tmp = out.with_suffix(".part")
    try:
        with source, tmp.open("wb") as sink:
            remaining = 64 * records
            while remaining:
                chunk = source.read(min(4 << 20, remaining))
                if not chunk:
                    raise EOFError("incomplete bounded prefix for " + name)
                sink.write(chunk)
                remaining -= len(chunk)
        probes = [0, records // 4, records // 2, records - 4096]
        with tmp.open("rb") as file:
            for pos in probes:
                file.seek(pos * 64)
                rows = list(struct.iter_unpack("<Q8B6Q", file.read(4096 * 64)))
                if any(not r[0] or r[1] not in (0, 1) or r[2] not in (0, 1) for r in rows):
                    raise ValueError("invalid standard 64-byte trace layout at " + str(pos))
        tmp.replace(out)
        meta.write_text(
            json.dumps(
                dict(source=url, records=records, record_bytes=64, validation_offsets=probes),
                indent=2,
            )
            + "\n"
        )
        print("ready " + name, flush=True)
    finally:
        if remote:
            remote.close()
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    TRACES.mkdir(parents=True, exist_ok=True)
    peer = TRACES / "619.lbm_s-3766B.champsimtrace"
    if not peer.exists():
        previous = ROOT.parent / "raw/traces" / peer.name
        if previous.is_file():
            peer.symlink_to(os.path.relpath(previous, peer.parent))
        else:
            prepare("619.lbm_s-3766B")
    peer_meta = TRACES / "619.lbm_s-3766B.source.json"
    if not peer_meta.exists():
        peer_meta.write_text(
            json.dumps(
                dict(
                    source="https://traces.rbera.com/dpc4/SPEC17/619.lbm_s-3766B.champsimtrace.xz",
                    records=peer.stat().st_size // 64,
                    record_bytes=64,
                    reused_previous_prefix=peer.is_symlink(),
                ),
                indent=2,
            )
            + "\n"
        )
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(prepare, ["bfs-3", "bfs-8", "605.mcf_s-994B", "bfs-3-long", "bfs-8-long"]))
