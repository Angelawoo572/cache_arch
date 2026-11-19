"""Test the binary stream validator and check saved views against their frozen provenance."""
import argparse
import csv
import gzip
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import prepare

from fetch import HERE, RAW

RECORD = struct.Struct("<Q8B6Q")

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def fixtures():
    def item(pc, branch=0, taken=0, stores=(0, 0), loads=(0, 0, 0, 0)):
        return RECORD.pack(pc, branch, taken, 8, 0, 7, 0, 0, 0, *stores, *loads)
    data = [item(0x1001, stores=(0x3900, 0), loads=(0x2400, 0x5970, 0, 0)),
            item(0x1234, 1, 1, loads=(0x3900, 0, 0, 0)), item(0x9000)]
    with tempfile.TemporaryDirectory(dir=RAW) as scratch:
        scratch = Path(scratch)
        output, seed = scratch / "view", scratch / "seed"
        command = [str(RAW / "verify_records"), str(output), "2", "1", str(seed)]
        actual = subprocess.run(command, input=b"".join(data), capture_output=True, check=True)
        assert output.read_bytes() == data[-1]
        assert seed.read_bytes() == struct.pack("<4Q", 1, 3, 2, 5)
        assert actual.stdout.decode().strip().split(",")[:9] == ["3","1","1","1","2","2","1","0","0"]
        for bad in [b"".join(data)[:-1], item(0), item(16, 0, 1), item(16, 2, 1)]:
            assert subprocess.run(command, input=bad, capture_output=True).returncode != 0
        (scratch / "verify_records").symlink_to(RAW / "verify_records")
        archive = scratch / "fixture.gz"
        payload = gzip.compress(b"".join(data))
        archive.write_bytes(payload)
        source = dict(case="fixture", manifest_path=archive.name, compressed_bytes=len(payload))
        view = dict(view="slice", start="1", count="1")
        with patch.object(prepare, "RAW", scratch), patch.object(prepare, "HERE", scratch):
            prepare.prepare(source, view)
            native = scratch / "slice.champsimtrace"
            assert native.read_bytes() == data[1]
            damaged = bytearray(payload)
            damaged[-8] ^= 1  # Corrupt the gzip CRC without changing the published size.
            archive.write_bytes(damaged)
            try:
                prepare.prepare(source, view)
            except RuntimeError:
                pass
            else:
                raise AssertionError("accepted a corrupt compressed stream")
            assert native.read_bytes() == data[1]
    print("Record fields, invalid records, ordered VPNs, sliced decode and corrupt-stream rejection passed.")

def online(sources):
    def fetch(url):
        return subprocess.check_output(["curl", "-fsSL", "--retry", "3", "--max-time", "40", url], text=True)
    manifest = [line.split("\t") for line in fetch("https://traces.rbera.com/dpc4/manifest.txt").splitlines() if not line.startswith("#")]
    listed = {v[0]: int(v[1]) for v in manifest if len(v) >= 2 and v[1].isdigit()}
    metadata = json.loads(fetch("https://zenodo.org/api/records/10959705"))
    listed.update({f["key"]: f["size"] for f in metadata["files"]})
    for row in sources:
        if listed[row["manifest_path"]] != int(row["compressed_bytes"]):
            raise ValueError("Published byte length differs: " + row["case"])
    print(f"Publisher metadata matched for all {len(sources)} frozen source entries.")

def artifacts(sources, views):
    by_case = {r["case"]: r for r in sources}
    for v in views:
        case = v["case"]
        row, = rows(HERE / (v["view"] + ".validation.csv"))
        assert row["case"] == case
        assert int(row["materialized_start"]) == int(v["start"])
        assert int(row["materialized_records"]) == int(v["count"])
        assert int(v["start"]) + int(v["count"]) <= int(row["source_records"])
        source = by_case[case]
        assert int(source["verified_source_records"]) == int(row["source_records"])
        archive = RAW / source["manifest_path"].split("/")[-1]
        assert archive.stat().st_size == int(row["compressed_bytes"]) == int(source["compressed_bytes"])
        assert (RAW / (v["view"]+".champsimtrace")).stat().st_size == 64 * int(v["count"])
        if int(v["start"]):
            data = (RAW / (v["view"]+".prefix_vpns.u64le")).read_bytes()
            assert len(data) == int(row["prefix_pages"]) * 8
            assert len(set(struct.iter_unpack("<Q",data))) == int(row["prefix_pages"])
    print(f"Verified {len(sources)} complete archives and {len(views)} native windows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="compare publisher sizes")
    parser.add_argument("--artifacts", action="store_true", help="check all prepared working windows")
    args = parser.parse_args()
    sources, views = rows(HERE / "selection.csv"), rows(HERE / "views.csv")
    fixtures()
    if args.online:
        online(sources)
    if args.artifacts:
        artifacts(sources,views)
