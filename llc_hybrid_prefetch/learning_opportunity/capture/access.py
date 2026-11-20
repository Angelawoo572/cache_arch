"""Read one bounded range from the native capture using its gzip member index."""

from bisect import bisect_right
import csv
import gzip
import json
from pathlib import Path

import numpy as np

COLUMNS = {
    "arrival": ("key", "pc", "physical_line", "cycle"),
    "ingress_state": ("order", "pressure"),
    "source_load": ("key", "fetch_cycle"),
    "cache": ("order", "key", "cycle", "physical_line", "code"),
    "offer": (
        "order",
        "anchor_key",
        "request_id",
        "target_line",
        "cycle",
        "code",
        "consumer_key",
    ),
    "head_wait": ("key", "retire_cycle", "exposed_cycles"),
    "prefetch_victim": ("request_id", "victim_line", "target_line", "cycle"),
}
TRACE_DTYPE = np.dtype(
    [
        ("pc", "<u8"),
        ("branch", "u1"),
        ("taken", "u1"),
        ("dst_reg", "u1", (2,)),
        ("src_reg", "u1", (4,)),
        ("store_va", "<u8", (2,)),
        ("load_va", "<u8", (4,)),
    ]
)


class IndexedStream:
    """A seekable table. Row bounds use zero-based, half-open indices."""

    def __init__(self, folder, name, record_checks=False):
        self.checks = {} if record_checks else None
        self.width = len(COLUMNS[name])
        self.path = Path(folder) / (name + ".u64le.gz")
        with self.path.with_name(name + ".index.csv").open(newline="") as file:
            self.members = [
                tuple(
                    int(row[k])
                    for k in ("first_row", "rows", "byte_offset", "compressed_bytes")
                )
                for row in csv.DictReader(file)
            ]
        row_pos = byte_pos = 0
        for first, count, offset, size in self.members:
            if first != row_pos or offset != byte_pos or count <= 0 or size <= 0:
                raise ValueError(f"Invalid member index in {self.path}")
            row_pos += count
            byte_pos += size
        self.rows = row_pos
        self.starts = [row[0] for row in self.members]
        if self.rows and byte_pos != self.path.stat().st_size:
            raise ValueError(f"Incomplete member index in {self.path}")
        if not self.rows and gzip.decompress(self.path.read_bytes()):
            raise ValueError(f"Nonempty payload with empty index in {self.path}")

    def chunks(self, start=0, stop=None):
        stop = self.rows if stop is None else stop
        if not 0 <= start <= stop <= self.rows:
            raise ValueError("Row interval outside stream")
        if start == stop:
            return
        with self.path.open("rb") as source:
            i = bisect_right(self.starts, start) - 1
            source.seek(self.members[i][2])
            for first, count, _, size in self.members[i:]:
                if first >= stop:
                    break
                member = source.read(size)
                data = gzip.decompress(member)
                if self.checks is not None:
                    self.checks[first] = (
                        count,
                        int.from_bytes(member[-8:-4], "little"),
                    )
                block = np.frombuffer(data, dtype="<u8").reshape(count, self.width)
                yield block[max(0, start - first) : min(count, stop - first)]

    def read(self, start, stop):
        blocks = list(self.chunks(start, stop))
        return (
            np.concatenate(blocks) if blocks else np.empty((0, self.width), dtype="<u8")
        )


def source_records(folder, cache_root, keys):
    """Join operand keys to the original control/register/virtual-address records for offline analysis."""
    config = json.loads((Path(folder) / "complete.json").read_text())["config"]
    trace = np.memmap(
        Path(cache_root) / config["traces"][0], mode="r", dtype=TRACE_DTYPE
    )
    keys = np.asarray(keys, dtype="<u8")
    ordinal = keys >> 2
    origin = config.get("source_ordinal_base", 0)
    if np.any(ordinal <= origin) or np.any(ordinal > len(trace) + origin):
        raise ValueError("Source ordinal outside the materialized trace")
    records = trace[ordinal - origin - 1]
    va = np.take_along_axis(
        records["load_va"], (keys & 3).astype("intp")[..., None], axis=-1
    )[..., 0]
    if np.any(va == 0):
        raise ValueError("Operand key joins to an inactive source slot")
    return records, va
