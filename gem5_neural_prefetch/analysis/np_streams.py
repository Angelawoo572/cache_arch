"""Read the capture streams of DATA_COLLECTION.md into NumPy arrays."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

COLUMNS = {
    "uop": ("pc", "meta", "src", "dst", "dst_value"),
    "load": ("uop", "pc", "vaddr", "size_flags", "value"),
    "store": ("uop", "pc", "vaddr", "size_flags", "value"),
    "load_time": (
        "seq",
        "paddr",
        "t_fetch",
        "t_rename",
        "t_issue",
        "t_data",
        "t_head",
        "t_commit",
        "service",
    ),
    "cache": ("tick", "code", "paddr_line", "vaddr_line", "who", "pc"),
    "fill_content": ("tick", "paddr_line", "vaddr_line")
    + tuple(f"w{i}" for i in range(8)),
}

# Bits of the S6 code column.
EV_HIT, EV_MISS, EV_FILL = 1, 2, 4
F_PREFETCHED = 1 << 8  # hit: first use of a prefetched line
F_HWPF = 1 << 9  # the packet is a hardware prefetch
F_WRITE = 1 << 10
F_INST = 1 << 11
F_SWPF = 1 << 12
F_EVICT = 1 << 13  # an eviction or writeback announced by the level above


class Stream:
    """One table. Rows are zero-based; row ranges are half-open."""

    def __init__(self, folder, name):
        self.name = name
        self.columns = COLUMNS[name]
        self.width = len(self.columns)
        self.path = Path(folder) / f"{name}.u64le.gz"
        with (Path(folder) / f"{name}.index.csv").open(newline="") as file:
            self.members = [
                {key: int(value) for key, value in row.items()}
                for row in csv.DictReader(file)
            ]
        self.rows = sum(m["rows"] for m in self.members)

    def read(self, start=0, stop=None):
        """Rows [start, stop) as an array of shape (rows, width)."""
        stop = self.rows if stop is None else min(stop, self.rows)
        parts = []
        with self.path.open("rb") as file:
            for member in self.members:
                first, count = member["first_row"], member["rows"]
                if first + count <= start or first >= stop:
                    continue
                file.seek(member["byte_offset"])
                raw = gzip.decompress(file.read(member["compressed_bytes"]))
                block = np.frombuffer(raw, dtype="<u8").reshape(-1, self.width)
                if len(block) != count:
                    raise ValueError(f"{self.path}: member at row {first} is short")
                parts.append(block[max(start - first, 0) : stop - first])
        if not parts:
            return np.empty((0, self.width), dtype="<u8")
        return np.concatenate(parts)

    def column(self, name):
        return self.read()[:, self.columns.index(name)]

    def table(self):
        """Every column as a named array."""
        data = self.read()
        return {name: data[:, i] for i, name in enumerate(self.columns)}


class Capture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.counts = json.loads((self.folder / "capture_counts.json").read_text())

    def has(self, name):
        return (self.folder / f"{name}.index.csv").exists()

    def stream(self, name):
        return Stream(self.folder, name)

    def table(self, name):
        return self.stream(name).table()


def cache_fields(code):
    """Split the S6 code column."""
    code = np.asarray(code)
    return {
        "level": (code & 0xF).astype(np.int8),
        "event": ((code >> 4) & 0xF).astype(np.int8),
        "prefetched": (code & F_PREFETCHED) != 0,
        "hwpf": (code & F_HWPF) != 0,
        "write": (code & F_WRITE) != 0,
        "inst": (code & F_INST) != 0,
        "swpf": (code & F_SWPF) != 0,
        "evict": (code & F_EVICT) != 0,
        "requestor": ((code >> 16) & 0xFFFF).astype(np.int32),
    }
