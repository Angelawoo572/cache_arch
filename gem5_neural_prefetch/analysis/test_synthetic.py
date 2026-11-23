#!/usr/bin/env python3
"""Directed test of the slicer and the relation fit on a hand-built trace.

Writes a small capture whose dependence structure is known by construction,
runs np_slice on it, fits relations and checks them. Needs a C++ compiler and
zlib. Run from anywhere:

    python analysis/test_synthetic.py
"""

import gzip
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from np_relations import fit, read_edges  # noqa: E402
from np_streams import Capture  # noqa: E402

INT = 0  # integer register class
NONE16 = 0xFFFF


def reg(index):
    return (INT << 12) | index


def pack(ids, slots):
    word = 0xFFFFFFFFFFFFFFFF
    for k, r in enumerate(ids[:slots]):
        word &= ~(0xFFFF << (16 * k))
        word |= r << (16 * k)
    return word


class Trace:
    def __init__(self):
        self.uop, self.load, self.store = [], [], []
        self.value = {}

    def op(self, pc, srcs, dst, result, load=None, store=None):
        meta = (len(srcs) << 32) | (1 << 40) | (1 << 23 if dst is not None else 0)
        if load is not None:
            meta |= 1 << 16
            vaddr, size = load
            self.load.append((len(self.uop), pc, vaddr, size, result))
        if store is not None:
            meta |= 1 << 17
            vaddr, size, stored = store
            self.store.append((len(self.uop), pc, vaddr, size, stored))
        dsts = [] if dst is None else [dst]
        self.uop.append((pc, meta, pack(srcs, 4), pack(dsts, 2), result if dst is not None else 0))
        if dst is not None:
            self.value[dst] = result

    def write(self, folder):
        for name, rows, width in (
            ("uop", self.uop, 5),
            ("load", self.load, 5),
            ("store", self.store, 5),
        ):
            data = np.array(rows, dtype="<u8").reshape(-1, width)
            raw = gzip.compress(data.tobytes())
            (folder / f"{name}.u64le.gz").write_bytes(raw)
            (folder / f"{name}.index.csv").write_text(
                "first_row,rows,byte_offset,compressed_bytes\n"
                f"0,{len(data)},0,{len(raw)}\n"
            )
        (folder / "capture_counts.json").write_text("{}\n")


def build():
    rng = np.random.default_rng(7)
    t = Trace()
    R_CUR, R_KEY, R_SUM, R_E, R_IDX, R_VAL, R_BASE, R_NB, R_SP, R_TMP = range(1, 11)

    # A linked list: key at +0, next at +16, nodes at random addresses.
    nodes = (rng.permutation(4000)[:600].astype(np.uint64) * 64 + 0x10000000).tolist()
    t.op(0x400000, [], R_CUR, nodes[0])
    for i, node in enumerate(nodes):
        t.op(0x400010, [reg(R_CUR)], reg(R_KEY), int(rng.integers(1, 1 << 40)), load=(node, 8))
        t.op(0x400014, [reg(R_SUM), reg(R_KEY)], reg(R_SUM), i)
        nxt = nodes[i + 1] if i + 1 < len(nodes) else 0
        t.op(0x400018, [reg(R_CUR)], reg(R_CUR), nxt, load=(node + 16, 8))

    # An index array feeding a value array: values[idx[e]], 4-byte indices.
    idx_base, val_base = 0x20000000, 0x30000000
    t.op(0x400100, [], reg(R_NB), idx_base)
    t.op(0x400104, [], reg(R_BASE), val_base)
    t.op(0x400108, [], reg(R_E), 0)
    for e in range(600):
        index = int(rng.integers(0, 1 << 20))
        t.op(0x400110, [reg(R_NB), reg(R_E)], reg(R_IDX), index, load=(idx_base + 4 * e, 4))
        t.op(0x400114, [reg(R_BASE), reg(R_IDX)], reg(R_VAL), 5, load=(val_base + 8 * index, 8))
        t.op(0x400118, [reg(R_E)], reg(R_E), e + 1)

    # A pointer spilled to the stack and reloaded before use.
    stack = 0x7FFF0000
    objects = (rng.permutation(4000)[:300].astype(np.uint64) * 64 + 0x50000000).tolist()
    table = 0x60000000
    for i, obj in enumerate(objects):
        t.op(0x400200, [], reg(R_TMP), table + 8 * i)
        t.op(0x400204, [reg(R_TMP)], reg(R_CUR), obj, load=(table + 8 * i, 8))
        t.op(0x400208, [reg(R_SP), reg(R_CUR)], None, 0, store=(stack, 8, obj))
        t.op(0x40020C, [], reg(R_CUR), 0)  # the register is reused
        t.op(0x400210, [reg(R_SP)], reg(R_KEY), obj, load=(stack, 8))
        t.op(0x400214, [reg(R_KEY)], reg(R_VAL), 9, load=(obj + 24, 8))
    return t


def main():
    scratch = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "np_synthetic"
    scratch.mkdir(parents=True, exist_ok=True)
    build().write(scratch)

    binary = scratch / "np_slice"
    subprocess.check_call(
        [os.environ.get("CXX", "c++"), "-O2", "-std=c++17", "-o", str(binary),
         str(HERE / "np_slice.cc"), "-lz"]
    )
    edges_path = scratch / "edges.u64le"
    subprocess.check_call([str(binary), str(scratch), str(edges_path)])

    load = Capture(scratch).table("load")
    rel = fit(load, read_edges(edges_path))
    table = rel["relations"]
    found = {
        (int(a), int(b), int(s), int(base))
        for a, b, s, base in zip(
            table["producer_pc"], table["consumer_pc"], table["shift"], table["base"]
        )
    }
    expected = {
        (0x400018, 0x400010, 0, 0),  # next pointer -> key of the next node
        (0x400018, 0x400018, 0, 16),  # next pointer -> next field of that node
        (0x400110, 0x400114, 3, 0x30000000),  # index -> values[index]
        (0x400204, 0x400214, 0, 24),  # table entry -> field, through the stack
        (0x400210, 0x400214, 0, 24),  # the reload itself is also a producer
    }
    missing = expected - found
    # The reload is the nearest producer, so the through-memory relation may
    # be shadowed by it. One of the two must be present.
    stack_ok = (0x400204, 0x400214, 0, 24) in found or (0x400210, 0x400214, 0, 24) in found
    missing -= {(0x400204, 0x400214, 0, 24), (0x400210, 0x400214, 0, 24)}
    pc = load["pc"]
    explained = lambda p: float(rel["derived"][pc == p].mean())
    print("relations found:")
    for row in sorted(found):
        print("  producer %#x -> consumer %#x  shift %d  base %#x" % row)
    print("explained share  key %.3f  next %.3f  values %.3f  field %.3f  index %.3f" % (
        explained(0x400010), explained(0x400018), explained(0x400114),
        explained(0x400214), explained(0x400110)))
    assert not missing, f"missing relations: {missing}"
    assert stack_ok, "the pointer that went through the stack was not traced"
    assert explained(0x400010) > 0.99 and explained(0x400018) > 0.99
    assert explained(0x400114) > 0.99 and explained(0x400214) > 0.99
    assert explained(0x400110) == 0.0, "the index array has no producer"
    field = np.flatnonzero(pc == 0x400214)
    origin_pc = pc[rel["origin"][field]]
    print("field loads whose origin is the table load: %.3f" % float((origin_pc == 0x400204).mean()))
    assert (origin_pc == 0x400204).mean() > 0.99, "origin did not follow the stack"
    print("PASS")


if __name__ == "__main__":
    main()
