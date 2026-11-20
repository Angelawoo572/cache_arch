"""Collect a matched executable trace of one BFS kernel on two named SNAP graphs."""

import argparse
import csv
import gzip
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RAW = HERE.parent / "raw/graph_inputs"
sys.path.insert(0, str(PROJECT / "opportunity_study/workloads"))
from record_compiled import (
    CODE,
    STACK,
    STOP,
    PACK,
    REG,
    fixed,
    ELFFile,
    Cs,
    CS_ARCH_X86,
    CS_MODE_64,
    CS_GRP_JUMP,
    CS_GRP_RET,
    CS_GRP_CALL,
    Uc,
    UC_ARCH_X86,
    UC_MODE_64,
    UC_HOOK_CODE,
    UC_HOOK_MEM_READ,
    UC_HOOK_MEM_WRITE,
    UC_MEM_READ,
)
from unicorn.x86_const import (
    UC_X86_REG_RDI,
    UC_X86_REG_RSI,
    UC_X86_REG_RDX,
    UC_X86_REG_RCX,
    UC_X86_REG_R8,
    UC_X86_REG_RAX,
    UC_X86_REG_RSP,
)

NAMES = {"snap_stanford": "web-Stanford", "snap_google": "web-Google"}
BUDGET = 400_000


def record(case):
    source = NAMES[case]
    archive = RAW / (source + ".txt.gz")
    url = f"https://snap.stanford.edu/data/{source}.txt.gz"
    if not archive.exists():
        subprocess.run(
            ["curl", "-fLsS", "--retry", "3", url, "-o", str(archive)], check=True
        )
    with gzip.open(archive, "rt") as stream:
        pairs = np.loadtxt(stream, dtype=np.uint32)
    # Preserve vertex IDs and the original neighbor order within each adjacency list.
    source_edges = len(pairs)
    pairs = np.concatenate((pairs, pairs[:, ::-1]))
    n = int(pairs.max()) + 1
    degree = np.bincount(pairs[:, 0], minlength=n)
    root = int(degree.argmax())
    offset = np.r_[0, degree.cumsum()].astype("<u4")
    neighbor = pairs[np.argsort(pairs[:, 0], kind="stable"), 1].astype("<u4")
    seen = np.zeros(n, np.uint8)
    queue = np.zeros(n, "<u4")
    queue[0], seen[root] = root, 1
    arrays = [offset, neighbor, seen, queue]

    obj = RAW / (case + ".o")
    subprocess.run(
        [
            "clang",
            "-O1",
            "-fomit-frame-pointer",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-unroll-loops",
            "-target",
            "x86_64-unknown-none-elf",
            "-ffreestanding",
            "-fno-pic",
            "-fno-stack-protector",
            "-c",
            str(HERE / "bfs.c"),
            "-o",
            str(obj),
        ],
        check=True,
    )
    with obj.open("rb") as f:
        elf = ELFFile(f)
        code = elf.get_section_by_name(".text").data()
        assert (
            elf.get_section_by_name(".symtab").get_symbol_by_name("bfs")[0]["st_value"]
            == 0
        )
        assert not any(s.name.startswith(".rel") for s in elf.iter_sections())
    machine = Uc(UC_ARCH_X86, UC_MODE_64)
    machine.mem_map(CODE, 4096)
    machine.mem_write(CODE, code)
    machine.mem_map(STACK, 8192)
    machine.reg_write(UC_X86_REG_RSP, STACK + 4096)
    machine.mem_write(STACK + 4096, struct.pack("<Q", STOP))
    for i, (arr, reg) in enumerate(
        zip(arrays, (UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RDX, UC_X86_REG_RCX))
    ):
        address = (i + 1) * 0x10000000
        machine.mem_map(address, (arr.nbytes + 4095) & ~4095)
        machine.mem_write(address, arr.tobytes())
        machine.reg_write(reg, address)
    machine.reg_write(UC_X86_REG_R8, BUDGET)

    decoder = Cs(CS_ARCH_X86, CS_MODE_64)
    decoder.detail = True
    metadata = {}
    for op in decoder.disasm(code, CODE):
        branch = any(op.group(g) for g in (CS_GRP_JUMP, CS_GRP_RET, CS_GRP_CALL))
        rd, wr = op.regs_access()
        inputs = [] if op.mnemonic == "nop" else [REG[op.reg_name(r)] for r in rd]
        outputs = [] if op.mnemonic == "nop" else [REG[op.reg_name(r)] for r in wr]
        if branch:
            inputs.append(26)
            outputs.append(26)
        metadata[op.address] = (op.size, branch, fixed(outputs, 2), fixed(inputs, 4))
    count = 0
    current = []
    tmp = RAW / (case + ".partial")
    with tmp.open("wb") as trace:

        def flush(next_pc):
            nonlocal count
            if current:
                pc, reads, writes = current
                size, branch, wr, rd = metadata[pc]
                trace.write(
                    PACK.pack(
                        pc,
                        branch,
                        branch and next_pc != pc + size,
                        *wr,
                        *rd,
                        *fixed(writes, 2),
                        *fixed(reads, 4),
                    )
                )
                count += 1

        def step(uc, pc, size, user):
            flush(pc)
            current[:] = [pc, [], []]

        def memory(uc, access, address, size, value, user):
            current[1 if access == UC_MEM_READ else 2].append(address)

        machine.hook_add(UC_HOOK_CODE, step)
        machine.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, memory)
        machine.emu_start(CODE, STOP)
        flush(STOP)

    expected = [root]
    touched = {root}
    examined = 0
    for v in expected:
        for dest in neighbor[offset[v] : offset[v + 1]]:
            key = int(dest)
            if key not in touched:
                touched.add(key)
                expected.append(key)
            examined += 1
            if examined == BUDGET:
                break
        if examined == BUDGET:
            break
    returned = machine.reg_read(UC_X86_REG_RAX)
    actual = np.frombuffer(machine.mem_read(0x40000000, 4 * returned), "<u4")
    assert np.array_equal(actual, expected) and returned == len(touched)
    assert count * 64 == tmp.stat().st_size and count > 2_251_024
    tmp.replace(RAW / (case + ".champsimtrace"))
    with (HERE / (case + ".csv")).open("w") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(
            (
                "case",
                "graph_url",
                "input_edges",
                "root",
                "visited",
                "examined_edges",
                "recorded_instructions",
            )
        )
        out.writerow((case, url, source_edges, root, returned, examined, count))
    print(case, count, "instructions", examined, "edges", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=NAMES)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    record(args.case)
