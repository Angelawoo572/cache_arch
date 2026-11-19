"""Compile and functionally trace a real freestanding C function, independently checking its returned value and memory stream."""

import csv, ctypes, gzip, json, platform, struct, subprocess, sys
from pathlib import Path
from cases import RAW, TRACES, specs

sys.path.insert(0, str(RAW / "tools"))
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_GRP_JUMP, CS_GRP_RET, CS_GRP_CALL
from elftools.elf.elffile import ELFFile
from unicorn import (
    Uc,
    UC_ARCH_X86,
    UC_MODE_64,
    UC_HOOK_CODE,
    UC_HOOK_MEM_READ,
    UC_HOOK_MEM_WRITE,
    UC_MEM_READ,
)
from unicorn.x86_const import UC_X86_REG_RDI, UC_X86_REG_RSI, UC_X86_REG_RSP, UC_X86_REG_RAX

BASE = 0x20000000
CODE = 0x400000
STACK = 0x700000
STOP = 0x4F0000
ROWS = 192
PACK = struct.Struct("<Q8B6Q")
REG = {}
for code, aliases in [
    (1, "rax eax ax al ah"),
    (2, "rbx ebx bx bl bh"),
    (3, "rcx ecx cx cl ch"),
    (4, "rdx edx dx dl dh"),
    (5, "rsi esi si sil"),
    (6, "rsp esp sp spl"),
    (7, "rdi edi di dil"),
    (8, "rbp ebp bp bpl"),
    (25, "rflags eflags flags"),
    (26, "rip eip ip"),
]:
    for alias in aliases.split():
        REG[alias] = code
for i in range(8, 16):
    for suffix in ("", "d", "w", "b"):
        REG["r" + str(i) + suffix] = i + 1


def fixed(xs, size):
    regs = list(dict.fromkeys(xs))
    if len(regs) > size:
        raise ValueError("trace format capacity exceeded")
    return regs + [0] * (size - len(regs))


def main():
    spec = next(s for s in specs() if s["case"] == "compiled_rows")
    src = Path(__file__).with_name("compiled_rows.c")
    obj = RAW / "compiled_rows.o"
    common = [
        "clang",
        "-O1",
        "-fomit-frame-pointer",
        "-fno-unroll-loops",
        "-fno-vectorize",
        "-fno-slp-vectorize",
    ]
    cross = common + [
        "-target",
        "x86_64-unknown-none-elf",
        "-ffreestanding",
        "-fno-pic",
        "-fno-stack-protector",
        "-fno-asynchronous-unwind-tables",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    subprocess.run(cross, check=True)
    with obj.open("rb") as f:
        elf = ELFFile(f)
        if any(s.name.startswith(".rel") for s in elf.iter_sections()):
            raise ValueError("unexpected code relocation")
        code = elf.get_section_by_name(".text").data()
        entry = elf.get_section_by_name(".symtab").get_symbol_by_name("compiled_rows")[0][
            "st_value"
        ]
    native = RAW / ("compiled_rows.dylib" if platform.system() == "Darwin" else "compiled_rows.so")
    subprocess.run(common + ["-shared", "-fPIC", str(src), "-o", str(native)], check=True)
    words = (ctypes.c_uint64 * (ROWS * 32 * 8))()
    for i in range(len(words)):
        words[i] = i * 17 + 29
    library = ctypes.CDLL(str(native))
    fn = library.compiled_rows
    fn.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32)
    fn.restype = ctypes.c_uint64
    reference = fn(words, ROWS)
    machine = Uc(UC_ARCH_X86, UC_MODE_64)
    machine.mem_map(CODE, 4096)
    machine.mem_write(CODE, code)
    machine.mem_map(BASE, ctypes.sizeof(words))
    machine.mem_write(BASE, bytes(words))
    machine.mem_map(STACK, 8192)
    sp = STACK + 8192 - 8
    machine.mem_write(sp, struct.pack("<Q", STOP))
    for reg, val in [(UC_X86_REG_RDI, BASE), (UC_X86_REG_RSI, ROWS), (UC_X86_REG_RSP, sp)]:
        machine.reg_write(reg, val)
    decoder = Cs(CS_ARCH_X86, CS_MODE_64)
    decoder.detail = True
    metadata = {}
    for inst in decoder.disasm(code, CODE):
        br = any(inst.group(t) for t in (CS_GRP_JUMP, CS_GRP_RET, CS_GRP_CALL))
        rd, wr = inst.regs_access()
        reads = [] if inst.mnemonic == "nop" else [REG[inst.reg_name(r)] for r in rd]
        writes = [] if inst.mnemonic == "nop" else [REG[inst.reg_name(r)] for r in wr]
        if br:
            reads.append(26)
            writes.append(26)
        metadata[inst.address] = (inst.size, br, fixed(writes, 2), fixed(reads, 4))
    raw = TRACES / "compiled_rows.champsimtrace"
    label = TRACES / "compiled_rows.labels.csv.gz"
    prev = []
    num = 0
    loadnum = 0
    observed = []
    with raw.open("wb") as out, gzip.open(str(label), "wt") as lab:
        w = csv.writer(lab)
        w.writerow(
            (
                "cpu",
                "instr_id",
                "operand",
                "load_ordinal",
                "measurement",
                "label",
                "group_index",
                "position",
                "length",
                "previous_length",
                "virtual_line",
                "cross4k",
                "pc",
            )
        )

        def flush(nextpc):
            nonlocal num, loadnum
            if not prev:
                return
            pc, reads, writes = prev
            size, br, reg_w, reg_r = metadata[pc]
            num += 1
            out.write(
                PACK.pack(
                    pc,
                    int(br),
                    int(br and nextpc != pc + size),
                    *reg_w,
                    *reg_r,
                    *fixed(writes, 2),
                    *fixed(reads, 4),
                )
            )
            for operand, va in enumerate(dict.fromkeys(reads)):
                if BASE <= va < BASE + ctypes.sizeof(words):
                    if (va - BASE) % 64:
                        raise ValueError("unexpected non-line-aligned C input")
                    row, col = divmod((va - BASE) // 64, 32)
                    span = (8, 24)[row % 2]
                    vline = va // 64
                    phase = (
                        "warmup"
                        if num <= spec["warmup"]
                        else ("roi" if num <= spec["warmup"] + spec["roi"] else "tail")
                    )
                    w.writerow(
                        (
                            0,
                            num,
                            operand,
                            loadnum,
                            phase,
                            "row_start" if col == 0 else "row_body",
                            row,
                            col,
                            span,
                            0 if row == 0 else (8, 24)[(row - 1) % 2],
                            vline,
                            int(bool(observed) and vline // 64 != observed[-1] // 64),
                            pc,
                        )
                    )
                    observed.append(vline)
                    loadnum += 1

        def step(uc, address, size, user):
            flush(address)
            prev[:] = [address, [], []]

        def memory(uc, access, address, size, value, user):
            prev[1 if access == UC_MEM_READ else 2].append(address)

        machine.hook_add(UC_HOOK_CODE, step)
        machine.hook_add(UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE, memory)
        machine.emu_start(CODE + entry, STOP, count=2_000_000)
        flush(STOP)
    if machine.reg_read(UC_X86_REG_RAX) != reference:
        raise ValueError("native and emulated results differ")
    expected = [BASE // 64 + 32 * r + c for r in range(ROWS) for c in range((8, 24)[r % 2])]
    if observed != expected or raw.stat().st_size != 64 * num:
        raise ValueError("dynamic trace verification failed")
    if num < spec["warmup"] + spec["roi"] + 1:
        raise ValueError("too short")
    compiler = subprocess.check_output(["clang", "--version"], text=True).splitlines()[0]
    (TRACES / "compiled_rows.source.json").write_text(
        json.dumps(
            dict(
                origin="real compiled C executed by Unicorn",
                compiler=compiler,
                arguments=cross,
                records=num,
                dynamic_data_loads=loadnum,
                row_count=ROWS,
                native_and_emulated_return=reference,
                pc_count=len(metadata),
                static_text_bytes=len(code),
                versions={"unicorn": "2.1.4", "capstone": "5.0.7", "pyelftools": "0.32"},
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "native return and dynamic loads verified:",
        num,
        "instructions,",
        loadnum,
        "kernel input loads",
    )


if __name__ == "__main__":
    main()
