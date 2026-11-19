"""Construct declared synthetic traces and separate offline ground truth."""

import argparse, csv, gzip, itertools, json, struct
from concurrent.futures import ThreadPoolExecutor
from cases import TRACES, specs

REC = struct.Struct("<Q8B6Q")
BASE = 0x20000000
PC = 0x400000


def noise(n):
    x = (n + 0x9E3779B9) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return x ^ (x >> 16)


def trace_events(name):
    # Every tuple is a virtual line, observable PC index, evaluator label, row/group and position.
    if name == "row_hierarchy_paced":
        previous = 0
        for slab in itertools.count():
            for band in range(3):
                band_base = slab * 2048 + (0, 320 + 64 * (slab % 2), 768 + 128 * (slab % 2))[band]
                row_base = band_base
                for row in range(4):
                    phase = (slab + 2 * band + row) % 4
                    size = (8, 20, 12, 24)[phase]
                    pitch = (32, 48, 40, 64)[phase]
                    for index in range(size):
                        if index:
                            label = "row_body"
                        elif row:
                            label = "subrow_start"
                        elif band:
                            label = "band_start"
                        else:
                            label = "slab_start"
                        yield row_base + index, 0, label, slab * 12 + band * 4 + row, index, size, previous
                    previous = size
                    row_base += pitch
        return
    if name in ("streams_three_switch_paced", "streams_three_identified_paced"):
        counts = [0, 0, 0]
        for n in itertools.count():
            phase = n // 48
            order = (0, 1, 2, 0, 2, 1) if phase % 2 == 0 else (2, 0, 1, 2, 1, 0)
            lane = order[n % 6]
            position = counts[lane]
            counts[lane] += 1
            address = lane * (1 << 21) + position * (lane + 1)
            mark = "schedule_switch" if n and n % 48 == 0 else "stream_" + str(lane)
            pc = lane if name == "streams_three_identified_paced" else 0
            yield address, pc, mark, phase, n % 48, 0, 0
        return
    if name in ("xor_context_paced", "xor_maskfirst_paced"):
        for n in itertools.count():
            trial, pos = divmod(n, 8)
            a, b = noise(2 * trial) & 1, noise(2 * trial + 1) & 1
            if pos == 0:
                yield trial * 64, 1 + (
                    a if name == "xor_context_paced" else 0
                ), "first_cue", trial, pos, 0, 0
            elif pos == 4:
                yield trial * 64 + 2, 3 + b, "second_cue", trial, pos, 0, 0
            elif pos == 7:
                yield trial * 64 + (8, 40)[a ^ b], 6, "joint_target", trial, pos, 0, 0
            else:
                yield (1 << 21) + trial * 64 + pos * 3, 5, "intervening", trial, pos, 0, 0
        return
    if name.startswith("row_"):
        last = 0
        for row in itertools.count():
            if name.startswith("row_period2"):
                length = (8, 24)[row % 2]
            elif name.startswith("row_period5"):
                length = (8, 20, 12, 24, 16)[row % 5]
            else:
                key = row // 2 if name == "row_previous_paced" else row
                length = (8, 12, 16, 24)[noise(key) % 4]
            for pos in range(length):
                label = "row_body"
                if pos == 0:
                    label = (
                        ("repeated_length_row_start" if row % 2 else "new_length_row_start")
                        if name == "row_previous_paced"
                        else "row_start"
                    )
                yield row * 32 + pos, 0, label, row, pos, length, last
            last = length
        return
    for n in itertools.count():
        if name.startswith("streams_"):
            lane = n % 2
            pos = n // 2
            x = lane * (1 << 21) + pos
            yield x, lane if name == "streams_pc_paced" else 0, "stream_" + str(
                lane
            ), lane, pos, 0, 0
        elif name.startswith("crosspc_"):
            trial, pos = divmod(n, 4)
            bit = noise(trial) & 1
            if pos == 0:
                yield trial * 64, (
                    1 + bit if name == "crosspc_cue_paced" else 1
                ), "cue", trial, pos, 0, 0
            elif pos == 2:
                yield trial * 64 + (8, 40)[bit], 3, "choice_target", trial, pos, 0, 0
            else:
                yield (1 << 22) + trial * 64 + pos * 5, 4, "intervening", trial, pos, 0, 0
        elif name == "motif_noise_paced":
            block, pos = divmod(n, 8)
            if pos in (1, 4):
                yield (1 << 22) + noise(n) % (1 << 18), 1, "noise", block, pos, 0, 0
            else:
                yield block * 32 + (0, 0, 11, 3, 0, 24, 7, 19)[
                    pos
                ], 0, "motif_target", block, pos, 0, 0
        elif name == "phase_flip_paced":
            block, pos = divmod(n, 64)
            off = pos if pos < 32 else 32 + 3 * (pos - 32)
            yield block * 128 + off, 0, (
                "phase_start" if pos % 32 == 0 else "interior"
            ), block, pos, 0, 0
        elif name.startswith("page_"):
            trial, pos = divmod(n, 32)
            row, col = divmod(pos, 16)
            off = trial * 128 + row * 32 + col + (32 if name == "page_cross_paced" else 0)
            mark = "pair_second_row" if pos == 16 else ("pair_start" if pos == 0 else "row_body")
            yield off, 0, mark, trial, pos, 16, 16
        elif name.startswith("ws_"):
            cap = 4096 if name == "ws_small" else 131072
            yield n % cap, 0, "revisit" if n >= cap else "first_pass", n // cap, n % cap, 0, 0
        else:
            yield n, 0, "stream", 0, n, 0, 0


def ins(ip, d=0, s=0, t=0, va=0, br=0):
    return REC.pack(ip, br, br, d, 0, s, t, 0, 0, 0, 0, va, 0, 0, 0)


def make(spec):
    name = spec["case"]
    stride = spec["period"]
    gap = stride - 4
    lanes = spec["chains"]
    nload = spec["warm_loads"] + spec["roi_loads"] + 64
    path = TRACES / (name + ".champsimtrace")
    tmp = path.with_suffix(".part")
    labels = TRACES / (name + ".labels.csv.gz")
    # Non-memory operations are prepacked. Distinct register chains change the
    # dependency graph while preserving PC and address order in the MLP controls.
    bodies = {}
    for lane in range(lanes):
        addr = 1 + lane
        load = 14 + lane
        address_step = ins(PC, d=addr, s=addr)
        after = bytearray(ins(PC + 8, d=addr, s=load))
        for j in range(gap):
            after.extend(ins(PC + 12 + 4 * j, d=addr, s=addr))
        after.extend(ins(PC + 12 + gap * 4, d=26, s=25, t=26, br=1))
        bodies[lane] = (addr, load, address_step, after)
    last_line = None
    with gzip.open(str(labels), "wt", compresslevel=1) as lab, tmp.open("wb") as dest:
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
        buf = bytearray()
        for n, (line, p, label, row, pos, l, prev) in enumerate(
            itertools.islice(trace_events(name), nload)
        ):
            lane = n % lanes
            r, d, pre, post = bodies[lane]
            pc = PC + 0xE00 + 4 * p
            va = BASE + line * 64
            vi = va // 64
            i = stride * n + 2
            phase = (
                "warmup"
                if n < spec["warm_loads"]
                else ("roi" if n < spec["warm_loads"] + spec["roi_loads"] else "tail")
            )
            w.writerow(
                (
                    0,
                    i,
                    0,
                    n,
                    phase,
                    label,
                    row,
                    pos,
                    l,
                    prev,
                    vi,
                    int(last_line is not None and last_line // 64 != vi // 64),
                    pc,
                )
            )
            last_line = vi
            buf.extend(pre)
            buf.extend(ins(pc, d=d, s=r, va=va))
            buf.extend(post)
            if len(buf) >= 1 << 20:
                dest.write(buf)
                buf.clear()
        dest.write(buf)
    assert tmp.stat().st_size == nload * stride * 64
    tmp.replace(path)
    (TRACES / (name + ".source.json")).write_text(
        json.dumps(dict(kind="generated dynamic trace", **spec), indent=2) + "\n"
    )
    print("ready " + name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", action="append")
    opts = p.parse_args()
    with ThreadPoolExecutor(3) as pool:
        list(
            pool.map(
                make,
                [
                    s
                    for s in specs()
                    if s["group"] == "synthetic" and (not opts.case or s["case"] in opts.case)
                ],
            )
        )
