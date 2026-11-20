"""Count aligned first-use gains and losses for one fixed-work replay."""

import gzip
import numpy as np
from select_targets import RUNS

ROI = 200_000_000
CHUNK = 262_144


def turnover(case, folder, targets, policy):
    paths = (RUNS / (case + "__nopf"), RUNS / (case + "__" + policy), folder)
    streams = [gzip.open(p / "outcomes_core0.u16le.gz", "rb") for p in paths]
    gained = lost = 0
    statuses = {}
    try:
        for offset in range(0, ROI, CHUNK):
            n = min(CHUNK, ROI - offset)
            mats = [
                np.frombuffer(stream.read(2 * n), dtype="<u2") for stream in streams
            ]
            assert all(len(x) == n for x in mats)
            start = 50_000_001 + offset
            for instr, op in targets:
                if start <= instr < start + n:
                    statuses[instr, op] = tuple(
                        int(x[instr - start]) >> (4 * op) & 15 for x in mats
                    )
            # Keep all four source operands independent.
            for op in range(4):
                no, spp, treatment = ((x >> (op * 4)) & 15 for x in mats)
                base = no == 2
                gained += np.count_nonzero(base & (spp != 9) & (treatment == 9))
                lost += np.count_nonzero(base & (spp == 9) & (treatment != 9))
        assert all(s.read(1) == b"" for s in streams)
        assert set(targets) == set(statuses)
    finally:
        for stream in streams:
            stream.close()
    return gained, lost, statuses
