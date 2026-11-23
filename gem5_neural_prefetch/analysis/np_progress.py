#!/usr/bin/env python3
"""How far along is a capture that is still being written?

    python analysis/np_progress.py RUN_DIR [RUN_DIR ...]

Counts the load rows already on disk and the simulated cycles they span.
"""

import sys
import zlib
from pathlib import Path

import numpy as np

PERIOD = 250  # ticks per cycle at 4 GHz


def rows_so_far(path, width):
    """Decode every complete gzip member already on disk."""
    raw = path.read_bytes()
    parts = []
    while raw:
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            chunk = inflater.decompress(raw)
        except zlib.error:
            break
        if not inflater.eof:
            break
        parts.append(chunk)
        raw = inflater.unused_data
    data = np.frombuffer(b"".join(parts), dtype="<u8")
    return data.reshape(-1, width)


def main():
    for run in sys.argv[1:]:
        timing = rows_so_far(Path(run) / "capture" / "load_time.u64le.gz", 9)
        if not len(timing):
            print(f"{Path(run).name}: no complete member yet")
            continue
        cycles = (int(timing[-1, 7]) - int(timing[0, 7])) / PERIOD
        uops = rows_so_far(Path(run) / "capture" / "uop.u64le.gz", 5) if (Path(run) / "capture" / "uop.u64le.gz").exists() else []
        print(
            f"{Path(run).name}: loads {len(timing):9d}  cycles {cycles:13.0f}  "
            f"loads per kilocycle {1000 * len(timing) / max(cycles, 1):6.2f}  uops {len(uops)}"
        )


if __name__ == "__main__":
    main()
