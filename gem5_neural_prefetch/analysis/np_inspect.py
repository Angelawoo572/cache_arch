#!/usr/bin/env python3
"""Look inside one capture: event counts per cache level and the history of
one line.

    python analysis/np_inspect.py RUN_DIR [--line N | --pick LEVEL]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from np_join import run_info  # noqa: E402
from np_streams import EV_FILL, EV_HIT, EV_MISS, Capture, cache_fields  # noqa: E402

NAMES = {EV_HIT: "hit", EV_MISS: "miss", EV_FILL: "fill"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--line", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--pick", type=int, default=2, help="level to pick a line from")
    args = parser.parse_args()

    period = run_info(args.run_dir)["clock_period_ticks"]
    capture = Capture(Path(args.run_dir) / "capture")
    events = capture.table("cache")
    f = cache_fields(events["code"])
    print(f"cache rows {len(events['tick'])}")
    header = "level event     total     hwpf    evict  first_use     inst  with_seq"
    print(header)
    for level in (1, 2, 3):
        for event, name in NAMES.items():
            m = (f["level"] == level) & (f["event"] == event)
            print(
                "L%d    %-5s %9d %8d %8d %10d %8d %9d"
                % (
                    level,
                    name,
                    m.sum(),
                    (m & f["hwpf"]).sum(),
                    (m & f["evict"]).sum(),
                    (m & f["prefetched"]).sum(),
                    (m & f["inst"]).sum(),
                    (m & (events["who"] != 0)).sum(),
                )
            )

    line = args.line
    if line is None:
        fills = events["paddr_line"][(f["level"] == args.pick) & (f["event"] == EV_FILL)]
        if not len(fills):
            return
        line = int(fills[len(fills) // 2])
    rows = np.flatnonzero(events["paddr_line"] == line)
    print(f"history of line {line:#x}: {len(rows)} events")
    start = int(events["tick"][rows].min())
    for i in rows[:40]:
        flags = "".join(
            tag
            for tag, on in (
                (" prefetch", f["hwpf"][i]),
                (" evict", f["evict"][i]),
                (" first-use", f["prefetched"][i]),
                (" write", f["write"][i]),
                (" inst", f["inst"][i]),
            )
            if on
        )
        print(
            "  +%7d cycles  L%d %-4s seq %-10d requestor %d%s"
            % (
                (int(events["tick"][i]) - start) // period,
                f["level"][i],
                NAMES.get(int(f["event"][i]), "?"),
                int(events["who"][i]),
                f["requestor"][i],
                flags,
            )
        )


if __name__ == "__main__":
    main()
