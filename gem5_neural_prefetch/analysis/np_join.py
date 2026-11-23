"""Per-load outcomes in one arm, and the join of two arms by dynamic load.

Every committed load has the same index in every arm (label L0), so two arms
are joined row by row. Inside one arm, cache events are attached to loads
through gem5's sequence number.
"""

import json
from pathlib import Path

import numpy as np

from np_streams import EV_FILL, EV_HIT, EV_MISS, Capture, cache_fields

UNSET = np.uint64(0xFFFFFFFFFFFFFFFF)


def run_info(run_dir):
    return json.loads((Path(run_dir) / "m5out" / "run_info.json").read_text())


class Arm:
    """Timing and cache outcomes of every committed load in one run."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.info = run_info(run_dir)
        self.period = self.info["clock_period_ticks"]
        self.capture = Capture(self.run_dir / "capture")
        self.counts = self.capture.counts

        timing = self.capture.table("load_time")
        self.n = len(timing["seq"])
        self.seq = timing["seq"]
        self.t_issue = timing["t_issue"].astype(np.int64)
        self.t_data = timing["t_data"].astype(np.int64)
        self.t_head = timing["t_head"].astype(np.int64)
        self.t_commit = timing["t_commit"].astype(np.int64)
        self.via_cache = (timing["service"] & 1) != 0

        known = (timing["t_data"] != 0) & (timing["t_issue"] != UNSET)
        self.latency = np.where(known, self.t_data - self.t_issue, 0) / self.period
        self.stall = (
            np.where(timing["t_data"] != 0, np.maximum(self.t_data - self.t_head, 0), 0)
            / self.period
        )

        events = self.capture.table("cache")
        fields = cache_fields(events["code"])
        self.events = events
        self.fields = fields
        data = ~fields["inst"] & ~fields["evict"]
        demand = data & ~fields["hwpf"] & ~fields["swpf"]

        order = np.argsort(self.seq, kind="stable")
        sorted_seq = self.seq[order]
        position = np.minimum(np.searchsorted(sorted_seq, events["who"]), self.n - 1)
        owner = np.where(sorted_seq[position] == events["who"], order[position], -1)
        self.event_load = owner

        self.miss = {}
        self.pf_hit = {}
        self.late = {}
        for level in (1, 2, 3):
            at = fields["level"] == level
            miss_rows = demand & at & (fields["event"] == EV_MISS) & (owner >= 0)
            flag = np.zeros(self.n, dtype=bool)
            flag[owner[miss_rows]] = True
            self.miss[level] = flag

            hit_rows = (
                demand
                & at
                & (fields["event"] == EV_HIT)
                & fields["prefetched"]
                & (owner >= 0)
            )
            flag = np.zeros(self.n, dtype=bool)
            flag[owner[hit_rows]] = True
            self.pf_hit[level] = flag

            # A demand miss is late when the next fill of its line at this
            # level answers a prefetch: the prefetch was already in flight.
            fill_rows = np.flatnonzero(data & at & (fields["event"] == EV_FILL))
            flag = np.zeros(self.n, dtype=bool)
            if len(fill_rows) and miss_rows.any():
                rows = np.flatnonzero(miss_rows)
                # One sortable key per event: a dense line id above 40 bits
                # of tick, so "first fill of this line at or after the miss"
                # is a single binary search.
                t0 = events["tick"].min()
                lines = np.concatenate(
                    (events["paddr_line"][fill_rows], events["paddr_line"][rows])
                )
                _, dense = np.unique(lines, return_inverse=True)
                dense = dense.astype(np.uint64)
                fill_key = (dense[: len(fill_rows)] << np.uint64(40)) | (
                    events["tick"][fill_rows] - t0
                )
                miss_key = (dense[len(fill_rows) :] << np.uint64(40)) | (
                    events["tick"][rows] - t0
                )
                by = np.argsort(fill_key, kind="stable")
                fill_key = fill_key[by]
                is_pf = fields["hwpf"][fill_rows][by]
                pos = np.searchsorted(fill_key, miss_key, side="left")
                inside = pos < len(fill_key)
                pos = np.minimum(pos, len(fill_key) - 1)
                same_line = (fill_key[pos] >> np.uint64(40)) == (
                    miss_key >> np.uint64(40)
                )
                late_rows = rows[inside & same_line & is_pf[pos]]
                flag[owner[late_rows]] = True
            self.late[level] = flag

    def service_level(self):
        """1 = no L1D miss, 2 = served by L2, 3 = by the LLC, 4 = by memory.
        A load that merged into another load's miss keeps the level its own
        events show, so use latency when the exact source matters."""
        level = np.ones(self.n, dtype=np.int8)
        level[self.miss[1]] = 2
        level[self.miss[2]] = 3
        level[self.miss[3]] = 4
        return level

    def window_cycles(self):
        return (self.counts["finish_tick"] - self.counts["enable_tick"]) / self.period


def fast_outcome(fast, level):
    """Outcome in a prefetching arm, for loads that own a miss at `level` in
    the arm without prefetching. 0 timely first use of a prefetched line,
    1 late, 2 miss again, 3 anything else."""
    again = np.zeros(fast.n, dtype=bool)
    for deeper in range(level, 4):
        again |= fast.miss[deeper]
    late = np.zeros(fast.n, dtype=bool)
    timely = np.zeros(fast.n, dtype=bool)
    for upper in range(1, level):
        late |= fast.late[upper]
        timely |= fast.pf_hit[upper]
    late |= fast.late[level] if level < 3 else False
    outcome = np.full(fast.n, 3, dtype=np.int8)
    outcome[timely & ~late & ~again] = 0
    outcome[again & ~late] = 2
    outcome[late] = 1
    return outcome


OUTCOME_NAMES = ("timely", "late", "miss", "other")
