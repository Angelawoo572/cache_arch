"""Read a compressed single-core admission stream. Ranks start at zero."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "raw/labels"
HORIZONS = (1, 2, 4, 8, 16, 32, 64)
SCOPES = (
    "global",
    "same_pc",
    "other_pc",
    "same_page",
    "other_page",
    "same_region",
    "other_region",
)
FIELDS = ("idop", "pc", "line", "cycle")


class Sequence:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "index.json").read_text())
        self.size = self.meta["observations"]
        self.chunk = self.meta["chunk_rows"]
        self.chunk_ends = np.array([c[3] for c in self.meta["chunks"]], dtype="<u8")
        self.keys = {}
        self.first = {}
        self.ends = {}
        self.ranks = {}
        for key in ("pc", "line", "page", "region"):
            self.keys[key] = self._column(key, "keys")
            self.first[key] = self._column(key, "first")
            if key != "line":
                self.ends[key] = self._column(key, "ends")
                self.ranks[key] = self._column(key, "ranks")

    def _column(self, key, name):
        return np.load(self.path / f"{key}_{name}.npy", mmap_mode="r")

    def _bounds(self, anchor, start, stop):
        stop = self.size if stop is None else stop
        if not 0 <= start <= anchor < stop <= self.size:
            raise ValueError("Anchor must lie inside the requested half-open interval")
        return stop

    @lru_cache(maxsize=12)
    def _events(self, block):
        with np.load(self.path / f"events_{block:04d}.npz") as data:
            return np.column_stack([data[k] for k in FIELDS])

    @lru_cache(maxsize=12)
    def _labels(self, block):
        with np.load(self.path / f"outcomes_{block:04d}.npz") as data:
            return data["masks"]

    def _take(self, ranks, loader, shape, dtype):
        indices = np.asarray(ranks, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Event rank outside observed stream")
        out = np.empty((len(indices), *shape), dtype=dtype)
        blocks = indices // self.chunk
        for block in np.unique(blocks):
            mask = blocks == block
            out[mask] = loader(int(block))[indices[mask] % self.chunk]
        return out

    def observations(self, ranks):
        """Return only (source operand key, PC, physical line, cycle), in requested order."""
        return self._take(ranks, self._events, (4,), "<u8")

    def outcomes(self, ranks):
        """Return retrospective operand nibbles in NoPF, Stride, Multi-delta, SPP order."""
        masks = self._take(ranks, self._labels, (), "<u2")
        return ((masks[:, None] >> np.arange(0, 16, 4, dtype="<u2")) & 15).astype("u1")

    def observed_before(self, key, values, anchor):
        """Has this physical line/page/region or PC appeared at or before the anchor?"""
        keys = self.keys[key]
        values = np.asarray(values, dtype=keys.dtype)
        ix = keys.searchsorted(values)
        valid = ix < len(keys)
        ix = np.minimum(ix, len(keys) - 1)
        return valid & (keys[ix] == values) & (self.first[key][ix] <= anchor)

    def _group(self, anchor, key):
        _, pc, line, _ = self._events(anchor // self.chunk)[anchor % self.chunk]
        value = pc if key == "pc" else int(line) >> (6 if key == "page" else 12)
        group = self.keys[key].searchsorted(np.uint64(value))
        lo = int(self.ends[key][group - 1]) if group else 0
        hi = int(self.ends[key][group])
        ranks = self.ranks[key][lo:hi]
        return ranks, int(ranks.searchsorted(np.uint32(anchor)))

    @staticmethod
    def _outside(ranks, index):
        # Excluded rank r[j] has r[j] - j nonmembers before it.
        lo, hi = 0, len(ranks)
        while lo < hi:
            mid = (lo + hi) // 2
            if int(ranks[mid]) - mid <= index:
                lo = mid + 1
            else:
                hi = mid
        return index + lo

    def relative(self, anchor, steps, scope="global", start=0, stop=None):
        """Resolve signed event offsets. -1 marks a censored position."""
        stop = self._bounds(anchor, start, stop)
        steps = np.asarray(steps, dtype=np.int64)
        if np.any(steps == 0) or steps.ndim != 1 or scope not in SCOPES:
            raise ValueError(
                "Use nonzero relative event offsets and a registered scope"
            )
        if scope == "global":
            selected = anchor + steps
        else:
            relation, key = scope.split("_", 1)
            group, pos = self._group(anchor, key)
            if relation == "same":
                idx = pos + steps
                selected = np.where(
                    (idx >= 0) & (idx < len(group)),
                    group[np.clip(idx, 0, len(group) - 1)].astype(np.int64),
                    -1,
                )
            else:
                before = anchor - pos
                idx = before + steps - (steps > 0)
                selected = np.array(
                    [
                        self._outside(group, i)
                        if 0 <= i < self.size - len(group)
                        else -1
                        for i in idx
                    ],
                    dtype=np.int64,
                )
        return np.where((selected >= start) & (selected < stop), selected, -1)

    def history(self, anchor, length, scope="global", start=0):
        """Prior events only. The triggering observation is available via observations([anchor])."""
        if length < 1:
            raise ValueError("History length must be positive")
        ranks = self.relative(
            anchor, np.arange(-length, 0), scope, start=start, stop=anchor + 1
        )
        ranks = ranks[ranks >= 0]
        return ranks, self.observations(ranks)

    def context(self, anchor, length, scope="global", start=0):
        """Causal sequence including the trigger, with signed physical and temporal gaps."""
        ranks, _ = self.history(anchor, length, scope, start=start)
        ranks = np.append(ranks, anchor)
        obs = self.observations(ranks)
        return {
            "ranks": ranks,
            "source_keys": obs[:, 0],
            "pc_sequence": obs[:, 1],
            "physical_lines": obs[:, 2],
            "pages": obs[:, 2] >> 6,
            "offsets": obs[:, 2] & 63,
            "cycles": obs[:, 3],
            "line_deltas": np.diff(obs[:, 2].astype(np.int64)),
            "cycle_gaps": np.diff(obs[:, 3]),
            "rank_gaps": np.diff(ranks),
        }

    def pressure(self, anchor, run=None):
        """Read counters latched at this admission from a richer matched capture."""
        import sys
        from pathlib import Path

        capture = Path(__file__).resolve().parents[1] / "capture"
        if str(capture) not in sys.path:
            sys.path.insert(0, str(capture))
        from access import IndexedStream

        folder = (
            Path(run)
            if run is not None
            else ROOT.parent
            / "capture/runs"
            / f"{self.meta['case']}__{self.meta['policy']}"
        )
        row = IndexedStream(folder, "arrival").read(anchor, anchor + 1)
        if not np.array_equal(row, self.observations([anchor])):
            raise ValueError(
                "State and sequence differ at this source operand, line or cycle"
            )
        order, packed = map(
            int, IndexedStream(folder, "ingress_state").read(anchor, anchor + 1)[0]
        )
        names = ("llc_mshr", "llc_rq", "llc_pq", "dram_rq", "dram_wq", "llc_wq")
        return {
            "observer_order": order,
            **{key: (packed >> (8 * i)) & 255 for i, key in enumerate(names)},
        }

    def future(self, anchor, horizons=HORIZONS, scope="global", stop=None):
        """Offline target records and matched outcomes. Missing horizons retain an explicit flag."""
        horizons = np.asarray(horizons, dtype=np.int64)
        if horizons.ndim != 1 or not len(horizons) or np.any(horizons <= 0):
            raise ValueError("Future horizons must be positive")
        ranks = self.relative(anchor, horizons, scope, stop=stop)
        valid = ranks >= 0
        rows = self.observations(ranks[valid])
        return {
            "ranks": ranks,
            "valid": valid,
            "observations": rows,
            "outcomes": self.outcomes(ranks[valid]),
            "lead_cycles": rows[:, 3] - self.observations([anchor])[0, 3],
            "line_seen": self.observed_before("line", rows[:, 2], anchor),
            "page_seen": self.observed_before("page", rows[:, 2] >> 6, anchor),
            "region_seen": self.observed_before("region", rows[:, 2] >> 12, anchor),
        }

    def batches(
        self,
        start,
        stop,
        *,
        history=32,
        horizons=HORIZONS,
        batch_size=2048,
        history_start=None,
    ):
        """Yield global arrival windows for training, with observation and target tensors separate.

        History ends at the anchor. A missing prefix or right-censored label is zero-padded
        and carries an explicit validity mask. No feature or label crosses its supplied bound.
        """
        if not 0 <= start < stop <= self.size or min(history, batch_size) <= 0:
            raise ValueError("Invalid batch interval or dimensions")
        lower = start if history_start is None else history_start
        if not 0 <= lower <= start:
            raise ValueError("History must start at or before the first anchor")
        steps = np.asarray(horizons, dtype=np.int64)
        if steps.ndim != 1 or not len(steps) or np.any(steps <= 0):
            raise ValueError("Batch horizons must be positive")
        for lo in range(start, stop, batch_size):
            anchors = np.arange(lo, min(lo + batch_size, stop))
            begin = max(lower, int(anchors[0]) - history + 1)
            end = min(stop, int(anchors[-1] + steps.max() + 1))
            ranks = np.arange(begin, end)
            data = self.observations(ranks)
            past = anchors[:, None] - np.arange(history - 1, -1, -1)
            past_valid = past >= lower
            observed = data[np.maximum(past - begin, 0)].copy()
            observed[~past_valid] = 0
            future = anchors[:, None] + steps[None, :]
            valid = future < stop
            indices = np.minimum(future - begin, len(data) - 1)
            targets = data[indices].copy()
            target_status = self.outcomes(ranks)[indices].copy()
            targets[~valid] = 0
            target_status[~valid] = 0
            yield {
                "observations": {
                    "anchor_ranks": anchors,
                    "history": observed,
                    "history_valid": past_valid,
                },
                "labels": {
                    "future": targets,
                    "outcomes": target_status,
                    "target_valid": valid,
                },
            }

    def time_window(self, anchor, cycles, stop=None):
        """All subsequent admissions through anchor cycle + cycles, including equal-cycle events."""
        stop = self._bounds(anchor, 0, stop)
        if cycles < 0:
            raise ValueError("Cycle window must be nonnegative")
        deadline = int(self.observations([anchor])[0, 3]) + cycles
        block = int(self.chunk_ends.searchsorted(np.uint64(deadline), side="right"))
        if block == len(self.chunk_ends):
            end = self.size
        else:
            col = self._events(block)[:, 3]
            end = block * self.chunk + int(
                col.searchsorted(np.uint64(deadline), side="right")
            )
        ranks = np.arange(anchor + 1, min(stop, end), dtype=np.int64)
        rows = self.observations(ranks)
        # A later observed cycle witnesses completeness, including all equal-cycle arrivals.
        complete = int(self.observations([stop - 1])[0, 3]) > deadline
        return dict(
            ranks=ranks,
            observations=rows,
            lines=np.unique(rows[:, 2]),
            outcomes=self.outcomes(ranks),
            complete=complete,
        )
