#!/usr/bin/env python3
"""Measurements M1 to M8 of the design memo for one workload window.

    python analysis/np_measure.py NOPF_RUN FAST_RUN EDGES OUT_JSON

NOPF_RUN and FAST_RUN are run folders of the same checkpoint and window.
EDGES is the np_slice output for the NOPF_RUN capture. No neural network and
no fitted predictor other than count tables is involved.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from np_join import OUTCOME_NAMES, Arm, aligned, fast_outcome  # noqa: E402
from np_relations import fit, read_edges  # noqa: E402
from np_streams import EV_FILL  # noqa: E402

LONG_LATENCY = 40.0  # cycles: at least an LLC access under this hierarchy


def pct(values, points=(50, 90, 99)):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {f"p{p}": None for p in points}
    return {f"p{p}": float(np.percentile(values, p)) for p in points}


def share(part, whole):
    return float(part) / float(whole) if whole else None


def previous_same(key):
    """Index of the previous row with the same key, or -1."""
    n = len(key)
    order = np.lexsort((np.arange(n), key))
    prev = np.full(n, -1, dtype=np.int64)
    same = key[order][1:] == key[order][:-1]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def mechanism_classes(load, relations, miss_l1):
    """For every load, which causal constructions name its address (M2)."""
    n = len(load["pc"])
    vaddr = load["vaddr"].astype(np.int64)

    # Same-PC delta: the last two steps of this PC were equal.
    p1 = previous_same(load["pc"])
    p2 = np.where(p1 >= 0, p1[np.maximum(p1, 0)], -1)
    have = p2 >= 0
    d1 = vaddr - vaddr[np.maximum(p1, 0)]
    d2 = vaddr[np.maximum(p1, 0)] - vaddr[np.maximum(p2, 0)]
    delta = have & (d1 == d2)

    # Offset on a known page: the page of one of the last 64 L1D-miss loads.
    page = load["vaddr"] >> np.uint64(12)
    miss_idx = np.flatnonzero(miss_l1)
    miss_rank = np.cumsum(miss_l1) - 1  # rank of the latest miss at or before i
    prev_miss_same_page = previous_same(page[miss_idx])
    known_page = np.zeros(n, dtype=bool)
    # For a missing load: an earlier miss on its page within 64 misses.
    recent = (prev_miss_same_page >= 0) & (
        np.arange(len(miss_idx)) - prev_miss_same_page <= 64
    )
    known_page[miss_idx[recent]] = True

    # Temporal repeat: this (previous miss line, this line) pair occurred before.
    line = load["vaddr"] >> np.uint64(6)
    pair = np.zeros(len(miss_idx), dtype=np.uint64)
    if len(miss_idx) > 1:
        pair[1:] = (line[miss_idx[:-1]] << np.uint64(32)) ^ line[miss_idx[1:]]
    repeat = np.zeros(n, dtype=bool)
    seen_before = previous_same(pair) >= 0
    seen_before[:1] = False
    repeat[miss_idx[seen_before]] = True

    return {
        "delta": delta,
        "value": relations["derived"],
        "known_page": known_page,
        "repeat": repeat,
    }


def cheapest(classes, depth_chain):
    """One class per load, in order of hardware cost."""
    n = len(classes["delta"])
    label = np.full(n, 5, dtype=np.int8)  # none
    label[classes["repeat"]] = 4
    label[classes["known_page"]] = 3
    label[classes["value"] & (depth_chain >= 2)] = 2
    label[classes["value"] & (depth_chain <= 1)] = 1
    label[classes["delta"]] = 0
    return label


CHEAPEST_NAMES = (
    "same_pc_delta",
    "value_depth_1",
    "value_depth_2_plus",
    "offset_on_known_page",
    "temporal_repeat",
    "none",
)


def chain_depth(relations, long_latency):
    """How many dependent long-latency loads in a row end at each load."""
    producer = relations["producer"]
    derived = relations["derived"]
    depth = np.zeros(len(producer), dtype=np.int32)
    for i in np.flatnonzero(derived):
        p = producer[i]
        depth[i] = 1 + (depth[p] if (derived[p] and long_latency[p]) else 0)
    return depth


def held_out_bits(context, label, train):
    """Bits per decision of a count table fitted on the `train` rows and
    scored on the others. Returns (with context, prior only)."""
    context = np.asarray(context)
    label = np.asarray(label, dtype=np.int64)
    k = int(label.max()) + 1
    _, ctx = np.unique(context, return_inverse=True)
    ctx = ctx.reshape(-1)
    n_ctx = int(ctx.max()) + 1
    table = np.zeros((n_ctx, k), dtype=np.float64)
    np.add.at(table, (ctx[train], label[train]), 1.0)
    prior = np.bincount(label[train], minlength=k) + 1.0
    prior /= prior.sum()
    # Back off to the prior with weight one.
    prob = (table + prior) / (table.sum(axis=1, keepdims=True) + 1.0)
    test_ctx, test_label = ctx[~train], label[~train]
    return float(-np.mean(np.log2(prob[test_ctx, test_label]))), float(
        -np.mean(np.log2(prior[test_label]))
    )


def online_bits(context, label):
    """Bits per decision of a count table that learns as it goes: every row
    is scored with the counts of the rows before it, so warm-up and phase
    changes are charged. Returns (with context, running prior only)."""
    label = np.asarray(label, dtype=np.int64)
    n = len(label)
    k = int(label.max()) + 1
    _, ctx = np.unique(np.asarray(context), return_inverse=True)
    ctx = ctx.reshape(-1).astype(np.int64)
    index = np.arange(n)

    def earlier_with_same(*keys):
        order = np.lexsort((index,) + tuple(reversed(keys)))
        start = np.zeros(n, dtype=bool)
        start[0] = True
        for key in keys:
            sorted_key = key[order]
            start[1:] |= sorted_key[1:] != sorted_key[:-1]
        position = index - np.maximum.accumulate(np.where(start, index, 0))
        rank = np.empty(n, dtype=np.int64)
        rank[order] = position
        return rank

    seen_label = earlier_with_same(label)
    seen_context = earlier_with_same(ctx)
    seen_both = earlier_with_same(ctx, label)
    prior = (seen_label + 1.0) / (index + k)
    prob = (seen_both + prior) / (seen_context + 1.0)
    return float(-np.mean(np.log2(prob))), float(-np.mean(np.log2(prior)))


def table_bits(context, label):
    """Both estimates for one decision: {estimate: (with context, prior)}.
    "interleaved" holds out alternate blocks of up to 4,096 decisions, so
    training and test share the program phase. "online" learns as it goes."""
    n = len(label)
    block = max(1, min(4096, n // 8))
    train = (np.arange(n) // block) % 2 == 0
    return {
        "interleaved": held_out_bits(context, label, train),
        "online": online_bits(context, label),
    }


def recent_outcomes(group, outcome):
    """For every row: the outcome of the previous row of its group, of the
    one before (2 = there is none), and how many of the previous eight were
    positive (9 = fewer than eight exist)."""
    n = len(group)
    order = np.lexsort((np.arange(n), group))
    g, y = group[order], outcome[order]
    start = np.ones(n, dtype=bool)
    start[1:] = g[1:] != g[:-1]
    position = np.arange(n) - np.maximum.accumulate(np.where(start, np.arange(n), 0))
    before = np.cumsum(y) - y  # positives earlier in the sorted array
    in_group = before - np.maximum.accumulate(np.where(start, before, 0))

    def lagged(k):
        value = np.full(n, 2, dtype=np.int64)
        value[k:] = np.where(position[k:] >= k, y[:-k], 2)
        return value

    count8 = np.full(n, 9, dtype=np.int64)
    count8[8:] = np.where(position[8:] >= 8, in_group[8:] - in_group[:-8], 9)
    result = []
    for sorted_values in (lagged(1), lagged(2), count8):
        values = np.empty(n, dtype=np.int64)
        values[order] = sorted_values
        result.append(values)
    return result


def value_hash(arm):
    """(what the hash covers, the hash). Captures written before the hashes
    were split carry one hash over load addresses and values together."""
    counts = arm.counts
    if "load_value_hash" in counts:
        return ("values", counts["load_value_hash"])
    return ("addresses and values", counts.get("v1_load_hash"))


def same_values(a, b):
    """Did two address-aligned arms read the same values? None when their
    captures hash different fields and cannot be compared."""
    (kind_a, hash_a), (kind_b, hash_b) = value_hash(a), value_hash(b)
    return hash_a == hash_b if kind_a == kind_b else None


class Window:
    """Everything about one window that does not depend on the arm."""

    def __init__(self, nopf_dir, edges_path):
        self.nopf = Arm(nopf_dir)
        self.load = self.nopf.capture.table("load")
        self.n = self.nopf.n
        edges = read_edges(edges_path)
        if len(edges) != self.n:
            raise SystemExit(f"edges cover {len(edges)} loads, capture has {self.n}")
        self.relations = fit(self.load, edges)
        self.classes = mechanism_classes(self.load, self.relations, self.nopf.miss[1])


def measure(window, fast_dir):
    nopf, fast = window.nopf, Arm(fast_dir)
    if not aligned(nopf, fast):
        raise SystemExit("V1 failed: the two arms committed different loads")
    load, n, relations = window.load, window.n, window.relations

    out = {
        "window": {
            "instructions": nopf.info["measured_instructions"],
            "loads": int(n),
            "uops": int(nopf.counts["committed_uops"]),
            "same_load_values_as_nopf": same_values(nopf, fast),
            "cycles_nopf": nopf.window_cycles(),
            "cycles_fast": fast.window_cycles(),
            "ipc_nopf": nopf.info["measured_instructions"] / nopf.window_cycles(),
            "ipc_fast": fast.info["measured_instructions"] / fast.window_cycles(),
        }
    }

    # ---- M1: residual opportunity ------------------------------------
    m1 = {}
    for level, name in ((2, "beyond_l2"), (3, "beyond_llc")):
        original = nopf.miss[level]
        outcome = fast_outcome(fast, level)
        row = {
            "original_misses": int(original.sum()),
            "stall_nopf": float(nopf.stall[original].sum()),
            "stall_fast_same_loads": float(fast.stall[original].sum()),
        }
        for code, label in enumerate(OUTCOME_NAMES):
            chosen = original & (outcome == code)
            row[label] = int(chosen.sum())
            row[f"stall_fast_{label}"] = float(fast.stall[chosen].sum())
        # What "other" is: the load found its line in L2, or in L1 where an
        # earlier load or prefetch had already brought and used it.
        other = original & (outcome == 3)
        row["other_found_in_l2"] = int((other & fast.miss[1]).sum())
        row["other_found_in_l1"] = int((other & ~fast.miss[1]).sum())
        m1[name] = row
    m1["stall_all_loads_nopf"] = float(nopf.stall.sum())
    m1["stall_all_loads_fast"] = float(fast.stall.sum())
    out["M1"] = m1

    # Residual targets: original misses beyond L2 that the fast path left
    # late or missed, weighted by the stall that remains.
    original = nopf.miss[2]
    outcome2 = fast_outcome(fast, 2)
    residual = original & ((outcome2 == 1) | (outcome2 == 2))
    weight = fast.stall
    # A producer counts as a link of a serial chain only if it still waits a
    # long time in this arm, with its conventional prefetchers running.
    long_latency = fast.latency >= LONG_LATENCY

    # ---- M2: mechanism census ----------------------------------------
    classes = window.classes
    depth = chain_depth(relations, long_latency)
    label = cheapest(classes, depth)
    m2 = {"targets": int(residual.sum()), "stall": float(weight[residual].sum())}
    for code, name in enumerate(CHEAPEST_NAMES):
        chosen = residual & (label == code)
        m2[name] = {
            "targets": int(chosen.sum()),
            "stall": float(weight[chosen].sum()),
        }
    m2["membership"] = {
        name: {
            "targets": int((residual & flag).sum()),
            "stall": float(weight[residual & flag].sum()),
        }
        for name, flag in classes.items()
    }
    # Of the value classes: targets named through a bit-vector index.
    bit_vector = relations["derived"] & (relations["shift"] < 0)
    m2["value_through_bit_vector"] = {
        "targets": int((residual & bit_vector).sum()),
        "stall": float(weight[residual & bit_vector].sum()),
    }
    m2["all_original_misses"] = {
        name: int((original & (label == code)).sum())
        for code, name in enumerate(CHEAPEST_NAMES)
    }
    out["M2"] = m2

    # ---- M3: lead budget for value-derived residual targets ----------
    target = residual & relations["derived"]
    idx = np.flatnonzero(target)
    nearest = relations["producer"][idx]
    # Lead is counted from the origin: the first load that carried the value,
    # which is where a traversal could start.
    prod = relations["origin"][idx]
    # Timing is taken in the prefetching arm: that is the machine a slow
    # path would be added to.
    period = fast.period
    slack = (fast.t_issue[idx] - fast.t_data[prod]) / period
    slack_nearest = (fast.t_issue[idx] - fast.t_data[nearest]) / period
    latency = fast.latency[idx]
    producer_long = long_latency[prod]
    producer_delta = classes["delta"][prod]
    w = weight[idx]
    total_w = float(w.sum())
    w_origin_differs = share(w[prod != nearest].sum(), total_w)
    m3 = {
        "targets": int(len(idx)),
        "stall": total_w,
        "slack_cycles": pct(slack),
        "slack_from_nearest_producer_cycles": pct(slack_nearest),
        "origin_differs_from_nearest_stall_share": w_origin_differs,
        "target_latency_cycles": pct(latency),
        "producer_is_long_latency": share(w[producer_long].sum(), total_w),
        "producer_address_is_delta_predictable": share(
            w[producer_delta].sum(), total_w
        ),
        "serial_chain_stall_share": share(
            w[producer_long & ~producer_delta].sum(), total_w
        ),
    }
    for frac in (0.25, 0.5, 1.0):
        m3[f"stall_share_slack_at_least_{int(frac * 100)}pct_of_latency"] = share(
            w[slack >= frac * latency].sum(), total_w
        )
    # Why slack exists where it does. The consumer may not have been fetched
    # yet when the value returned (it sits behind a branch that resolved
    # late), or the L1D cache may have had no free miss register.
    m3["stall_share_consumer_fetched_after_value"] = share(
        w[fast.t_fetch[idx] > fast.t_data[prod]].sum(), total_w
    )
    blocked = fast.statistic("board.cache_hierarchy.l1d.blockedCycles::no_mshrs")
    m3["l1d_blocked_no_miss_register_share_of_cycles"] = (
        None if blocked is None else blocked / fast.window_cycles()
    )
    # Upper bound on what one traversal hop can remove, with no compounding
    # along a chain: a request issued when the origin's data returns saves at
    # most the slack; if the origin's own address is delta-predictable the
    # root can be fetched early and the whole stall is in reach.
    reach = np.where(producer_delta, w, np.minimum(w, np.maximum(slack, 0.0)))
    m3["stall_share_removable_by_one_hop_bound"] = share(reach.sum(), total_w)
    m3["stall_removable_by_one_hop_bound"] = float(reach.sum())
    # Several hops: the stall is in reach of a bounded traversal if some
    # ancestor within D dependence hops has a delta-predictable address, so
    # that the traversal can start ahead of the program at that root.
    ancestor = prod.copy()
    reachable = np.zeros(len(idx), dtype=bool)
    alive = np.ones(len(idx), dtype=bool)
    for hops in (1, 2, 3, 4):
        reachable |= alive & classes["delta"][ancestor]
        m3[f"stall_share_with_delta_root_within_{hops}_hops"] = share(
            w[reachable].sum(), total_w
        )
        alive &= relations["derived"][ancestor] & ~reachable
        ancestor = np.where(alive, relations["origin"][ancestor], ancestor)
    m3["chain_depth"] = {
        str(k): share(w[depth[idx] == k].sum(), total_w) for k in (1, 2, 3)
    }
    m3["chain_depth"]["4_plus"] = share(w[depth[idx] >= 4].sum(), total_w)
    out["M3"] = m3

    # ---- M4: decision entropy with count tables -----------------------
    trigger = np.flatnonzero(fast.miss[1])
    is_producer = np.zeros(n, dtype=bool)
    is_producer[prod] = True
    m4 = {"triggers": int(len(trigger))}
    if len(trigger) > 1000:
        # The default trigger is a load that missed L1D. Does it start a
        # residual target, and how much of the residual starts there at all?
        gate = is_producer[trigger].astype(np.int64)
        m4["gate_positive_rate"] = float(gate.mean())
        for estimate, (bits, bits_prior) in table_bits(load["pc"][trigger], gate).items():
            m4[f"gate_bits_prior.{estimate}"] = bits_prior
            m4[f"gate_bits_given_pc.{estimate}"] = bits
        m4["residual_stall_whose_origin_missed_l1d"] = share(
            w[fast.miss[1][prod]].sum(), total_w
        )
    # The gate as a relation table would face it: at every load whose address
    # some relation names, is that load a residual target? Contexts are
    # nested: the relation, the cache level that served the producer, and the
    # outcomes of the relation's last instances.
    consumers = np.flatnonzero(relations["derived"])
    if len(consumers) > 1000:
        rel_all = relations["relation"][consumers].astype(np.int64)
        outcome = residual[consumers].astype(np.int64)
        producer_all = relations["producer"][consumers]
        level = fast.miss[1][producer_all].astype(np.int64) + fast.miss[2][producer_all]
        m4["relation_instances"] = int(len(consumers))
        m4["relation_instance_positive_rate"] = float(outcome.mean())
        last1, last2, last8 = recent_outcomes(rel_all, outcome)
        contexts = {
            "relation": rel_all,
            "relation_producer_level": rel_all * 3 + level,
            "relation_last_1": rel_all * 3 + last1,
            "relation_last_2": (rel_all * 3 + last1) * 3 + last2,
            "relation_last_8_count": rel_all * 10 + last8,
            "relation_producer_level_last_2": ((rel_all * 3 + level) * 3 + last1) * 3 + last2,
        }
        for name, context in contexts.items():
            for estimate, (bits, bits_prior) in table_bits(context, outcome).items():
                m4[f"instance_gate_bits_given_{name}.{estimate}"] = bits
                m4[f"instance_gate_bits_prior.{estimate}"] = bits_prior
    if len(idx) > 1000:
        # Which static relation does a value-derived residual target use,
        # given its own PC, and given the PC of the load a traversal starts at.
        rel = relations["relation"][idx]
        _, rel_dense = np.unique(rel, return_inverse=True)
        for name, context in (("consumer_pc", load["pc"][idx]), ("origin_pc", load["pc"][prod])):
            for estimate, (bits, bits_prior) in table_bits(context, rel_dense).items():
                m4[f"relation_bits_prior.{estimate}"] = bits_prior
                m4[f"relation_bits_given_{name}.{estimate}"] = bits
    out["M4"] = m4

    # ---- M5: relation reuse -------------------------------------------
    m5 = {"static_relations_all_loads": int(len(relations["relations"].get("shift", [])))}
    if len(idx):
        rel = relations["relation"][idx]
        uniq, inv = np.unique(rel, return_inverse=True)
        rel_weight = np.bincount(inv, weights=w, minlength=len(uniq))
        order = np.argsort(-rel_weight)
        cumulative = np.cumsum(rel_weight[order]) / max(total_w, 1e-12)
        m5["relations_used_by_residual_targets"] = int(len(uniq))
        for frac in (0.5, 0.9, 0.99):
            m5[f"relations_for_{int(frac * 100)}pct_of_stall"] = int(
                np.searchsorted(cumulative, frac) + 1
            )
        top = uniq[order[:12]]
        table = relations["relations"]
        m5["top"] = [
            {
                "producer_pc": hex(int(table["producer_pc"][r])),
                "consumer_pc": hex(int(table["consumer_pc"][r])),
                "depth": int(table["depth"][r]),
                "shift": int(table["shift"][r]),
                "base": hex(int(table["base"][r])),
                "instances": int(table["instances"][r]),
                "stall_share": float(rel_weight[order[k]] / max(total_w, 1e-12)),
            }
            for k, r in enumerate(top)
        ]
    out["M5"] = m5

    # ---- M6: trigger rate and bursts ----------------------------------
    m6 = {}
    if len(idx):
        # Loads at the PCs where a traversal for a residual target would
        # start. This is the rate at which a relation table is consulted.
        starts = np.isin(load["pc"], np.unique(load["pc"][prod]))
        m6["loads_at_origin_pcs_per_kilocycle"] = 1000.0 * int(starts.sum()) / fast.window_cycles()
        m6["origin_pcs"] = int(len(np.unique(load["pc"][prod])))
    for name, arm in (("nopf", nopf), ("fast", fast)):
        t = np.sort(arm.t_data[arm.miss[1] & (arm.t_data > 0)])
        gaps = np.diff(t) / arm.period
        fields = arm.fields
        l2_fills = int(
            ((fields["level"] == 2) & (fields["event"] == EV_FILL) & ~fields["inst"]).sum()
        )
        m6[name] = {
            "l1d_miss_loads_per_kilocycle": 1000.0 * len(t) / arm.window_cycles(),
            "l2_fills_per_kilocycle": 1000.0 * l2_fills / arm.window_cycles(),
            "gap_cycles": pct(gaps, (10, 50, 90)),
            "share_gaps_under_4_cycles": share((gaps < 4).sum(), len(gaps)),
            "share_gaps_under_16_cycles": share((gaps < 16).sum(), len(gaps)),
        }
    out["M6"] = m6

    # ---- M7: operand age ----------------------------------------------
    miss_rank = np.cumsum(fast.miss[1])
    m7 = {
        # From the origin: how long a traversal start is separated from its
        # target. From the nearest producer: how long the value sits in the
        # core before the consumer uses it, which sizes a table of recent
        # producer loads.
        "loads_between_producer_and_consumer": pct(idx - prod),
        "l1d_misses_between": pct(miss_rank[idx] - miss_rank[prod]),
        "loads_between_nearest_producer_and_consumer": pct(idx - nearest),
        "l1d_misses_between_nearest": pct(miss_rank[idx] - miss_rank[nearest]),
        "cycles_between_nearest_data_and_issue": pct(slack_nearest),
        "cycles_between_data_and_issue": pct(slack),
        "micro_ops_on_path": pct(relations["depth"][idx]),
        "through_memory_share": share(
            w[relations["through_memory"][idx]].sum(), total_w
        ),
    }
    out["M7"] = m7

    # ---- M8: translation exposure (no TLB capture yet) ----------------
    target_page = load["vaddr"][idx] >> np.uint64(12)
    producer_page = load["vaddr"][prod] >> np.uint64(12)
    out["M8"] = {
        "target_on_other_page_than_producer": share(
            w[target_page != producer_page].sum(), total_w
        ),
        "target_page_among_last_64_miss_pages": share(
            w[classes["known_page"][idx]].sum(), total_w
        ),
        "note": "data-TLB misses are not captured yet",
    }
    return out


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    result = measure(Window(sys.argv[1], sys.argv[3]), sys.argv[2])
    Path(sys.argv[4]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[4]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["window"], indent=2))


if __name__ == "__main__":
    main()
