#!/usr/bin/env python3
"""Measure the value of older page-source PCs in a fixed NoPF candidate pool."""

import argparse
from collections import Counter, defaultdict, deque
import csv
from functools import lru_cache
import json
from pathlib import Path
from random import Random
from typing import NamedTuple


LINES_PER_PAGE = 64
PAGE_HISTORY = 32
WARMUP_EVENTS = 128
VISIBLE_HISTORY = (0, 8, 16, 32)
OFFSETS = (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6)
SHUFFLE_SEEDS = (10, 20, 30)


class Demand(NamedTuple):
    event: int
    cycle: int
    line: int
    pc: int
    miss: bool


class Candidate(NamedTuple):
    key: tuple
    line: int
    target: int
    lag: int


def read_demands(directory):
    with (directory / "summary.json").open() as source:
        result = next(row for row in json.load(source) if row["policy"] == "nopf")
    if result["prefetch_issued"] != 0:
        raise ValueError("the NoPF run issued prefetches")
    path = (directory / result["log"]).with_name("demand.csv")
    per_core = defaultdict(list)
    with path.open() as source:
        for row in csv.DictReader(source):
            per_core[int(row["cpu"])].append(Demand(
                int(row["event_id"]), int(row["ingress_cycle"]),
                int(row["physical_line"]), int(row["pc"]), row["hit"] == "0",
            ))
    if sum(len(rows) for rows in per_core.values()) != result["llc_demand_accesses"]:
        raise ValueError("{} does not match the LLC lookup count".format(path))
    # Match the simulator's ordering of initial events across cores.
    for _, rows in sorted(per_core.items(), key=lambda item: min(row.event for row in item[1])):
        yield sorted(rows, key=lambda row: row.event)


def candidate_sets(directory, lead, stride):
    """Keep all events for state and sample only the anchors used for scoring."""
    minimum, maximum = lead
    for demands in read_demands(directory):
        first_access = {}
        for row in demands:
            first_access.setdefault(row.line, row)

        prior_pages = {}
        seen_lines = set()
        history = deque()
        last_cycle = demands[-1].cycle
        for index, row in enumerate(demands):
            page, offset = divmod(row.line, LINES_PER_PAGE)
            if index >= WARMUP_EVENTS and index % stride == 0 and row.cycle + maximum <= last_cycle:
                candidates = []
                for prior_page, (prior_index, prior) in prior_pages.items():
                    lag = index - prior_index
                    if prior_page == page or lag > PAGE_HISTORY:
                        continue
                    for delta in OFFSETS:
                        line = prior.line + delta
                        if line // LINES_PER_PAGE != prior_page or line in seen_lines:
                            continue
                        future = first_access.get(line)
                        timely = (future is not None and future.miss
                                  and minimum <= future.cycle - row.cycle <= maximum)
                        key = (delta, (lag - 1).bit_length(), prior.pc, row.pc,
                               offset // 16, (prior.line % LINES_PER_PAGE) // 16)
                        candidates.append(Candidate(key, line, future.event if timely else 0, lag))
                if candidates:
                    yield candidates

            seen_lines.add(row.line)
            prior_pages[page] = (index, row)
            history.append((index, page))
            while history[0][0] < index - PAGE_HISTORY:
                previous_index, previous_page = history.popleft()
                if prior_pages[previous_page][0] == previous_index:
                    del prior_pages[previous_page]


def feature_keys(candidate_key):
    delta, age, prior_pc, current_pc, current_offset, prior_offset = candidate_key
    partial = (delta, age, current_pc, current_offset, prior_offset)
    return (delta, age), partial, partial + (prior_pc,)


def frozen_rates(directory, lead, stride):
    total = [Counter(), Counter(), Counter()]
    positive = [Counter(), Counter(), Counter()]
    for candidates in candidate_sets(directory, lead, stride):
        for candidate in candidates:
            for level, key in enumerate(feature_keys(candidate.key)):
                total[level][key] += 1
                positive[level][key] += candidate.target != 0

    @lru_cache(maxsize=None)
    def lookup(candidate_key):
        rate = 0.005
        rates = []
        keys = feature_keys(candidate_key)
        for level, key in enumerate(keys):
            rate = (positive[level][key] + 20 * rate) / (total[level][key] + 20)
            rates.append(rate)
        return rates[1], rates[2], total[2][keys[2]]

    return lookup


def rank_decisions(directory, lead, stride, lookup):
    tie_break = Random(37)
    decisions = {window: [] for window in VISIBLE_HISTORY}
    shuffled_modes = {"shuffle_{}".format(seed): Random(seed) for seed in SHUFFLE_SEEDS}
    decisions.update({mode: [] for mode in shuffled_modes})
    for candidates in candidate_sets(directory, lead, stride):
        tie = tie_break.random()
        winners = {}
        page_pc = {(candidate.line - candidate.key[0], candidate.lag): candidate.key[2]
                   for candidate in candidates if candidate.lag > 8}
        shuffled = {}
        for mode, rng in shuffled_modes.items():
            pcs = list(page_pc.values())
            rng.shuffle(pcs)
            shuffled[mode] = dict(zip(page_pc, pcs))

        for candidate in candidates:
            partial, full, support = lookup(candidate.key)
            scores = {window: (full, support) if candidate.lag <= window else (partial, -1)
                      for window in VISIBLE_HISTORY}
            for mode, pcs in shuffled.items():
                if candidate.lag <= 8:
                    scores[mode] = full, support
                else:
                    source = candidate.line - candidate.key[0], candidate.lag
                    key = candidate.key[:2] + (pcs[source],) + candidate.key[3:]
                    _, altered, altered_support = lookup(key)
                    scores[mode] = altered, altered_support

            for mode, (score, support) in scores.items():
                order = (score, -candidate.lag, -abs(candidate.key[0]), -candidate.line)
                if mode not in winners or order > winners[mode][0]:
                    winners[mode] = order, candidate, support
        for mode, (order, candidate, support) in winners.items():
            decisions[mode].append((order[0], tie, candidate, support))
    return decisions


def summarize(decisions, budgets):
    for mode, rows in decisions.items():
        rows.sort(reverse=True)
        distinct_requests = []
        seen = set()
        for row in rows:
            line = row[2].line
            if line not in seen:
                seen.add(line)
                distinct_requests.append(row)
        for budget in budgets:
            if budget > len(rows):
                raise ValueError("not enough eligible anchors for budget {}".format(budget))
            selected = rows[:budget]
            deduped = distinct_requests[:budget]
            yield {
                "visible_source_pc_events": mode,
                "eligible_anchors": len(rows),
                "budget": budget,
                "unique_targets": len({candidate.target for _, _, candidate, _ in selected
                                       if candidate.target}),
                "older_than_8_decisions": sum(candidate.lag > 8 for _, _, candidate, _ in selected),
                "full_key_unseen": sum(support == 0 for _, _, _, support in selected),
                "full_key_used": sum(support >= 0 for _, _, _, support in selected),
                "deduped_decisions": len(deduped),
                "deduped_hits": sum(candidate.target != 0 for _, _, candidate, _ in deduped),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path, help="NoPF development run directory")
    parser.add_argument("evaluation", type=Path, help="NoPF evaluation run directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lead", nargs=2, type=int, metavar=("MIN", "MAX"), default=(300, 2000))
    parser.add_argument("--budgets", nargs="+", type=int, default=(1000, 2000))
    parser.add_argument("--dev-stride", type=int, default=4)
    parser.add_argument("--eval-stride", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.lead[0] <= args.lead[1]:
        parser.error("lead bounds must be positive and increasing")
    if min(args.dev_stride, args.eval_stride, min(args.budgets)) < 1:
        parser.error("strides and budgets must be positive")

    lookup = frozen_rates(args.development, args.lead, args.dev_stride)
    decisions = rank_decisions(args.evaluation, args.lead, args.eval_stride, lookup)
    rows = list(summarize(decisions, args.budgets))
    with args.output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
