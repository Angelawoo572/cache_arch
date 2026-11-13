#!/usr/bin/env python3
"""Measure which NoPF misses are reachable under different label and candidate limits."""

import argparse
from bisect import bisect_right
from collections import defaultdict, deque
import csv
import json
from pathlib import Path

from compare import read_events
from dataset import FIRST_ANCHOR, LINES_PER_PAGE, PAGE_HISTORY, read_demands, recent_candidates


PROTOCOLS = ((16, 300, 2000), (16, 150, 2000), (16, 300, 4000), (64, 300, 2000))


def reach(run):
    by_core, summary = read_demands(run)
    targets = {protocol: set() for protocol in PROTOCOLS}
    cap = max(protocol[0] for protocol in PROTOCOLS)
    max_lead = max(protocol[2] for protocol in PROTOCOLS)
    for core, demands in by_core.items():
        by_line = defaultdict(list)
        for demand in demands:
            by_line[demand.line].append(demand)
        event_ids = {line: [demand.event for demand in visits] for line, visits in by_line.items()}
        history = deque(maxlen=PAGE_HISTORY)
        for index, demand in enumerate(demands):
            if index >= FIRST_ANCHOR and demand.cycle + max_lead <= demands[-1].cycle:
                candidates = recent_candidates(history, demand.line // LINES_PER_PAGE, limit=cap)
                for slot, (line, _, _, _) in enumerate(candidates):
                    ids = event_ids.get(line, ())
                    later = bisect_right(ids, demand.event)
                    if later == len(ids) or not by_line[line][later].miss:
                        continue
                    target = by_line[line][later]
                    lead = target.cycle - demand.cycle
                    for protocol in PROTOCOLS:
                        width, low, high = protocol
                        if slot < width and low <= lead <= high:
                            targets[protocol].add((core, target.instruction, target.operand))
            history.append(demand)
    return targets, summary['counters']['skip'], summary['llc_demand_misses']


def late_uses(run):
    summaries = json.loads((run / 'summary.json').read_text())
    by_policy = {row['policy']: row for row in summaries}

    def events(policy):
        result = by_policy[policy]
        return read_events((run / result['log']).with_name('demand.csv'), result)

    misses = {key for key, (_, hit, _) in events('nopf').items() if not hit}
    for policy in ('streamer', 'spp'):
        late = {key for key, (_, _, use) in events(policy).items() if use == 'late'}
        yield policy, misses & late


def write_rows(path, rows):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sensitivity = []
    conventional = []
    for run in args.runs:
        targets, skip, misses = reach(run)
        region = skip // 1000000
        for (cap, low, high), hits in targets.items():
            sensitivity.append(dict(region=region, candidate_cap=cap, lead_min=low, lead_max=high,
                                    reachable=len(hits), nopf_misses=misses, reach=len(hits) / misses))
        baseline = targets[PROTOCOLS[0]]
        conventional.append(dict(region=region, policy='nopf', noPF_misses_in_subset=misses,
                                 reachable=len(baseline), reach=len(baseline) / misses))
        for policy, subset in late_uses(run):
            count = len(subset & baseline)
            conventional.append(dict(region=region, policy=policy, noPF_misses_in_subset=len(subset),
                                     reachable=count, reach=count / len(subset) if subset else None))
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / 'reach_sensitivity.csv', sensitivity)
    write_rows(args.output / 'candidate_reach.csv', conventional)


if __name__ == '__main__':
    main()
