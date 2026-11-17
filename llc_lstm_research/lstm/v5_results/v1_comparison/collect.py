#!/usr/bin/env python3
"""Join matched replays to NoPF and publish English per-region and pooled data."""

import csv
import json
from pathlib import Path
import sys

from replay import CONDITIONS, FRESH_REGIONS, NEWEST, OLD_REGIONS, REGIONS, ROOT, V1, anchor, target

sys.path.insert(0, str(ROOT / 'lstm'))
from compare import rows

OUTPUT = Path(__file__).resolve().parent
ARMS = ('v1', 'newest')
WEIGHTS = {'v1': (V1, -1037), 'newest': (NEWEST, -950)}
COUNTS = ('instructions', 'roi_cycles', 'nopf_misses', 'matched_nopf_misses', 'direct_matches',
          'timely_matches', 'llc_demand_hits', 'llc_demand_misses', 'llc_demand_accesses',
          'prefetch_issued', 'prefetch_timely', 'prefetch_late', 'prefetch_fills', 'prefetch_unused',
          'prefetch_redundant', 'prefetch_unresolved', 'core0_issued', 'core1_issued',
          'dram_total_bytes', 'dram_demand_read_bytes', 'dram_prefetch_read_bytes', 'dram_write_bytes')


def json_at(run, name):
    return json.loads((run / name).read_text())


def write_rows(filename, records):
    with (OUTPUT / filename).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def summarize(region, condition, arm):
    reference, output = anchor(region), target(region, condition, arm)
    row = next(row for row in rows([reference, output], nopf_only_from_first=True)
               if row['policy'] == 'lstm')
    row.pop('policy')
    initial_config = json_at(reference, 'config.json')
    config = json_at(output, 'config.json')
    settings = config['learned']
    weights, threshold = WEIGHTS[arm]
    model = settings['models']['lstm']
    cap = 0 if condition == 'native' else int(condition[3:])
    assert config['binary'] == initial_config['binary']
    assert all(settings[key] == value for key, value in {
        'prediction_latency': 32, 'prediction_interval': 8, 'prediction_state_cycles': 8,
        'prediction_depth': 4, 'issue_limit': cap,
    }.items())
    assert model['threshold'] == threshold and Path(model['file']['path']).samefile(weights)
    assert settings.get('position_scorer', False) == (arm == 'newest')
    raw, = json_at(output, 'summary.json')
    row['direct_matches'] = round(row['coverage'] * row['nopf_misses'])
    row['timely_matches'] = round(row['timely_coverage'] * row['nopf_misses'])
    row['core0_issued'], row['core1_issued'] = (core['prefetch_issued'] for core in raw['cores'])
    row.update({key: raw[key] for key in COUNTS if key in raw})
    assert not cap or max(row['core0_issued'], row['core1_issued']) <= cap
    return dict(region_m=region, condition=condition, model=arm, issue_cap_per_core=cap,
                threshold_q10=threshold, cpu_mhz=raw['cpu_mhz'],
                reference_path=str(reference.relative_to(ROOT)), run_path=str(output.relative_to(ROOT)), **row)


def pool(records, condition, cohort, regions, arm):
    common = [row for row in records
              if row['condition'] == condition and row['model'] == arm and row['region_m'] in regions]
    assert len(common) == len(regions)
    totals = {key: sum(row[key] for row in common) for key in COUNTS}
    ratio = lambda numerator, denominator: totals[numerator] / totals[denominator]
    used = totals['prefetch_timely'] + totals['prefetch_late']
    return dict(condition=condition, cohort=cohort, regions_m=' '.join(map(str, regions)), model=arm,
                **totals, accuracy=used / totals['prefetch_issued'], timeliness=totals['prefetch_timely'] / used,
                coverage=ratio('direct_matches', 'nopf_misses'),
                timely_coverage=ratio('timely_matches', 'nopf_misses'),
                llc_hit_rate=ratio('llc_demand_hits', 'llc_demand_accesses'),
                llc_miss_rate=ratio('llc_demand_misses', 'llc_demand_accesses'),
                llc_mpki=1000 * ratio('llc_demand_misses', 'instructions'),
                aggregate_ipc=ratio('instructions', 'roi_cycles'),
                roi_seconds=sum(row['roi_seconds'] for row in common))


def main():
    records = [summarize(region, condition, arm) for region in REGIONS
               for condition in CONDITIONS for arm in ARMS]
    write_rows('per_region.csv', records)
    pooled = []
    for condition in CONDITIONS:
        for group, regions in (('v1_test_regions', OLD_REGIONS), ('newest_test_regions', FRESH_REGIONS),
                               ('all_regions', REGIONS)):
            pooled.extend(pool(records, condition, group, regions, arm) for arm in ARMS)
        if condition != 'native':
            cap = int(condition[3:])
            exact = tuple(region for region in REGIONS
                          if all(row['core0_issued'] == row['core1_issued'] == cap for row in records
                                 if row['condition'] == condition and row['region_m'] == region))
            if exact and exact != REGIONS:
                pooled.extend(pool(records, condition, 'exact_count_regions', exact, arm) for arm in ARMS)
    write_rows('pooled.csv', pooled)


if __name__ == '__main__':
    main()
