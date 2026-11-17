#!/usr/bin/env python3
"""Generate the numeric tables drawn in the Chinese report."""

from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'lstm' / 'v5_results'
REGIONS = (310, 330, 350, 370, 390)
POLICIES = dict(nopf='NoPF', next_line='Next-Line', stride='Stride', streamer='Streamer',
                ampm='AMPM', spp='SPP', event='Event-only', lstm='LSTM')
QUALITY = ('accuracy', 'timeliness', 'direct_coverage', 'timely_coverage', 'llc_hit_rate', 'llc_miss_rate')
FRESH_REGIONS = (410, 430, 450)


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def heat_cells(table, digits, direct_scale=False):
    """Return annotated cells with intensity scaled independently per column."""
    columns = list(zip(*table))
    cells = []
    for col, values in enumerate(columns):
        finite = [value for value in values if value is not None]
        bottom = 0 if direct_scale else min(finite)
        spread = (100 if direct_scale else max(finite) - bottom) or 1
        for row, value in enumerate(values):
            label = '—' if value is None else format(value, f'.{digits[col]}f')
            shade = 0 if value is None else round(6 + 36 * (value - bottom) / spread)
            cells.append(f'{col}/{row}/{shade}/{label}')
    return ','.join(cells)


def phase_cells(table, metric, factor, digits, regions=REGIONS):
    return heat_cells([[factor * float(table[policy, phase][metric]) for phase in regions]
                       for policy in POLICIES], (digits,) * len(regions), direct_scale=factor == 100)


def values(row, names, factor):
    return [factor * float(row[name]) if row[name] else None for name in names]


def clean_totals(replays):
    groups = [[row for row in replays if row['policy'] == policy] for policy in POLICIES]
    quality, cost = [], []
    for rows in groups:
        def total(field):
            return sum(int(row[field]) for row in rows)

        timely = total('prefetch_timely')
        issued = total('prefetch_issued')
        used = timely + total('prefetch_late')
        retired = total('retired')
        hits, misses = total('llc_hits'), total('llc_misses')
        cycles = total('roi_cycles')
        quality.append([100 * used / issued if issued else None, 100 * timely / used if used else None,
                        100 * total('matched_served') / total('nopf_misses'),
                        100 * total('matched_timely') / total('nopf_misses'),
                        100 * hits / (hits + misses), 100 * misses / (hits + misses)])
        cost.append([retired / cycles, cycles / 4e6, 1000 * misses / retired,
                     issued / 1000, total('dram_total_bytes') / 1e6])
    return quality, cost


def main():
    raw = read(RESULTS / 'per_phase.csv')
    table = {(row['policy'], int(row['region'])): row for row in raw}
    totals = {row['policy']: row for row in read(RESULTS / 'policy_totals.csv')}
    macros = {
        'ReportPolicies': ','.join(f'{i}/{label}' for i, label in enumerate(POLICIES.values())),
        'ReportRegions': ','.join(f'{i}/{region}M' for i, region in enumerate(REGIONS)),
        'PhaseIpcCells': phase_cells(table, 'ipc', 1, 4),
        'PhaseCoverageCells': phase_cells(table, 'direct_coverage', 100, 2),
        'QualityCells': heat_cells([values(totals[key], QUALITY, 100) for key in POLICIES],
                                   (2,) * len(QUALITY), direct_scale=True),
        'CostCells': heat_cells([[float(totals[key]['ipc']), float(totals[key]['roi_seconds']) * 1000,
                                 float(totals[key]['llc_mpki']), int(totals[key]['issued']) / 1000,
                                 int(totals[key]['dram_bytes']) / 1e6] for key in POLICIES],
                                (5, 3, 3, 1, 2)),
    }
    fresh = [row for row in read(RESULTS / 'clean32' / 'per_phase.csv')
             if row['label_latency'] == '32' and row['group'] == 'fresh']
    fresh_table = {(row['policy'], int(row['region'])): row for row in fresh}
    quality, cost = clean_totals(fresh)
    macros['FreshRegions'] = ','.join(f'{i}/{region}M' for i, region in enumerate(FRESH_REGIONS))
    macros['FreshIpcCells'] = phase_cells(fresh_table, 'aggregate_ipc', 1, 4, FRESH_REGIONS)
    macros['FreshQualityCells'] = heat_cells(quality, (2,) * len(QUALITY), direct_scale=True)
    macros['FreshCostCells'] = heat_cells(cost, (5, 3, 3, 1, 2))
    target = ROOT / 'report' / 'figures.tex'
    target.write_text('% Generated from v5_results CSVs by make_figures.py.\n'
                      + ''.join(f'\\newcommand{{\\{name}}}{{{value}}}\n' for name, value in macros.items()))


if __name__ == '__main__':
    main()
