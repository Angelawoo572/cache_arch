#!/usr/bin/env python3
"""Plot complete live H8/H16 action distributions and saved H16 progress.

Read-only inputs: original formal CSVs, retained width CSVs, completion CSVs.
Writes matching SVG/PNG and exact plot-source CSVs under the current results.
No simulation, fitting, interpolation, or first-crossing history estimate.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'formal_study/results/602_width_templates'
REFERENCE = ROOT / 'formal_NN_training/experiments/602_four_prefetcher_formal/results/formal_20261017'
for key, name in (('TMPDIR', 'tmp'), ('TMP', 'tmp'), ('TEMP', 'tmp'),
                  ('MPLCONFIGDIR', 'mpl'), ('XDG_CACHE_HOME', 'xdg')):
    folder = ROOT / '.runtime' / name
    folder.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(folder)
import tempfile
tempfile.tempdir = str(ROOT / '.runtime/tmp')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np

POLICIES = ('stride', 'streamer', 'ampm', 'spp')
NAMES = dict(stride='Stride', streamer='Streamer', ampm='AMPM', spp='SPP')
COLORS = dict(stride='#0072B2', streamer='#009E73', ampm='#D55E00', spp='#CC79A7')
WIDTH_COLORS = {8: '#0072B2', 16: '#D55E00'}
plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def dump(name, rows):
    assert rows, name
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return str(path.relative_to(ROOT))


def save(fig, name, figures):
    for ext in ('svg', 'png'):
        destination = OUT / (name + '.' + ext)
        fig.savefig(destination, dpi=160, bbox_inches='tight')
        if ext == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
    plt.close(fig)
    figures.append(name)


def budget(cap):
    return 'uncapped' if cap == 0 else 'ceiling 16,144 / M'


def bundle(row):
    if row['hidden_size'] == 8:
        return REFERENCE, ''
    return OUT, '_h16_completion' if row['policy'] in ('stride', 'streamer') else ''


def load():
    audit = json.loads((OUT / 'audit_h16_completion.json').read_text())
    assert audit['status'] == 'PASS' and audit['complete_count'] == 30
    rows = []
    for name in ('per_run.csv', 'per_run_h16_completion.csv'):
        for s in read(OUT / name):
            if s['stage'] != 'test' or s['mode'] != 'full' or s['method'] not in ('student', 'stride_student'):
                continue
            r = dict(s)
            for key in ('roi', 'budget', 'instructions', 'cycles', 'roi_demand_callbacks', 'roi_fill_callbacks', 'roi_completed_updates'):
                r[key] = int(s[key])
            r['hidden_size'] = int(s.get('hidden_size') or s.get('raw__h') or 0)
            r['metrics_source_csv'] = relative(OUT / name)
            assert r['hidden_size'] in (8, 16) and r['status'] == 'complete'
            assert 25_000_000 <= r['instructions'] <= 25_000_004
            assert math.isclose(float(r['ipc']), r['instructions'] / r['cycles'], rel_tol=1e-12)
            assert r['roi_completed_updates'] == r['roi_demand_callbacks'] + r['roi_fill_callbacks']
            assert r['policy'] == 'spp' or r['roi_fill_callbacks'] == 0
            rows.append(r)
    assert len(rows) == 32 and len({r['result_id'] for r in rows}) == 32
    lookup = {(r['policy'], r['hidden_size'], r['roi'], r['budget']): r for r in rows}
    assert len(lookup) == 32
    return rows, lookup


def exact_sources(rows):
    cache = {}
    full, phase, snapshots, progress = {}, {}, {}, {}
    for r in rows:
        directory, suffix = bundle(r)
        key = (directory, suffix)
        if key not in cache:
            cache[key] = {kind: read(directory / (kind + suffix + '.csv')) for kind in
                          ('action_histograms_full_roi', 'action_histograms', 'progress_snapshots', 'phase_progress')}
        source = cache[key]
        rid = r['run_id']
        full[rid] = defaultdict(Counter)
        phase[rid] = defaultdict(lambda: defaultdict(Counter))
        for s in source['action_histograms_full_roi']:
            if s['run_id'] == rid and s['stream'] == 'raw':
                full[rid][s['statistic']][int(s['value'])] += int(s['count'])
        for s in source['action_histograms']:
            if s['run_id'] == rid and s['stream'] == 'raw':
                phase[rid][int(s['phase'])][s['statistic']][int(s['value'])] += int(s['count'])
        assert full[rid] and phase[rid], rid
        for stat, support in full[rid].items():
            total = Counter()
            for hists in phase[rid].values():
                total.update(hists[stat])
            assert total == support, (rid, stat, 'full equals every recorded phase')
        kh, dh, fh, eh = (full[rid][stat] for stat in ('K', 'delta', 'fill', 'emit'))
        assert sum(kh.values()) == sum(eh.values()) == r['roi_demand_callbacks'], rid
        actions = sum(k * count for k, count in kh.items())
        assert actions == sum(dh.values()) == sum(fh.values()), rid
        assert eh[1] == sum(count for k, count in kh.items() if k > 0), rid
        assert set(fh) <= {2, 4}, rid
        for stat in ('delta', 'fill'):
            merged = Counter()
            for name, hist in full[rid].items():
                if name.startswith(stat + '_slot_'):
                    merged.update(hist)
            if merged:
                assert merged == full[rid][stat], (rid, stat, 'slot conservation')
            else:
                assert r['policy'] == 'stride' and stat == 'fill' and set(full[rid]['fill']) == {2}, (rid, stat)
        snapshots[rid] = sorted((s for s in source['progress_snapshots'] if s['run_id'] == rid), key=lambda s: int(s['instructions']))
        progress[rid] = sorted((s for s in source['phase_progress'] if s['run_id'] == rid), key=lambda s: int(s['phase']))
        last = snapshots[rid][-1]
        assert int(last['instructions']) == r['instructions'] and int(last['cycles']) == r['cycles']
        assert int(last['roi_demand_callbacks']) == r['roi_demand_callbacks']
        assert int(last['roi_fill_callbacks']) == r['roi_fill_callbacks']
        assert int(last['roi_completed_updates']) == r['roi_completed_updates']
        assert 0 <= r['instructions'] - sum(int(s['interval_instructions']) for s in progress[rid]) <= 4
        assert sum(int(s['interval_cycles']) for s in progress[rid]) == r['cycles']
        r['histogram_source_csv'] = relative(directory / ('action_histograms_full_roi' + suffix + '.csv'))
        r['phase_histogram_source_csv'] = relative(directory / ('action_histograms' + suffix + '.csv'))
        r['snapshot_source_csv'] = relative(directory / ('progress_snapshots' + suffix + '.csv'))
        r['phase_progress_source_csv'] = relative(directory / ('phase_progress' + suffix + '.csv'))
    return full, phase, snapshots, progress


def identity(r):
    return {k: r[k] for k in ('result_id', 'run_id', 'protocol', 'evidence_origin', 'policy', 'hidden_size', 'roi', 'budget')}


def source_tables(rows, full, phase, snapshots, progress):
    distribution, slots, phase_rows, startup = [], [], [], []
    for r in rows:
        rid = r['run_id']
        demand = r['roi_demand_callbacks']
        for stat, support in sorted(full[rid].items()):
            total = sum(support.values())
            for value, count in sorted(support.items()):
                row = identity(r)
                row.update(statistic=stat, value=value, count=count, histogram_denominator=total,
                           histogram_fraction=count / total, demand_decisions=demand,
                           occurrences_per_demand_decision=count / demand,
                           source_csv=r['histogram_source_csv'], support_basis='recorded complete ROI histogram', scope='exact all recorded phases of complete live ROI; raw pre-filter outputs')
                (slots if '_slot_' in stat else distribution).append(row)
        if r['policy'] == 'stride':
            # The retained Stride adapter records delta/slot and one-valued L2
            # fill, rather than joint slot/fill. This exact deduction is labeled.
            for stat, support in sorted(full[rid].items()):
                if not stat.startswith('delta_slot_'):
                    continue
                slot = int(stat.rsplit('_', 1)[1])
                for delta, count in sorted(support.items()):
                    row = identity(r)
                    row.update(statistic='delta_fill_2_slot_' + str(slot), value=delta, count=count,
                               histogram_denominator=sum(support.values()), histogram_fraction=count / sum(support.values()),
                               demand_decisions=demand, occurrences_per_demand_decision=count / demand,
                               source_csv=r['histogram_source_csv'],
                               support_basis='derived exact joint: recorded delta_slot plus single-valued complete ROI L2 fill',
                               scope='entire live ROI; raw pre-filter outputs; joint slot/fill not separately recorded')
                    slots.append(row)
        bounds = {int(s['phase']): s for s in progress[rid]}
        for p, hist in sorted(phase[rid].items()):
            kh, eh, fh = (hist[stat] for stat in ('K', 'emit', 'fill'))
            decisions = sum(kh.values())
            actions = sum(k * n for k, n in kh.items())
            assert decisions > 0 and actions == sum(hist['delta'].values()) == sum(fh.values())
            end = min((p + 1) * 5_000_000, r['instructions'])
            bound = bounds.get(p)
            v = identity(r)
            v.update(phase=p, profile_end_instructions=end, demand_decisions=decisions, decoded_slots=actions,
                     emit_count=eh[1], emit_fraction=eh[1] / decisions, mean_K=actions / decisions,
                     L2_slots=fh[2], LLC_slots=fh[4], L2_slot_fraction=fh[2] / actions if actions else '',
                     LLC_slot_fraction=fh[4] / actions if actions else '',
                     interval_ipc=bound['interval_ipc'] if bound else '',
                     actual_end_instructions=bound['actual_end_instructions'] if bound else '',
                     interval_demand_callbacks=bound['interval_demand_callbacks'] if bound else '',
                     interval_fill_callbacks=bound['interval_fill_callbacks'] if bound else '',
                     interval_completed_updates=bound['interval_completed_updates'] if bound else '',
                     histogram_source_csv=r['phase_histogram_source_csv'], progress_source_csv=r['phase_progress_source_csv'],
                     scope='exact callback-assigned 5M instruction profile; saved interval uses actual snapshot boundaries')
            phase_rows.append(v)
        for s in snapshots[rid]:
            if int(s['instructions']) == 0:
                continue
            v = identity(r)
            v.update({k: s[k] for k in ('instructions', 'cycles', 'cumulative_ipc', 'roi_demand_callbacks', 'roi_fill_callbacks',
                                       'roi_completed_updates', 'warmup_completed_updates', 'l2_load_misses', 'dram_total_bus_bytes')})
            v.update(source_csv=r['snapshot_source_csv'], scope='all saved actual progress coordinates; ROI-only x axes; cache warmup retained')
            startup.append(v)
    dump('action_plot_data_h16_completion.csv', distribution)
    dump('slot_support_h16_completion.csv', slots)
    dump('phase_plot_data_h16_completion.csv', phase_rows)
    dump('startup_plot_data_h16_completion.csv', startup)
    return distribution, slots, phase_rows, startup


def bars(ax, runs, full, stat):
    supports = [full[r['run_id']][stat] for r in runs]
    values = sorted(set().union(*(set(hist) for hist in supports)))
    assert len(values) <= 16, ('explicit display limit; retain Other if extending support', stat, values)
    if stat == 'fill':
        values = [2, 4]
    for i, (r, support) in enumerate(zip(runs, supports)):
        total = sum(support.values())
        ys = [100 * support[v] / total for v in values]
        xs = np.arange(len(values)) + (i - .5) * .36
        ax.bar(xs, ys, .34, color=WIDTH_COLORS[r['hidden_size']], label='H' + str(r['hidden_size']), zorder=3)
        for x, value, y in zip(xs, values, ys):
            n = support[value]
            if n == 0:
                continue
            if y < .05:
                label = 'n=' + str(n)
                height = 1.1
            else:
                label = f'{y:.1f}' if y < 99.95 else '100'
                height = y
            ax.text(x, height, label, ha='center', va='bottom', fontsize=6.5, rotation=90 if len(values) > 5 else 0)
    labels = [str(v) for v in values]
    if stat == 'fill':
        labels = ['L2', 'LLC']
    elif stat == 'delta':
        labels = [f'{v:+d}' if v else '0\ncurrent' for v in values]
    ax.set_xticks(range(len(values)), labels, fontsize=8)
    ax.set_ylim(0, 116)
    ax.grid(axis='y', alpha=.18, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel('% demand decisions' if stat == 'K' else '% decoded slots')


def distribution_figures(lookup, full, figures):
    for roi in (0, 1):
        for cap in (0, 16144):
            fig, axes = plt.subplots(4, 3, figsize=(13.5, 11.6))
            for i, policy in enumerate(POLICIES):
                runs = [lookup[(policy, h, roi, cap)] for h in (8, 16)]
                for j, stat in enumerate(('K', 'delta', 'fill')):
                    ax = axes[i, j]
                    bars(ax, runs, full, stat)
                    ax.set_title(NAMES[policy] + ' · ' + {'K': 'decoded K', 'delta': 'delta (cache lines)', 'fill': 'decoded fill intent'}[stat])
                axes[i, 0].set_title(NAMES[policy] + ' · decoded K\n' + f"D8={runs[0]['roi_demand_callbacks']:,}; D16={runs[1]['roi_demand_callbacks']:,}", fontsize=9)
            fig.legend(handles=[Line2D([0], [0], lw=8, color=WIDTH_COLORS[h], label='H' + str(h) + ' NN') for h in (8, 16)],
                       loc='upper center', bbox_to_anchor=(.5, .956), ncol=2, frameon=False)
            fig.suptitle(f'Whole-ROI actions · ROI{roi} · {budget(cap)} · seed7 · MAC0', fontsize=14)
            fig.text(.04, .035, 'All support values shown. Labels below 0.05% give exact n; source CSV retains every count and slot.', fontsize=9)
            fig.text(.04, .018, 'Raw outputs precede current-line/page/credit filters. K includes silent demand decisions; delta/fill weight every decoded slot.', fontsize=9)
            fig.text(.04, .001, 'SPP fill callbacks update state but do not enter the demand-decision denominator. Decoded fill intent is separate from tracked cache cohorts.', fontsize=9)
            fig.tight_layout(rect=(0, .055, 1, .932))
            save(fig, f'actions_all_widths_roi{roi}_cap{cap}', figures)


def slot_figures(lookup, full, figures):
    # Per-slot frequencies use every eligible demand as denominator. An absent
    # cell is zero, not missing; count labels preserve tiny nonzero supports.
    for roi in (0, 1):
        for cap in (0, 16144):
            fig, axes = plt.subplots(4, 2, figsize=(13, 10.5))
            for i, policy in enumerate(POLICIES):
                runs = [lookup[(policy, h, roi, cap)] for h in (8, 16)]
                supports = []
                slots = set()
                categories = set()
                for r in runs:
                    joint = Counter()
                    for name, hist in full[r['run_id']].items():
                        if not name.startswith('delta_fill_'):
                            continue
                        parts = name.split('_')
                        fill, slot = int(parts[2]), int(parts[4])
                        for delta, n in hist.items():
                            joint[(slot, delta, fill)] += n
                            slots.add(slot)
                            categories.add((delta, fill))
                    if not joint:
                        assert r['policy'] == 'stride' and set(full[r['run_id']]['fill']) == {2}
                        for name, hist in full[r['run_id']].items():
                            if name.startswith('delta_slot_'):
                                slot = int(name.rsplit('_', 1)[1])
                                for delta, n in hist.items():
                                    joint[(slot, delta, 2)] += n
                                    slots.add(slot)
                                    categories.add((delta, 2))
                    assert sum(joint.values()) == sum(full[r['run_id']]['delta'].values())
                    supports.append(joint)
                slots, categories = sorted(slots), sorted(categories)
                for j, (r, support) in enumerate(zip(runs, supports)):
                    ax = axes[i, j]
                    matrix = np.array([[support[(slot, delta, fill)] / r['roi_demand_callbacks'] for delta, fill in categories] for slot in slots])
                    assert matrix.shape[0] > 0 and matrix.shape[1] > 0
                    ax.imshow(np.ma.masked_equal(matrix, 0), cmap='Blues' if r['hidden_size'] == 8 else 'Oranges',
                              norm=LogNorm(vmin=1 / r['roi_demand_callbacks'], vmax=1), aspect='auto')
                    for a, slot in enumerate(slots):
                        for b, (delta, fill) in enumerate(categories):
                            n = support[(slot, delta, fill)]
                            if n:
                                fraction = n / r['roi_demand_callbacks']
                                label = f'{fraction * 100:.1f}%' if fraction >= .0005 else 'n=' + str(n)
                                ax.text(b, a, label, ha='center', va='center', fontsize=7, color='white' if fraction > .15 else '#222222')
                    ax.set_xticks(range(len(categories)), [f'{delta:+d}\n' + ('L2' if fill == 2 else 'LLC') for delta, fill in categories], fontsize=7)
                    ax.set_yticks(range(len(slots)), ['slot ' + str(slot + 1) for slot in slots], fontsize=8)
                    ax.set_title(NAMES[policy] + ' · H' + str(r['hidden_size']))
                    ax.tick_params(length=0)
            fig.suptitle(f'Exact slot support · ROI{roi} · {budget(cap)} · seed7 · MAC0', fontsize=14)
            fig.text(.04, .028, 'Cells = occurrences / all demand decisions. Blank cells = zero; small cells retain exact n. Each slot can be absent when K is shorter.', fontsize=9)
            fig.text(.04, .01, 'Log color; labels rounded to 0.1%. Stride joint fill is deduced from recorded slot delta + constant L2 intent; CSV labels this deduction.', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .955))
            save(fig, f'slots_all_widths_roi{roi}_cap{cap}', figures)


def phase_figures(lookup, phase_rows, figures):
    for roi in (0, 1):
        for cap in (0, 16144):
            fig, axes = plt.subplots(4, 3, figsize=(13, 10.5))
            for i, policy in enumerate(POLICIES):
                for h in (8, 16):
                    r = lookup[(policy, h, roi, cap)]
                    stages = [s for s in phase_rows if s['result_id'] == r['result_id']]
                    for j, (field, label, scale) in enumerate((('emit_fraction', 'Emit / demand (%)', 100),
                            ('mean_K', 'Mean raw decoded K', 1), ('L2_slot_fraction', 'L2 intent / decoded slots (%)', 100))):
                        series = [s for s in stages if s[field] != '']
                        axes[i, j].plot([int(s['profile_end_instructions']) / 1e6 for s in series],
                                        [float(s[field]) * scale for s in series], color=WIDTH_COLORS[h],
                                        marker='o', ms=3, lw=1.5, label='H' + str(h))
                        axes[i, j].set_title(NAMES[policy] + ' · ' + label)
                        axes[i, j].set_xlabel('Profile phase end: ROI instructions (M)')
                        axes[i, j].grid(alpha=.18)
                        if field != 'mean_K':
                            axes[i, j].set_ylim(-3, 105)
                        else:
                            axes[i, j].set_ylim(0, max(1, max(float(s[field]) for s in stages if s[field] != '')) * 1.12)
            fig.legend(handles=[Line2D([0], [0], color=WIDTH_COLORS[h], lw=2, label='H' + str(h) + ' NN') for h in (8, 16)],
                       loc='upper center', bbox_to_anchor=(.5, .954), ncol=2, frameon=False)
            fig.suptitle(f'Complete ROI phase actions · ROI{roi} · {budget(cap)} · seed7 · MAC0', fontsize=14)
            fig.text(.04, .027, 'Every actual 5M instruction profile; no prefix extrapolation. Emit and K use demand decisions; L2 intent uses decoded slots before filters.', fontsize=9)
            fig.text(.04, .009, 'SPP fill callbacks update state separately and do not enter action denominators. Curves show execution progress, not training volume or required history.', fontsize=9)
            fig.tight_layout(rect=(0, .05, 1, .93))
            save(fig, f'phase_actions_all_widths_roi{roi}_cap{cap}', figures)


def history_figures(lookup, phase_rows, startup, figures):
    for roi in (0, 1):
        for cap in (0, 16144):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for policy in POLICIES:
                r = lookup[(policy, 16, roi, cap)]
                series = [s for s in startup if s['result_id'] == r['result_id']]
                stages = [s for s in phase_rows if s['result_id'] == r['result_id']]
                for j, (field, label, scale) in enumerate((('instructions', 'Retired ROI instructions (M)', 1e6),
                        ('roi_demand_callbacks', 'ROI demand callbacks (K)', 1e3),
                        ('roi_completed_updates', 'Completed ROI inferences (K)', 1e3))):
                    axes[0, j].plot([int(s[field]) / scale for s in series], [float(s['cumulative_ipc']) for s in series],
                                    color=COLORS[policy], label=NAMES[policy], lw=1.5)
                    axes[0, j].set_xlabel(label)
                    axes[0, j].set_ylabel('Cumulative measured IPC')
                for j, (field, label, scale) in enumerate((('interval_ipc', 'Measured phase IPC', 1),
                        ('mean_K', 'Mean decoded K / demand', 1), ('LLC_slot_fraction', 'LLC intent / decoded slots (%)', 100))):
                    eligible = [s for s in stages if s[field] != '']
                    xfield = 'actual_end_instructions' if field == 'interval_ipc' else 'profile_end_instructions'
                    axes[1, j].plot([int(s[xfield]) / 1e6 for s in eligible], [float(s[field]) * scale for s in eligible],
                                    color=COLORS[policy], marker='o', label=NAMES[policy], lw=1.5, ms=3)
                    axes[1, j].set_xlabel('Phase end: ROI instructions (M)')
                    axes[1, j].set_ylabel(label)
            for ax in axes.flat:
                ax.grid(alpha=.18)
            axes[1, 2].set_ylim(-3, 105)
            fig.legend(handles=[Line2D([0], [0], color=COLORS[p], label=NAMES[p], lw=2) for p in POLICIES],
                       loc='upper center', bbox_to_anchor=(.5, .94), ncol=4, frameon=False)
            fig.suptitle(f'H16 saved progress · ROI{roi} · {budget(cap)} · seed7 · MAC0', fontsize=14)
            fig.text(.04, .037, 'All actual saved snapshots; full 25M ROI and 5M common cache warmup. X axes exclude warmup; initial model state follows the matched recipe.', fontsize=9)
            fig.text(.04, .019, 'SPP completed inferences include demand + real fill callbacks. Other routes update only on demand. Curves describe execution progress, not training volume.', fontsize=9)
            fig.text(.04, .001, 'Phase IPC uses saved boundaries (terminal +2/+3 retired instructions share the final cycle). No first-crossing history estimate.', fontsize=9)
            fig.tight_layout(rect=(0, .065, 1, .91))
            save(fig, f'history_all_h16_roi{roi}_cap{cap}', figures)


def main():
    rows, lookup = load()
    full, phase, snapshots, progress = exact_sources(rows)
    distribution, slots, phase_rows, startup = source_tables(rows, full, phase, snapshots, progress)
    figures = []
    distribution_figures(lookup, full, figures)
    slot_figures(lookup, full, figures)
    phase_figures(lookup, phase_rows, figures)
    history_figures(lookup, phase_rows, startup, figures)
    manifest = dict(status='PASS', full_NN_test_rows=len(rows), hidden_widths=[8, 16], rois=[0, 1], budgets=[0, 16144],
                    source_histogram_rows=len(distribution), exact_slot_rows=len(slots), phase_rows=len(phase_rows), saved_snapshots=len(startup),
                    figures=figures, figure_formats=['svg', 'png'], no_simulation=True, source_files_modified=False,
                    full_ROI_not_prefix=True, exact_slot_conservation=True,
                    sources=sorted({r[k] for r in rows for k in ('metrics_source_csv', 'histogram_source_csv', 'phase_histogram_source_csv', 'snapshot_source_csv', 'phase_progress_source_csv')}),
                    plot_data=['action_plot_data_h16_completion.csv', 'slot_support_h16_completion.csv', 'phase_plot_data_h16_completion.csv', 'startup_plot_data_h16_completion.csv'],
                    limits=['Seed7 only; MAC0 synchronous inference, not a finite-cost IPC forecast.',
                            'K uses demand decisions; SPP fill callbacks advance recurrent state separately.',
                            'Raw decoded slots precede current-line, page and credit filtering; decoded intent is not tracked cohort fate.',
                            'Execution progress is separate from training data and required history; no first-crossing or extrapolation claim.',
                            'Saved phase IPC ends at 25M; terminal +2/+3 retired instructions share the final cycle. Cumulative IPC uses exact final instructions.'])
    (OUT / 'plot_manifest_h16_completion.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('status', 'full_NN_test_rows', 'source_histogram_rows', 'exact_slot_rows', 'phase_rows', 'saved_snapshots')}))
    print('Matching SVG/PNG pairs:', len(figures))


if __name__ == '__main__':
    main()
