#!/usr/bin/env python3
"""Collect the frozen study data and generate the numbers used in the LaTeX figures."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lstm'))
from compare import rows

PHASES = (310, 330, 350, 370, 390)
NATIVE = ('next_line', 'stride', 'streamer', 'ampm', 'spp')
DISPLAY = {'next_line': 'Next-Line', 'stride': 'Stride', 'streamer': 'Streamer',
           'ampm': 'AMPM', 'spp': 'SPP', 'event': 'Event-only', 'lstm': 'LSTM'}
RUNS = ROOT / 'conventional' / 'runs' / 'v5_comparison'
RESULTS = ROOT / 'lstm' / 'v5_results'
EXPERIMENTS = ROOT / 'lstm' / 'runs' / 'v5' / 'experiments'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def native_best(records, ipc_key):
    return max((row for row in records if row['policy'] in NATIVE), key=lambda row: float(row[ipc_key]))


def sum_key(records, key):
    return sum(int(row[key]) for row in records)


def speed(records, reference):
    return 1 - sum_key(records, 'roi_cycles') / sum_key(reference, 'roi_cycles')


def aggregate(records):
    total = lambda key: sum_key(records, key)
    return (100 * total('used') / total('issued'),
            100 * total('timely') / total('used'),
            100 * (total('matched_timely_uses') + total('matched_late_uses')) / total('nopf_misses'),
            100 * total('matched_timely_uses') / total('nopf_misses'),
            100 * total('llc_hits') / total('llc_lookups'),
            100 * total('llc_misses') / total('llc_lookups'),
            1000 * total('llc_misses') / total('instructions'))


def comparison(directory):
    return list(rows([directory]))


def development(path):
    model = json.loads((EXPERIMENTS / path / 'best.json').read_text())
    return model['total']


def main():
    data = read_csv(RESULTS / 'per_phase.csv')
    by_phase = {phase: [row for row in data if int(row['region']) == phase] for phase in PHASES}
    native = [native_best(by_phase[phase], 'ipc') for phase in PHASES]
    lstm = [next(row for row in by_phase[phase] if row['policy'] == 'lstm') for phase in PHASES]
    control = [next(row for row in by_phase[phase] if row['policy'] == 'event') for phase in PHASES]
    streamer = [next(row for row in by_phase[phase] if row['policy'] == 'streamer') for phase in PHASES]
    macros = {
        'NativeTimeSaved': f'{100 * speed(lstm, native):.2f}',
        'NativeTrafficSaved': f'{100 * (1 - sum_key(lstm, "dram_bytes") / sum_key(native, "dram_bytes")):.2f}',
        'FixedSaved': f'{100 * speed(lstm, streamer):.2f}',
        'EventSaved': f'{100 * speed(lstm, control):.2f}',
        'NativeBars': ','.join(
            f'{phase}/{4-index}/{100 * speed([model], [best]):.2f}/{float(model["ipc"]):.4f}/'
            f'{float(best["ipc"]):.4f}/{DISPLAY[best["policy"]]}'
            for index, (phase, model, best) in enumerate(zip(PHASES, lstm, native))),
        'HeavyBW': f'{float(lstm[-1]["dram_bytes_per_cycle"]) * 4:.2f}',
        'HeavyBestBW': f'{float(native[-1]["dram_bytes_per_cycle"]) * 4:.2f}',
    }
    lm, nm = aggregate(lstm), aggregate(native)
    metrics = ('Accuracy', 'Timeliness', 'Direct coverage', 'Timely coverage', 'LLC hit', 'LLC miss')
    macros['MetricBars'] = ','.join(
        f'{{{key}}}/{5-index}/{ours:.2f}/{base:.2f}'
        for index, (key, ours, base) in enumerate(zip(metrics, lm, nm)))
    macros['LstmMpki'], macros['BestMpki'] = f'{lm[-1]:.3f}', f'{nm[-1]:.3f}'
    macros['AbsoluteDev'] = f'{development("lstm_s2_w8"):.1f}'
    macros['RelativeDev'] = f'{development("lstm_relative_s2_w8"):.1f}'
    macros['EmbedDev'] = f'{development("lstm_relative_pc_s2_w8"):.1f}'
    macros['WindowDev'] = ' / '.join(f'{development(f"lstm_relative_pc_s2_w{length}"):.1f}' for length in (16, 32))
    macros['SlotsDev'] = ' / '.join(f'{development(f"lstm_relative_pc_s{slots}_w8"):.1f}' for slots in (1, 2, 4))

    sensitivity = []
    for cap, label in ((64, 'Low'), (256, 'High')):
        cap_native, cap_lstm = [], []
        for phase in PHASES:
            result = comparison(RUNS / f'cap{cap}_{phase}m')
            cap_native.append(native_best(result, 'aggregate_ipc'))
            cap_lstm.append(next(row for row in result if row['policy'] == 'lstm'))
            sensitivity.extend(dict(setting=f'cap{cap}', region=phase, **row) for row in result)
        macros[f'Cap{label}Wins'] = str(sum(model['roi_cycles'] < best['roi_cycles']
                                           for model, best in zip(cap_lstm, cap_native)))
        macros[f'Cap{label}Saved'] = f'{100 * speed(cap_lstm, cap_native):.2f}'
        macros[f'Cap{label}Change'] = f'{-100 * speed(cap_lstm, cap_native):+.2f}'
    slow = []
    for phase in PHASES:
        result = list(rows([RUNS / f'frozen_{phase}m', RUNS / f'latency64_{phase}m'], nopf_only_from_first=True))
        model = next(row for row in result if row['policy'] == 'lstm')
        slow.append(model)
        sensitivity.append(dict(setting='latency64', region=phase, **model))
    macros['LatencyWins'] = str(sum(model['roi_cycles'] < int(best['roi_cycles'])
                                     for model, best in zip(slow, native)))
    zero = []
    for phase in PHASES:
        result = rows([RUNS / f'frozen_{phase}m', RUNS / f'zero_state_{phase}m'], nopf_only_from_first=True)
        model = next(row for row in result if row['policy'] == 'event')
        zero.append(model)
        sensitivity.append(dict(setting='zero_state', region=phase, **model))
    macros['ZeroTimeSaved'] = f'{100 * speed(zero, lstm):.2f}'
    macros['ZeroSends'] = f'{sum_key(zero, "prefetch_issued"):,}'

    mixed_ours, mixed_native = [], []
    for phase in (310, 390):
        result = comparison(RUNS / f'mixed_605_{phase}m_602_0m')
        mixed_ours.append(next(row for row in result if row['policy'] == 'lstm'))
        mixed_native.append(native_best(result, 'aggregate_ipc'))
        sensitivity.extend(dict(setting='605+602', region=phase, **row) for row in result)
    wins = sum(model['roi_cycles'] < best['roi_cycles'] for model, best in zip(mixed_ours, mixed_native))
    macros['MixedResult'] = f'中仍赢 {wins}/2 段，总完成时间缩短 {100 * speed(mixed_ours, mixed_native):.2f}\\%'
    write_csv(RESULTS / 'sensitivity.csv', sensitivity)
    tex = '% Generated by report/build_data.py from the published replay data.\n'
    tex += ''.join(f'\\newcommand{{\\{name}}}{{{value}}}\n' for name, value in macros.items())
    (ROOT / 'report' / 'results.tex').write_text(tex)


if __name__ == '__main__':
    main()
