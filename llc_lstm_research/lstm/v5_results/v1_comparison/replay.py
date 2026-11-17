#!/usr/bin/env python3
"""Replay the original Colab V1 and the pure-NoPF-32 model on common settings."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'conventional/runs/v1_newest_replay'
BINARY = ROOT / 'conventional/runs/build-v5-direct-local/pythia/bin/champsim'
OLD_REGIONS = (90, 110)
FRESH_REGIONS = (410, 430, 450)
REGIONS = OLD_REGIONS + FRESH_REGIONS
CONDITIONS = ('native', 'cap1000', 'cap150')
V1 = ROOT / 'lstm/runs/experiments/mcf_605_colab/training/lstm.txt'
NEWEST = ROOT / 'lstm/v5_results/clean32/lstm_model.txt'


def anchor(region):
    if region in OLD_REGIONS:
        return ROOT / f'conventional/runs/v5_coverage_replay/colab_reference_nopf_{region}m'
    return ROOT / f'conventional/runs/v5_clean/label32/fresh/{region}m'


def trace_for(region):
    if region in OLD_REGIONS:
        return ROOT / f'conventional/runs/v5_coverage_replay/trace_slices/{region}m/605.mcf_s-994B.champsimtrace.gz'
    return ROOT / f'lstm/runs/v6/trace_slices/{region}m/605.mcf_s-994B.champsimtrace.gz'


def target(region, condition, arm):
    return RUNS / condition / f'{region}m' / arm


def run(region, condition, arm):
    output = target(region, condition, arm)
    weights, threshold = (V1, -1037) if arm == 'v1' else (NEWEST, -950)
    cmd = [sys.executable, str(ROOT / 'conventional/run.py'), '--binary', str(BINARY), '--output', str(output),
           '--warmup', '200000', '--instructions', '2000000', '--degree', '1', '--events', '--prefetch-events',
           '--policy', 'lstm', '--lstm-model', str(weights), f'--lstm-threshold={threshold}',
           '--prediction-latency', '32', '--prediction-interval', '8', '--prediction-depth', '4',
           '--prediction-state-cycles', '8']
    if arm == 'newest':
        cmd.append('--position-scorer')
    if condition.startswith('cap'):
        cmd += ['--issue-limit', condition[3:]]
    cmd += [str(trace_for(region))] * 2
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'launcher.log').open('w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result, = json.loads((output / 'summary.json').read_text())
    print(region, condition, arm, result['prefetch_issued'], result['aggregate_ipc'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--regions', nargs='+', type=int, choices=REGIONS, default=REGIONS)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    cases = [(region, condition, arm) for region in args.regions
             for condition in args.conditions for arm in ('v1', 'newest')]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in as_completed(pool.submit(run, *case) for case in cases):
            result.result()


if __name__ == '__main__':
    main()
