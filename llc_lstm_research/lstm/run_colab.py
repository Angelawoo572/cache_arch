#!/usr/bin/env python3
"""Prepare the uploaded NoPF archive and start training in a Colab runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


DEVELOPMENT = (10, 30, 50)
VALIDATION = (70,)
TEST = (90, 110)


def execute(*command):
    subprocess.run([sys.executable, '-u', *map(str, command)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('mcf_605_colab_results'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = output / 'data'
    with ZipFile(args.archive) as archive:
        archive.extractall(output)
    paths = {skip: data / '605_{}m.npz'.format(skip) for skip in DEVELOPMENT + VALIDATION + TEST}

    def export(skip):
        if not paths[skip].exists():
            execute(scripts / 'dataset.py', 'export', data / '605_nopf_{}m'.format(skip), paths[skip])

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(export, paths))
    train_files = [paths[skip] for skip in DEVELOPMENT]
    validation_files = [paths[skip] for skip in VALIDATION]
    test_files = [paths[skip] for skip in TEST]
    table = output / 'page_pc.txt'
    execute(scripts / 'dataset.py', 'table', table, *train_files)
    execute(scripts / 'train.py', '--train', *train_files, '--validation', *validation_files,
            '--test', *test_files, '--page-pc', table, '--output', output / 'training',
            '--threads', args.threads, '--epochs', args.epochs)
    final = output.with_suffix('.zip')
    with ZipFile(final, 'w', compression=ZIP_DEFLATED) as archive:
        archive.write(table, 'page_pc.txt')
        for path in sorted((output / 'training').rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(output))
        for path in sorted(data.glob('*.json')):
            archive.write(path, Path('data') / path.name)
    print('Saved results to {}'.format(final), flush=True)


if __name__ == '__main__':
    main()
