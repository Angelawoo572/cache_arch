#!/usr/bin/env python3
"""Run retained study scripts using canonical paths, without old filesystem aliases.

Historical JSON bytes stay unchanged. Only decoded path values are translated
inside this process; model values, protocol IDs and simulation controls are not
changed. --local also maps the recorded Sacramento cache root to this cache.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--check', action='store_true', help='read-only frozen source/model/build gate')
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cache = next(p for p in Path(__file__).resolve().parents if p.name == 'cache')
    script = args.script.resolve()
    if not script.is_file() or not script.is_relative_to(cache):
        raise ValueError('Study script must exist inside cache')
    original = json.loads

    def canonical(value):
        if isinstance(value, str):
            value = value.replace('/cache/docs/formal_study_20261016', '/cache/formal_study/archive/20261016/report')
            for day in ('20261016', '20261017'):
                value = value.replace('/cache/formal_study_' + day, '/cache/formal_study/archive/' + day)
            if args.local and (value == '/home/qianruw/cache' or value.startswith('/home/qianruw/cache/')):
                value = str(cache) + value[len('/home/qianruw/cache'):]
            return value
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        return value

    def loads(*values, **options):
        return canonical(original(*values, **options))

    json.loads = loads
    previous = sys.argv
    sys.argv = [str(script), *args.arguments]
    try:
        if args.check:
            namespace = runpy.run_path(str(script), run_name='canonical_path_check')
            namespace['require_gate']()
            print('PASS: canonical source/model/build bindings; no dated aliases required')
        else:
            runpy.run_path(str(script), run_name='__main__')
    finally:
        json.loads = original
        sys.argv = previous


if __name__ == '__main__':
    main()
