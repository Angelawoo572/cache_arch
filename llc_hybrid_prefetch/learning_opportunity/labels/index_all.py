"""Index aligned admissions from additional native policy runs as they finish."""

import argparse
import time

from build import ARMS, ReferenceCache, build, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--memory", action="store_true", help="Use RAM for transient arrays"
    )
    parser.add_argument(
        "--mask-cache-mib",
        type=int,
        default=0,
        help="Max MiB of retired masks retained between arms of one case",
    )
    args = parser.parse_args()
    if args.mask_cache_mib and not args.memory:
        parser.error("--mask-cache-mib requires --memory")
    cache = ReferenceCache(args.mask_cache_mib) if args.mask_cache_mib else None
    pending = [(c, a) for c in args.cases for a in args.arms]
    while pending:
        for case, arm in pending[:]:
            try:
                sources(case, arm)
            except FileNotFoundError:
                if not args.wait:
                    raise
                continue
            print(build(case, arm, memory=args.memory, mask_cache=cache), flush=True)
            pending.remove((case, arm))
        if pending:
            time.sleep(10)


if __name__ == "__main__":
    main()
