#!/usr/bin/env python3
"""Stable CLI entrypoint for the active 623 SPP v19 trainer."""
import json
import sys


if __name__ == "__main__":
    if sys.argv[1:] == ["--describe-model-points"]:
        from model_points_v19 import describe_model_points
        print(json.dumps(describe_model_points(), indent=2, sort_keys=True))
    else:
        from routed_grammar_v19 import main
        main()
