#!/usr/bin/env python3
"""Validate real completed paired trainings using saved tensors, without hashes."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def tensor_equal(left, right):
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def validate(paths):
    groups = defaultdict(dict)
    for path in paths:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        for field in ("seed", "emit_loss", "training_config", "initial_state_dict"):
            if field not in value:
                raise RuntimeError("{} lacks {}".format(path, field))
        key = (value["trace"], int(value["hidden_size"]), int(value["seed"]))
        loss = value["emit_loss"]
        if loss in groups[key]:
            raise RuntimeError("duplicate loss arm for {}".format(key))
        groups[key][loss] = (path, value)
    rows = []
    initial_by_workload = defaultdict(dict)
    for key, arms in sorted(groups.items()):
        if set(arms) != {"balanced", "unweighted"}:
            raise RuntimeError("missing paired arm for {}".format(key))
        left_path, left = arms["balanced"]
        right_path, right = arms["unweighted"]
        if not tensor_equal(left["initial_state_dict"], right["initial_state_dict"]):
            raise RuntimeError("paired initialization differs for {}".format(key))
        left_config = dict(left["training_config"])
        right_config = dict(right["training_config"])
        left_config.pop("emit_loss")
        right_config.pop("emit_loss")
        if left_config != right_config:
            raise RuntimeError("paired data or training budgets differ for {}".format(key))
        if left_config["optimizer_steps"] <= 0:
            raise RuntimeError("no completed optimizer steps for {}".format(key))
        if tensor_equal(left["initial_state_dict"], left["state_dict"]):
            raise RuntimeError("balanced checkpoint was never updated for {}".format(key))
        if tensor_equal(right["initial_state_dict"], right["state_dict"]):
            raise RuntimeError("unweighted checkpoint was never updated for {}".format(key))
        workload_key = key[:2]
        for other_seed, other_initial in initial_by_workload[workload_key].items():
            if tensor_equal(other_initial, left["initial_state_dict"]):
                raise RuntimeError("seeds {} and {} share initialization".format(other_seed, key[2]))
        initial_by_workload[workload_key][key[2]] = left["initial_state_dict"]
        rows.append({"trace": key[0], "hidden_size": key[1], "seed": key[2],
                     "balanced": str(left_path), "unweighted": str(right_path),
                     "paired_initialization_exact": True,
                     "paired_data_and_training_budget_exact": True,
                     "optimizer_steps_per_arm": left_config["optimizer_steps"],
                     "final_weights_differ": not tensor_equal(left["state_dict"], right["state_dict"])})
    return {"status": "PASS", "pairs": rows,
            "distinct_training_seeds_per_workload": [
                {"trace": key[0], "hidden_size": key[1], "seeds": sorted(values)}
                for key, values in sorted(initial_by_workload.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.checkpoints)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
