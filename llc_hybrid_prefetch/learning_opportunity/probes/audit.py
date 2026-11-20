"""Audit source/destination coverage and conventional joins in the published probe tables."""

import argparse
import csv

import numpy as np
import pandas as pd

from extract import HERE, STUDY
from fit import REPLAY
from suite import ENCODINGS, MODES, pairs

SCOPES = ("same_pc", "local_region", "global32")
STATUS = ("no_tag", "ordinary", "persistent", "late", "timely")


def main(require_complete):
    scores = pd.read_csv(HERE / "scores.csv")
    native = pd.read_csv(HERE / "conventional_outcomes.csv")
    pooled_path = HERE / "pooled_transfer.csv"
    pooled = (
        pd.read_csv(pooled_path)
        if pooled_path.is_file()
        else pd.DataFrame(columns=["destination", "scope", "cohort"])
    )
    data = pd.read_csv(HERE / "sampling.csv").set_index("case")
    identifiers = ["source", "destination", "split", "horizon", "encoding", "scope"]
    if scores.duplicated(identifiers).any():
        raise ValueError("Duplicated score")
    if not (
        scores.deadline.le(scores.hit)
        & scores.hit.le(scores.correct)
        & scores.correct.le(scores.scored)
        & scores.scored.le(scores.anchors)
    ).all():
        raise ValueError("Physical address count exceeds eligible denominator")
    if not (
        scores.forced_deadline.le(scores.forced_hit)
        & scores.forced_hit.le(scores.forced_correct)
        & scores.forced_correct.le(scores.anchors)
    ).all():
        raise ValueError("Forced-address count exceeds eligible denominator")
    for r in scores.itertuples():
        if int(r.anchors) != int(data.loc[r.destination, r.split]) or not np.isfinite(
            r.mean_bits
        ):
            raise ValueError(
                "Score population differs from sampled split or has undefined loss"
            )
    with (STUDY / "expansion/sources.csv").open() as f:
        registered = [r["case"] for r in csv.DictReader(f)]
    incomplete = []
    for source, dest in pairs():
        rows = scores[
            (scores.source == source)
            & (scores.destination == dest)
            & (scores.split == "test")
            & scores.scope.isin(MODES)
            & scores.encoding.isin(ENCODINGS)
        ]
        if len(rows) != len(MODES) * len(ENCODINGS) * 3:
            incomplete.append(f"{source} -> {dest} score matrix")
        if dest not in registered:
            continue
        native_rows = native[(native.source == source) & (native.destination == dest)]
        expected = {
            (arm, scope, status)
            for arm in REPLAY[1:]
            for scope in SCOPES
            for status in STATUS
        }
        observed = set(
            zip(native_rows.arm, native_rows.scope, native_rows.original_miss_outcome)
        )
        if observed != expected:
            incomplete.append(
                f"{source} -> {dest} outcome joins ({len(observed)}/{len(expected)})"
            )
    feedback_path = HERE / "external_feedback.csv"
    feedback = (
        pd.read_csv(feedback_path)
        if feedback_path.is_file()
        else pd.DataFrame(columns=["destination", "scope"])
    )
    for name in registered:
        pooled_native = native[
            (native.source == "pooled_three") & (native.destination == name)
        ]
        expected = {(a, s, k) for a in REPLAY[1:] for s in SCOPES for k in STATUS}
        if (
            set(
                zip(
                    pooled_native.arm,
                    pooled_native.scope,
                    pooled_native.original_miss_outcome,
                )
            )
            != expected
        ):
            incomplete.append(f"{name} pooled native outcomes")
        if set(feedback[feedback.destination == name].scope) != {
            "with_time",
            "with_feedback",
            "with_stride_feedback",
        }:
            incomplete.append(f"{name} rich feedback transfer")
        if len(pooled[pooled.destination == name]) != 12:
            incomplete.append(f"{name} pooled transfer")
    with (STUDY / "domains/sources.csv").open() as f:
        domain_cases = [r["case"] for r in csv.DictReader(f)]
    for name, count in (("domain_transfer", 9), ("domain_outcomes", 45)):
        path = HERE / f"{name}.csv"
        domain_rows = (
            pd.read_csv(path)
            if path.exists()
            else pd.DataFrame(columns=["destination"])
        )
        for case in domain_cases:
            if len(domain_rows[domain_rows.destination == case]) != count:
                incomplete.append(f"{case} {name}")
    if require_complete and incomplete:
        raise ValueError("Still outstanding: " + ", ".join(incomplete))
    print(
        f"Verified {len(scores):,} score rows; {len(incomplete)} outstanding matrix/arm/pooled collections"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args().require_complete)
