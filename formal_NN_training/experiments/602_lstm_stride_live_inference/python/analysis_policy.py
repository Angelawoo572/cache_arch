#!/usr/bin/env python3
"""Shared, explicit policies for 20M equivalence and stable plateaus."""

import math


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def relative_deviation(value, reference):
    if not valid_number(value) or not valid_number(reference) or reference == 0:
        return None
    return abs(float(value) - float(reference)) / abs(float(reference))


def first(rows, predicate):
    return next((row for row in rows if predicate(row)), None)


def reference_20m(rows, field="ipc"):
    return first(
        reversed(sorted(rows, key=lambda row: row["instruction_budget"])),
        lambda row: (
            row.get("instruction_budget") == 20000000
            and valid_number(row.get(field))
        ),
    )


def smallest_within(rows, reference, tolerance, field="ipc"):
    """First point with two-sided relative error no greater than tolerance."""
    if reference is None or not valid_number(reference.get(field)):
        return None
    reference_value = reference[field]
    return first(
        sorted(rows, key=lambda row: row["instruction_budget"]),
        lambda row: (
            relative_deviation(row.get(field), reference_value) is not None
            and relative_deviation(row.get(field), reference_value) <= tolerance
        ),
    )


def smallest_reaching_at_least(rows, reference, fraction, field="ipc"):
    """First point reaching a one-sided fraction of the reference value."""
    if reference is None or not valid_number(reference.get(field)):
        return None
    target = float(reference[field]) * float(fraction)
    return first(
        sorted(rows, key=lambda row: row["instruction_budget"]),
        lambda row: valid_number(row.get(field)) and row[field] >= target,
    )


def stable_plateau(rows, reference, tolerance=0.005, field="ipc"):
    """Return the first stable candidate and its observed suffix.

    A candidate must precede the 20M reference, have at least one subsequent
    observed point, and every observed point from the candidate through 20M
    must be within the two-sided tolerance of the 20M value. Missing/unrun
    budgets are never silently treated as stable observations.
    """
    if reference is None or not valid_number(reference.get(field)):
        return None, []
    reference_budget = reference["instruction_budget"]
    observed = sorted(
        (
            row for row in rows
            if row.get("instruction_budget") <= reference_budget
            and valid_number(row.get(field))
        ),
        key=lambda row: row["instruction_budget"],
    )
    if not observed or observed[-1]["instruction_budget"] != reference_budget:
        return None, []
    for index, candidate in enumerate(observed[:-1]):
        suffix = observed[index:]
        if all(
            relative_deviation(row[field], reference[field]) <= tolerance
            for row in suffix
        ):
            return candidate, suffix
    return None, []


def stability_evidence(rows, reference, fields, tolerance=0.05):
    """Report auxiliary closeness without using it to redefine IPC plateau."""
    evidence = {}
    for field in fields:
        reference_value = reference.get(field) if reference else None
        deviations = []
        missing = []
        for row in rows:
            deviation = relative_deviation(row.get(field), reference_value)
            if deviation is None:
                missing.append(row.get("budget_tag"))
            else:
                deviations.append((row.get("budget_tag"), deviation))
        evidence[field] = {
            "reference_value": reference_value,
            "relative_tolerance": tolerance,
            "max_relative_deviation": (
                max(item[1] for item in deviations) if deviations else None
            ),
            "all_observed_within_tolerance": (
                all(item[1] <= tolerance for item in deviations)
                if deviations and not missing else None
            ),
            "missing_budget_tags": missing,
        }
    return evidence
