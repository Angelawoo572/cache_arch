"""Publish the measured corpus inventory after both registered panels pass their source audits."""

import csv

from build import HERE
from run import CASES, POLICIES, external_cases


def table(path):
    with (HERE / path).open(newline="") as file:
        return list(csv.DictReader(file))


def panel(name, prefix, cases, policies):
    captures = table(prefix + "captures.csv")
    source = {row["case"]: row for row in table(prefix + "source_parity.csv")}
    expected = {(c, p) for c in cases for p in policies}
    if {(r["case"], r["policy"]) for r in captures} != expected or set(source) != set(
        cases
    ):
        raise ValueError("A registered validation is missing")
    for capture in captures:
        if int(capture["source_load_rows"]) != int(
            source[capture["case"]]["exact_operand_keys_per_policy"]
        ):
            raise ValueError("Source and native operand populations disagree")

    def number(key):
        return sum(int(r[key]) for r in captures)

    unique_source = sum(
        int(r["exact_operand_keys_per_policy"]) for r in source.values()
    )
    result = f"| {name} | {len(source)} | {len(captures)} | {unique_source:,} | {number('arrival_rows'):,} | {number('compressed_stream_bytes') / 2**30:.2f} |"
    events = [
        number(k + "_rows")
        for k in ("source_load", "cache", "offer", "head_wait", "prefetch_victim")
    ]
    event_row = f"| {name} | " + " | ".join(f"{v:,}" for v in events) + " |"
    return result, event_row


def main():
    primary, primary_events = panel("Primary", "", CASES, POLICIES)
    external, external_events = panel(
        "External", "external_", external_cases(), ["stride"]
    )
    report = f"""# Measured data

Each run warms the machine for 50M instructions, then records 200M retired instructions on one active core of the two-core machine. The primary panel covers nine windows from seven recordings. Every primary window has all four native policies. The eleven external recordings add a rich Stride stream at placement slot 0. Their four conventional comparisons come from the [expansion study](../expansion/README.md).

| Panel | Windows | Rich runs | Source load operands, once per window | LLC admissions across runs | Indexed gzip streams (GiB) |
|---|---:|---:|---:|---:|---:|
{primary}
{external}

An LLC admission appears once in the arrival stream and has a pressure snapshot at the same row. The gzip totals include the seven rich streams. They exclude the source traces, baseline outcome masks and derived label indexes.

| Panel | Source load rows across runs | Other cache callbacks | Offer lifecycle records | Positive retirement-wait records | Prefetch victims |
|---|---:|---:|---:|---:|---:|
{primary_events}
{external_events}

[Primary trace and run inventory](captures.csv) · [External trace and run inventory](external_captures.csv) · [Schema and raw run paths](README.md)

The [primary](replicas.csv) and [external](external_replicas.csv) replica tables list runs whose two site certificates match. Each reports the payload origin, row population, retirement outcomes and fixed-work cycles.

## Which loads reach the LLC?

![Original workloads under NoPF](source_visibility.svg)

![External workloads under Stride](external_source_visibility.svg)

Each percentage counts distinct dynamic source operands admitted to the LLC divided by all load operands in the measured instruction sequence. Every admitted line in these two plots was checked against its original PC, virtual operand and simulated page map. A load absorbed by the private caches contributes to the denominator. It supplies no LLC admission on which a predictor at this interface could trigger. The two panels use different policies.

[Primary source checks](source_parity.csv) · [External source checks](external_source_parity.csv) · [Plot script](plot_visibility.py)
"""
    (HERE / "MEASURED_DATA.md").write_text(report)


if __name__ == "__main__":
    main()
