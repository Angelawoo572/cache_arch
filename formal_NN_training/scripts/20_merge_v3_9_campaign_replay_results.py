#!/usr/bin/env python3
"""Merge frozen historical controls and v3.9 challenger replay summaries.

The result always has a best-known row for all five traces. Historical rows are
explicitly labeled `historical_frozen_control`; they are not falsely presented
as fresh v3.9 reruns. New challenger summaries are appended when present, and a
`best_known_after_v3_9` row is selected by IPC for each trace.

Python 3.6 compatible; standard library only.
"""
from __future__ import print_function

import argparse
import csv
import os
from pathlib import Path

FROZEN = {
    "602.gcc_s-734B": ("v3.5", 0.43249),
    "619.lbm_s-4268B": ("v3.1", 0.38492),
    "605.mcf_s-994B": ("v3.5", 0.19322),
    "620.omnetpp_s-874B": ("v3.7_static", 0.24663),
    "623.xalancbmk_s-700B": ("v3.8_candidate_attention", 0.37957),
}


def read_summary(path):
    with open(str(path), newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError("expected one summary row in {} but found {}".format(path, len(rows)))
    row = rows[0]
    if str(row.get("run_failed", "0")) not in ("", "0", "0.0", "False", "false"):
        raise RuntimeError("replay failed according to {}".format(path))
    if str(row.get("replay_transport_ok", "0")) not in ("1", "1.0", "True", "true"):
        raise RuntimeError("replay transport did not validate according to {}".format(path))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=os.path.expanduser("~/cache"))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    artifact = Path(args.artifact_root).resolve()
    plan_path = artifact / "v3_9_primary_replay_plan.csv"
    output = Path(args.output).resolve() if args.output else artifact / "v3_9_all_trace_replay_results.csv"
    rows = []
    for trace, (version, ipc) in sorted(FROZEN.items()):
        rows.append(dict(
            trace=trace,
            row_kind="historical_frozen_control",
            version=version,
            ipc="{:.8f}".format(ipc),
            source_summary="",
            replay_transport_ok="historical",
            note="historical validated replay; not rerun by v3.9 campaign",
        ))

    challengers = {}
    if plan_path.is_file():
        with open(str(plan_path), newline="") as handle:
            for plan in csv.DictReader(handle):
                tag = plan["tag"]
                trace = plan["trace"]
                summary = repo / "formal_NN_training/results/standalone_lstm_replay" / tag / "summary.csv"
                if not summary.is_file():
                    rows.append(dict(
                        trace=trace, row_kind="v3_9_challenger_missing", version=tag,
                        ipc="", source_summary=str(summary), replay_transport_ok="0",
                        note="planned challenger summary is not present yet",
                    ))
                    continue
                data = read_summary(summary)
                ipc = float(data["ipc"])
                challengers[trace] = (tag, ipc, summary, data)
                rows.append(dict(
                    trace=trace, row_kind="v3_9_challenger_replay", version=tag,
                    ipc="{:.8f}".format(ipc), source_summary=str(summary),
                    replay_transport_ok=data.get("replay_transport_ok", ""),
                    note="new specialist challenger; compare against frozen control",
                ))

    for trace, (control_version, control_ipc) in sorted(FROZEN.items()):
        tag, challenger_ipc, summary, _data = challengers.get(trace, ("", None, None, None))
        if challenger_ipc is not None and challenger_ipc > control_ipc:
            version, ipc, note = tag, challenger_ipc, "new challenger exceeds frozen control"
        else:
            version, ipc = control_version, control_ipc
            note = "frozen control retained" if challenger_ipc is None else "frozen control retained; challenger did not exceed IPC"
        rows.append(dict(
            trace=trace, row_kind="best_known_after_v3_9", version=version,
            ipc="{:.8f}".format(ipc), source_summary=str(summary or ""),
            replay_transport_ok="1" if challenger_ipc is not None else "historical", note=note,
        ))

    fields = ["trace", "row_kind", "version", "ipc", "source_summary", "replay_transport_ok", "note"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("[saved] {}".format(output))


if __name__ == "__main__":
    main()
