#!/usr/bin/env python3
"""Prepare a trace-specific LSTM action CSV for ChampSim replay.

This script prevents the replay-index bug that happened during 602/619 debugging.
It can:

1. restore full_lstm_cache_actions.csv from packed split gzip parts,
2. validate that the file belongs to the requested trace,
3. merge replay_access_idx from lstm_events_<TRACE>.csv when the Colab output is missing it,
4. copy the prepared file to formal_NN_training/artifacts/full_lstm_cache_actions.csv,
   which is the default path used by 03_run_lstm_replay.sh.

Important: Colab outputs may be a subset of the event CSV. For example, a trace can
start its exported action rows at event_id=31 rather than event_id=0. Therefore this
script merges replay_access_idx by event_id, not by row number.

Example:
  python3 formal_NN_training/scripts/07_prepare_actions_for_replay.py \
    --trace 619.lbm_s-4268B \
    --copy-default
"""

import argparse
import csv
import gzip
import shutil
from collections import Counter
from pathlib import Path


def trace_tag(trace: str) -> str:
    return trace.split(".", 1)[0]


def to_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def restore_from_packed(packed_dir: Path, out_csv: Path) -> bool:
    """Restore full_lstm_cache_actions.csv from packed .gz or .gz.part_* files."""
    gz = packed_dir / "full_lstm_cache_actions.csv.gz"
    parts = sorted(packed_dir.glob("full_lstm_cache_actions.csv.gz.part_*"))

    if gz.exists():
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"[restore] {gz} -> {out_csv}")
        with gzip.open(gz, "rb") as fin, out_csv.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        return True

    if parts:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"[restore] {len(parts)} split parts under {packed_dir} -> {out_csv}")
        with out_csv.open("wb") as fout:
            gz_stream = b"".join(p.read_bytes() for p in parts)
            fout.write(gzip.decompress(gz_stream))
        return True

    return False


def sample_validate_actions(actions: Path, trace: str, limit: int):
    with actions.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        traces = Counter()
        blank = 0
        nonblank = 0
        examples = []
        n = 0
        for row in reader:
            n += 1
            traces[row.get("trace", "")] += 1
            ridx = row.get("replay_access_idx", "")
            if ridx == "":
                blank += 1
            else:
                nonblank += 1
            if len(examples) < 5:
                examples.append(
                    (
                        row.get("trace"),
                        row.get("event_id"),
                        row.get("replay_access_idx"),
                        row.get("prefetch_addr") or row.get("pf_addr") or row.get("candidate_addr"),
                        row.get("pred_good_prefetch_prob"),
                    )
                )
            if n >= limit:
                break

    print(f"[validate] {actions}")
    print(f"[fields] {fields[:30]}")
    print(f"[checked] {n}")
    print(f"[traces] {dict(traces)}")
    print(f"[blank replay_access_idx] {blank}")
    print(f"[nonblank replay_access_idx] {nonblank}")
    print("[examples]")
    for e in examples:
        print("  ", e)

    wrong_trace = sum(v for k, v in traces.items() if k != trace)
    return {
        "fields": fields,
        "checked": n,
        "blank": blank,
        "nonblank": nonblank,
        "wrong_trace": wrong_trace,
    }


def event_addr(row):
    return row.get("addr_int", row.get("addr", ""))


def action_addr(row):
    return row.get("addr_int", row.get("addr", ""))


def merge_replay_idx_by_event_id(events: Path, actions: Path, out: Path, trace: str):
    """Merge replay_access_idx from events into actions by event_id.

    This intentionally allows actions to be a strict subset of events. Both files are
    expected to be ordered by event_id, which keeps memory use low even for large traces.
    """
    if not events.exists():
        raise SystemExit(f"[error] missing events CSV: {events}")
    if not actions.exists():
        raise SystemExit(f"[error] missing actions CSV: {actions}")

    tmp = out.with_suffix(out.suffix + ".tmp")
    rows = 0
    blank = 0
    missing = 0
    trace_bad = 0
    addr_mismatch = 0
    examples = []

    with actions.open(newline="") as fa, events.open(newline="") as fe, tmp.open("w", newline="") as fo:
        ra = csv.DictReader(fa)
        re = csv.DictReader(fe)

        action_fields = list(ra.fieldnames or [])
        event_fields = list(re.fieldnames or [])

        if "event_id" not in action_fields:
            raise SystemExit("[error] action CSV missing event_id")
        if "event_id" not in event_fields:
            raise SystemExit("[error] events CSV missing event_id")
        if "replay_access_idx" not in event_fields:
            raise SystemExit("[error] events CSV missing replay_access_idx")

        fields = list(action_fields)
        if "replay_access_idx" not in fields:
            pos = fields.index("event_id") + 1
            fields = fields[:pos] + ["replay_access_idx"] + fields[pos:]

        writer = csv.DictWriter(fo, fieldnames=fields)
        writer.writeheader()

        event_row = next(re, None)
        event_eid = to_int(event_row.get("event_id")) if event_row else None

        for a in ra:
            rows += 1
            if a.get("trace") != trace:
                trace_bad += 1

            a_eid = to_int(a.get("event_id"))
            if a_eid is None:
                missing += 1
                if len(examples) < 5:
                    examples.append(("bad_action_event_id", a.get("event_id")))
                continue

            while event_row is not None and event_eid is not None and event_eid < a_eid:
                event_row = next(re, None)
                event_eid = to_int(event_row.get("event_id")) if event_row else None

            if event_row is None or event_eid != a_eid:
                missing += 1
                if len(examples) < 5:
                    examples.append(("missing_event_id", a.get("event_id"), "current_event", event_eid))
                continue

            ridx = event_row.get("replay_access_idx", "")
            if ridx == "":
                blank += 1
            a["replay_access_idx"] = ridx

            aa = action_addr(a)
            ea = event_addr(event_row)
            if aa and ea and aa != ea:
                addr_mismatch += 1
                if len(examples) < 5:
                    examples.append(("addr_mismatch", a.get("event_id"), aa, ea))

            writer.writerow({k: a.get(k, "") for k in fields})

    print(
        f"[merge-by-event-id] rows={rows} blank_replay_access_idx={blank} "
        f"missing_event_id={missing} bad_trace_rows={trace_bad} addr_mismatch={addr_mismatch}"
    )
    if examples:
        print("[merge examples]")
        for e in examples:
            print("  ", e)

    if missing:
        raise SystemExit("[error] some action event_id values were not found in events")
    if blank:
        raise SystemExit("[error] replay_access_idx still blank after merge")
    if trace_bad:
        raise SystemExit("[error] wrong trace rows in action file")
    if addr_mismatch:
        raise SystemExit("[error] addr mismatch after event_id merge")

    tmp.replace(out)
    return True


def merge_replay_idx(events: Path, actions: Path, out: Path, trace: str, force: bool = False):
    before = sample_validate_actions(actions, trace, limit=100000)
    if before["wrong_trace"]:
        raise SystemExit(f"[error] action file contains rows from a different trace, expected {trace}")

    if before["blank"] == 0 and "replay_access_idx" in before["fields"] and not force:
        print("[merge] action CSV already has replay_access_idx; no merge needed")
        return False

    return merge_replay_idx_by_event_id(events, actions, out, trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--actions", type=Path, default=None)
    ap.add_argument("--packed-dir", type=Path, default=None)
    ap.add_argument("--restore-packed", action="store_true", help="Always restore actions from packed output before validating/merging")
    ap.add_argument("--copy-default", action="store_true", help="Copy prepared by_trace action CSV to artifacts/full_lstm_cache_actions.csv")
    ap.add_argument("--force-merge", action="store_true", help="Merge replay_access_idx even if the action CSV already has the column")
    ap.add_argument("--sample-limit", type=int, default=100000)
    args = ap.parse_args()

    root = args.root.resolve()
    trace = args.trace
    tag = trace_tag(trace)

    events = args.events or root / f"formal_NN_training/data/generated/lstm_events_{trace}.csv"
    actions = args.actions or root / f"formal_NN_training/artifacts/by_trace/{trace}/full_lstm_cache_actions.csv"
    packed_dir = args.packed_dir or root / f"formal_NN_training/artifacts/packed/{tag}"
    default_actions = root / "formal_NN_training/artifacts/full_lstm_cache_actions.csv"

    print("============================================================")
    print("PREPARE ACTIONS FOR REPLAY")
    print("trace     :", trace)
    print("events    :", events)
    print("actions   :", actions)
    print("packed dir:", packed_dir)
    print("default   :", default_actions)
    print("============================================================")

    if args.restore_packed:
        if actions.exists():
            backup = actions.with_suffix(".before_restore.csv")
            shutil.copy2(actions, backup)
            print(f"[backup-before-restore] {backup}")
        restored = restore_from_packed(packed_dir, actions)
        if not restored:
            raise SystemExit(f"[error] --restore-packed set but no packed file found under: {packed_dir}")
    elif not actions.exists():
        restored = restore_from_packed(packed_dir, actions)
        if not restored:
            raise SystemExit(f"[error] actions missing and no packed file found: {actions}")

    backup = actions.with_suffix(".no_replay_idx.csv")
    if not backup.exists() and actions.exists():
        shutil.copy2(actions, backup)
        print(f"[backup] {backup}")

    merged = merge_replay_idx(events, actions, actions, trace, force=args.force_merge)
    if merged:
        print(f"[write] merged replay_access_idx into {actions}")

    after = sample_validate_actions(actions, trace, limit=args.sample_limit)
    if after["blank"] != 0:
        raise SystemExit("[error] prepared action CSV still has blank replay_access_idx")

    if args.copy_default:
        default_actions.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actions, default_actions)
        print(f"[copy-default] {actions} -> {default_actions}")
        sample_validate_actions(default_actions, trace, limit=min(args.sample_limit, 10))

    print("[done] actions are ready for 03_run_lstm_replay.sh")


if __name__ == "__main__":
    main()
