"""Estimate address predictability on frozen chronological splits and new recordings."""

import argparse
from collections import Counter
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from extract import PREVIOUS, RAW, STUDY

HORIZONS = (1, 8, 32)
WIDTH = 64
OTHER = -(1 << 60)
SCOPES = (
    "prior",
    "current",
    "current_delta",
    "same_pc",
    "region",
    "global4",
    "global8",
    "global16",
    "global32",
    "global64",
    "shuffled",
    "no_pc_ids",
    "with_time",
    "local_region",
    "hide_other_pc",
    "with_pressure",
    "with_core",
    "with_feedback",
    "with_stride_feedback",
)
INFO = ("delta", "offset", "samepc", "samepage", "sameregion")
REPLAY = ("nopf", "stride", "multi", "spp")


def load(case):
    with np.load(RAW / f"{case}__stride.npz", allow_pickle=False) as data:
        total = int(data["total_arrivals"][0])
        frame = pd.DataFrame({k: data[k] for k in data.files if k != "total_arrivals"})
    pressure_file = RAW / f"{case}__pressure.npz"
    if pressure_file.is_file():
        with np.load(pressure_file, allow_pickle=False) as extra:
            if not np.array_equal(frame["key"].to_numpy(), extra["key"]):
                raise ValueError("Pressure key mismatch")
            frame["system_state"] = extra["system_state"]
    side_file = RAW / f"{case}__sideband.npz"
    if side_file.is_file():
        with np.load(side_file, allow_pickle=False) as side:
            if not np.array_equal(frame["key"].to_numpy(), side["key"]):
                raise ValueError("Sideband key mismatch")
            for k in side.files:
                if k.startswith("core_"):
                    frame[k] = side[k]
    feedback_file = RAW / f"{case}__feedback.npz"
    if feedback_file.is_file():
        with np.load(feedback_file, allow_pickle=False) as feedback:
            if not np.array_equal(frame["key"].to_numpy(), feedback["key"]):
                raise ValueError("Feedback key mismatch")
            for k in feedback.files:
                if k.startswith("prior_"):
                    frame[k] = feedback[k]
    if np.any(np.diff(frame["index"]) <= 0):
        raise ValueError("Unordered anchors")
    return frame, total


def splits(frame, count):
    idx = frame["index"].to_numpy()
    first, second = 3 * count // 5, 4 * count // 5
    return {
        "train": (idx + 32 < first),
        "dev": (idx >= first) & (idx + 32 < second),
        "test": (idx >= second) & (idx + 32 < count),
    }


def signed_log(arr):
    x = np.asarray(arr, dtype=np.float64)
    return np.sign(x) * np.log2(1 + np.abs(x))


def pc_ids(values, reference):
    values = np.asarray(values)
    idx = np.searchsorted(reference, values)
    valid = idx < len(reference)
    return np.where(
        valid & (reference[np.minimum(idx, len(reference) - 1)] == values), idx + 1, 0
    )


def features(frame, mode, reference):
    columns = [frame["line"].to_numpy() % 64]
    if mode != "no_pc_ids":
        columns.append(pc_ids(frame.pc, reference))
    if mode == "current_delta":
        columns.append(signed_log(frame.p1_delta.to_numpy()))
    if mode in ("current", "current_delta"):
        return np.column_stack(columns).astype("float32")

    groups = []
    if mode != "region":
        groups.extend("p" + str(i) for i in (1, 2, 4, 8))
    if mode != "same_pc":
        groups.extend("r" + str(i) for i in (1, 2, 4, 8))
    if mode.startswith("global"):
        width = int(mode[6:])
    elif mode in {
        "no_pc_ids",
        "shuffled",
        "with_time",
        "hide_other_pc",
        "with_pressure",
        "with_core",
        "with_feedback",
        "with_stride_feedback",
    }:
        width = 32
    else:
        width = 0
    groups.extend("g" + str(i) for i in range(1, width + 1))
    permutation = None
    if mode == "shuffled":
        rng = np.random.default_rng(20261119)
        permutation = np.argsort(rng.random((len(frame), width)), axis=1)
    for field in (*INFO, "pc", "dt"):
        if field == "dt" and mode not in (
            "with_time",
            "with_pressure",
            "with_core",
            "with_feedback",
            "with_stride_feedback",
        ):
            continue
        if field == "pc" and mode == "no_pc_ids":
            continue
        selected = [g for g in groups if field != "pc" or not g.startswith("p")]
        matrix = frame[[g + "_" + field for g in selected]].to_numpy()
        if mode == "hide_other_pc" and field != "samepc":
            visible = (
                frame[[f"g{i}_samepc" for i in range(1, width + 1)]].to_numpy() == 1
            )
            matrix = matrix.copy()
            matrix[:, -width:] = np.where(visible, matrix[:, -width:], -1000000000)
        if permutation is not None:
            matrix = matrix.copy()
            matrix[:, -width:] = np.take_along_axis(matrix[:, -width:], permutation, 1)
        if field == "pc":
            matrix = pc_ids(matrix, reference)
        elif field in ("delta", "dt"):
            matrix = signed_log(matrix)
        columns.append(matrix)
    if mode in ("with_feedback", "with_stride_feedback"):
        prefix = "prior_cache" if mode == "with_feedback" else "prior_"
        names = [k for k in frame if k.startswith(prefix)]
        if not names:
            raise ValueError("Native cache/offer callback streams required")
        columns.append(frame[names].to_numpy())
    if mode == "with_core":
        names = [k for k in frame if k.startswith("core_")]
        if not names:
            raise ValueError("Core metadata required for sideband ablation")
        columns.append(frame[names].to_numpy())
    if mode == "with_pressure":
        if "system_state" not in frame:
            raise ValueError("Capture required for pressure ablation")
        raw = frame.system_state.to_numpy()
        columns.extend((raw >> (8 * i)) & 255 for i in range(6))
    return np.column_stack(columns).astype("float32")


def action_codes(frame, h, encoding):
    future = frame[f"y{h}_line"].to_numpy()
    if encoding == "delta":
        return future - frame.line.to_numpy()
    bank = frame[[f"bank{i}" for i in range(8)]].to_numpy()
    match = bank == (future // 64)[:, None]
    return np.where(match.any(axis=1), 64 * match.argmax(axis=1) + future % 64, OTHER)


def encode(values, vocabulary):
    ids = np.searchsorted(vocabulary, values)
    equal = vocabulary[np.minimum(ids, len(vocabulary) - 1)] == values
    return np.where((ids < len(vocabulary)) & equal, ids + 1, 0)


def train(frame, mode, ref, h, encoding):
    actions = action_codes(frame, h, encoding)
    counts = Counter(int(value) for value in actions if value != OTHER)
    vocab = np.asarray(
        sorted(
            k for k, _ in sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:WIDTH]
        ),
        dtype=np.int64,
    )
    if not len(vocab):
        raise ValueError("No action in training vocabulary")
    labels = encode(actions, vocab)
    freq = np.bincount(labels, minlength=len(vocab) + 1) + 0.5
    prior = freq / freq.sum()
    model = None
    if mode != "prior":
        x = features(frame, mode, ref)
        model = ExtraTreesClassifier(
            n_estimators=16,
            max_leaf_nodes=192,
            min_samples_leaf=12,
            max_features=min(x.shape[1], 24),
            n_jobs=1,
            random_state=1911,
        )
        model.fit(x, labels)
    return vocab, prior, model


def predict(model_data, frame, mode, ref, h, encoding):
    vocab, prior, model = model_data
    probs = np.tile(prior, (len(frame), 1))
    if model is not None:
        # Smoothing is fixed before fitting. It leaves unseen evaluation outcomes finite.
        probs *= 0.05
        probs[:, model.classes_] += 0.95 * model.predict_proba(
            features(frame, mode, ref)
        )
    target = frame[f"y{h}_line"].to_numpy()
    values = action_codes(frame, h, encoding)
    actual = encode(values, vocab)
    prediction = probs.argmax(axis=1)
    actions = vocab[np.maximum(0, prediction - 1)]
    bank = frame[[f"bank{i}" for i in range(8)]].to_numpy()
    if encoding == "delta":
        proposal = frame.line.to_numpy() + actions
    else:
        page = bank[np.arange(len(bank)), actions // 64]
        proposal = page * 64 + actions % 64
    issued = prediction != 0
    legal = issued & (proposal >= 0) & ((proposal // 64)[:, None] == bank).any(axis=1)
    reachable = (bank == (target // 64)[:, None]).any(axis=1)
    correct = issued & (proposal == target)
    lead = frame[f"y{h}_cycle"].to_numpy() - frame.cycle.to_numpy()
    if np.any(lead < 0):
        raise ValueError("Target predates cue")
    forced_action = vocab[probs[:, 1:].argmax(axis=1)]
    if encoding == "delta":
        forced_line = frame.line.to_numpy() + forced_action
    else:
        forced_page = bank[np.arange(len(bank)), forced_action // 64]
        forced_line = forced_page * 64 + forced_action % 64
    forced_correct = forced_line == target
    forced_hit = forced_correct & reachable
    true_p = probs[np.arange(len(frame)), actual]
    conditional = -np.log2(true_p / np.maximum(1.0 - probs[:, 0], 1e-12))
    return dict(
        loss=-np.log2(true_p),
        conditional=conditional,
        expressible=actual != 0,
        reachable=reachable,
        pred=proposal,
        forced_correct=forced_correct,
        forced_hit=forced_hit,
        forced_deadline=forced_hit & (lead >= 130),
        scored=issued,
        correct=correct,
        legal=legal,
        hit=legal & correct,
        deadline=legal & correct & (lead >= 130),
        lead=lead,
    )


def counters(pred, mask):
    n = int(np.count_nonzero(mask))
    out = dict(anchors=n)
    out["mean_bits"] = float(np.mean(pred["loss"][mask])) if n else ""
    eligible = mask & pred["expressible"]
    out["conditional_mean_bits"] = (
        float(np.mean(pred["conditional"][eligible])) if eligible.any() else ""
    )
    for name in (
        "expressible",
        "reachable",
        "scored",
        "correct",
        "legal",
        "hit",
        "deadline",
        "forced_correct",
        "forced_hit",
        "forced_deadline",
    ):
        out[name] = int(pred[name][mask].sum())
    out["exact_rate"] = out["correct"] / n if n else ""
    return out


def masks_for(case, frame):
    keys = frame[[f"y{h}_key" for h in HORIZONS]].to_numpy()
    out = {}
    for arm in REPLAY:
        name = f"{case}__{arm}"
        candidates = [
            PREVIOUS / "raw/native/runs" / name,
            STUDY / "raw/expansion/runs" / name,
            STUDY / "raw/capture/runs" / name,
        ]
        folder = next(
            (
                p
                for p in candidates
                if (p / "complete.json").is_file()
                and (p / "outcomes_core0.u16le.gz").is_file()
            ),
            None,
        )
        if folder is None:
            continue
        with (folder / "phases.csv").open() as f:
            phases = list(csv.DictReader(f))
        origin = min(int(r["first_ordinal"]) for r in phases)
        end = max(int(r["last_ordinal"]) for r in phases)
        idx = (keys >> 2) - origin
        if idx.min() < 0 or idx.max() > end - origin:
            raise ValueError("Misaligned native masks")
        unique, inverse = np.unique(idx.ravel(), return_inverse=True)
        words = np.empty(len(unique), np.uint16)
        pos = 0
        with gzip.open(folder / "outcomes_core0.u16le.gz", "rb") as f:
            while chunk := f.read(1 << 21):
                data = np.frombuffer(chunk, "<u2")
                lo, hi = unique.searchsorted([pos, pos + len(data)])
                words[lo:hi] = data[unique[lo:hi] - pos]
                pos += len(data)
        if pos != end - origin + 1:
            raise ValueError("Partial native outcome stream")
        out[arm] = (words[inverse].reshape(keys.shape) >> (4 * (keys % 4))) & 15
    return out


def write(name, rows):
    if not rows:
        return
    with name.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(source, destinations, modes, encodings, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    training, total = load(source)
    training = training.loc[splits(training, total)["train"]].copy()
    reference = np.unique(training.pc)
    seen_pages = np.unique(training.line // 64)
    seen_regions = np.unique(training.line // 4096)
    models = {
        (mode, h, enc): train(training, mode, reference, h, enc)
        for mode in modes
        for h in HORIZONS
        for enc in encodings
    }
    summary, strata, blocks, native, examples = [], [], [], [], []
    for dest in destinations:
        whole, count = load(dest)
        for part in ("dev", "test") if dest == source else ("test",):
            eval_frame = whole.loc[splits(whole, count)[part]].copy()
            if eval_frame.empty:
                raise ValueError(f"Empty {part} in {dest}")
            masks = masks_for(dest, eval_frame) if part == "test" else {}
            local_pages = np.isin(eval_frame.line.to_numpy() // 64, seen_pages)
            local_regions = np.isin(eval_frame.line.to_numpy() // 4096, seen_regions)
            known_pc = np.isin(eval_frame.pc, reference)
            for (mode, h, enc), model_data in models.items():
                pred = predict(model_data, eval_frame, mode, reference, h, enc)
                base = dict(
                    source=source,
                    destination=dest,
                    split=part,
                    encoding=enc,
                    horizon=h,
                    scope=mode,
                )
                full = np.ones(len(eval_frame), dtype=bool)
                summary.append({**base, **counters(pred, full)})
                if part != "test":
                    continue
                for group in range(8):
                    idx = eval_frame["index"].to_numpy()
                    group_id = np.minimum(
                        (idx - (4 * count // 5)) * 8 // (count - 4 * count // 5), 7
                    )
                    blocks.append(
                        {**base, "block": group, **counters(pred, group_id == group)}
                    )
                if mode not in (
                    "current",
                    "same_pc",
                    "global32",
                    "no_pc_ids",
                    "with_time",
                    "hide_other_pc",
                ):
                    continue
                seen_line = eval_frame[f"y{h}_seen"].to_numpy().astype(bool)
                seen_page = eval_frame[f"y{h}_seenpage"].to_numpy().astype(bool)
                seen_reg = eval_frame[f"y{h}_seenregion"].to_numpy().astype(bool)
                classes = {
                    "seen_line": seen_line,
                    "new_line_known_page": ~seen_line & seen_page,
                    "new_page_known_region": ~seen_page & seen_reg,
                    "new_region": ~seen_reg,
                    "train_pc": known_pc,
                    "unseen_train_pc": ~known_pc,
                }
                if dest == source:
                    classes.update(
                        train_anchor_page=local_pages,
                        unseen_train_anchor_page=~local_pages,
                        train_anchor_region=local_regions,
                        unseen_train_anchor_region=~local_regions,
                    )
                for label, subset in classes.items():
                    strata.append({**base, "stratum": label, **counters(pred, subset)})
                if (
                    mode in ("same_pc", "global32", "hide_other_pc", "with_time")
                    and "nopf" in masks
                ):
                    k = HORIZONS.index(h)
                    original = masks["nopf"][:, k] == 2
                    for arm, outcomes in masks.items():
                        if arm == "nopf":
                            continue
                        for status, bits in (
                            ("no_tag", 0),
                            ("ordinary", 1),
                            ("persistent", 2),
                            ("late", 6),
                            ("timely", 9),
                        ):
                            subset = original & (outcomes[:, k] == bits)
                            native.append(
                                {
                                    **base,
                                    "arm": arm,
                                    "original_miss_outcome": status,
                                    **counters(pred, subset),
                                }
                            )
                if mode == "with_time" and enc == "delta":
                    # A reproducible first witness. Empty classes have no example.
                    for label, subset in (
                        ("new_line", ~seen_line),
                        ("seen_line", seen_line),
                    ):
                        indices = np.flatnonzero(subset & pred["deadline"])
                        if not len(indices):
                            continue
                        j = int(indices[0])
                        e = eval_frame.iloc[j]
                        examples.append(
                            {
                                **base,
                                "stratum": label,
                                "cue_key": int(e.key),
                                "cue_pc": int(e.pc),
                                "cue_line": int(e.line),
                                "target_key": int(e[f"y{h}_key"]),
                                "predicted_line": int(pred["pred"][j]),
                                "target_line": int(e[f"y{h}_line"]),
                                "lead_cycles": int(pred["lead"][j]),
                            }
                        )
            print(source, dest, part, len(eval_frame), flush=True)
        del whole
    for name, rows in (
        ("scores", summary),
        ("strata", strata),
        ("blocks", blocks),
        ("baseline_outcomes", native),
        ("examples", examples),
    ):
        write(outdir / f"{name}.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", nargs="+", required=True)
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=SCOPES,
        default=tuple(
            x
            for x in SCOPES
            if x
            not in (
                "with_pressure",
                "with_core",
                "with_feedback",
                "with_stride_feedback",
            )
        ),
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=("delta", "page_offset"),
        default=("delta", "page_offset"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.dest, args.scopes, args.encodings, args.out)
