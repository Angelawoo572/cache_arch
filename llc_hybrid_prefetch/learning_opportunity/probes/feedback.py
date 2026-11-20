"""Recover strictly earlier LLC lookup and Stride callback history at sampled cues."""

import argparse
import csv
import gzip

import numpy as np

from extract import RAW, STUDY

LAGS = np.array([1, 2, 4, 8])
STAGES = ("proposed", "silent", "rejected", "accepted", "timely", "late", "evicted")
REASONS = (
    "cold",
    "ready",
    "delta_changed",
    "zero_delta",
    "page",
    "resident",
    "mshr",
    "pq",
    "rq",
    "unused",
)


def scan(path, width, decode, cue_sequences, columns):
    """Use capture serials to disambiguate callbacks in one simulated cycle."""
    result = np.full((len(cue_sequences), len(LAGS), columns), -1, np.int64)
    past = np.empty((0, columns + 1), np.int64)
    covered = 0
    with gzip.open(path, "rb") as f:
        while chunk := f.read(65536 * width * 8):
            data = np.frombuffer(chunk, "<u8").reshape(-1, width)
            events = np.concatenate((past, decode(data)))
            end = cue_sequences.searchsorted(data[-1, 0], side="right")
            ids = (
                events[:, 0].searchsorted(cue_sequences[covered:end], side="left")[
                    :, None
                ]
                - LAGS
            )
            if len(events):
                result[covered:end] = np.where(
                    (ids >= 0)[:, :, None], events[np.maximum(ids, 0), 1:], -1
                )
            past = events[-max(LAGS) :]
            covered = end
    if covered < len(cue_sequences) and len(past):
        ids = (
            past[:, 0].searchsorted(cue_sequences[covered:], side="left")[:, None]
            - LAGS
        )
        result[covered:] = np.where(
            (ids >= 0)[:, :, None], past[np.maximum(ids, 0), 1:], -1
        )
    return result


def prepare(case):
    folder = STUDY / "raw/capture/runs" / f"{case}__stride"
    with np.load(RAW / f"{case}__stride.npz") as sample:
        indices, keys, cycles = sample["index"], sample["key"], sample["cycle"]
        expected = np.column_stack([keys, sample["pc"], sample["line"], cycles])
        count = int(sample["total_arrivals"][0])
    seq = np.empty(len(keys), np.int64)
    pos = 0
    with (
        gzip.open(folder / "ingress_state.u64le.gz", "rb") as f,
        gzip.open(folder / "arrival.u64le.gz", "rb") as af,
    ):
        while chunk := f.read(16 * 65536):
            ingress = np.frombuffer(chunk, "<u8").reshape(-1, 2)
            lo, hi = indices.searchsorted([pos, pos + len(ingress)])
            offsets = indices[lo:hi] - pos
            observed = np.frombuffer(af.read(32 * len(ingress)), "<u8").reshape(-1, 4)
            if not np.array_equal(observed[offsets], expected[lo:hi]):
                raise ValueError("Independent rich and source Stride arrivals differ")
            seq[lo:hi] = ingress[offsets, 0]
            pos += len(ingress)
        if af.read(1) or pos != count:
            raise ValueError(
                "Rich capture differs in length from sampled arrival stream"
            )
    if indices[-1] >= pos or (np.diff(seq) <= 0).any():
        raise ValueError("Sampling index does not align to capture")
    with (folder / "symbols.csv").open() as f:
        code = {r["value"]: int(r["id"]) for r in csv.DictReader(f)}

    def tags(data):
        raw = data[:, 4]
        hit = (raw >> 8) & 255
        keep = ((raw & 255) == 3) & (hit != 0)
        return np.column_stack((data[keep, 0], hit[keep] - 1, data[keep, 2])).astype(
            np.int64
        )

    def offers(data):
        stage, reason = data[:, 5] >> 32, data[:, 5] & 0xFFFFFFFF
        normalized = np.zeros(len(data), np.int64)
        why = np.zeros(len(data), np.int64)
        for i, name in enumerate(STAGES, start=1):
            if name in code:
                normalized[stage == code[name]] = i
        for i, name in enumerate(REASONS, start=1):
            if name in code:
                why[reason == code[name]] = i
        keep = normalized != 0
        return np.column_stack(
            (data[keep, 0], normalized[keep], why[keep], data[keep, 4])
        ).astype(np.int64)

    states = {}
    for stream, width, decoder, names in (
        ("cache", 5, tags, ("tag_hit", "age")),
        ("offer", 7, offers, ("conventional_stage", "conventional_reason", "age")),
    ):
        arr = scan(folder / f"{stream}.u64le.gz", width, decoder, seq, len(names))
        for j, lag in enumerate(LAGS):
            for k, name in enumerate(names):
                field = arr[:, j, k]
                if name == "age":
                    field = np.where(
                        field >= 0, np.log2(1 + np.maximum(0, cycles - field)), -1
                    )
                states[f"prior_{stream}{lag}_{name}"] = field
    np.savez_compressed(RAW / f"{case}__feedback.npz", key=keys, **states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+")
    for case in parser.parse_args().cases:
        prepare(case)
