"""Measure frozen address forecasts against the actual claiming request in SPP's clock."""

import csv
import gzip
import sys

import numpy as np
import pandas as pd

from extract import HERE, RAW, STUDY, extract
from fit import features, load, splits, train, write

sys.path.insert(0, str(STUDY / "capture"))
sys.path.insert(0, str(STUDY / "labels"))
from access import IndexedStream
from sequence import Sequence

CASES = ("bfs_transfer", "mcf17_transfer", "soplex06_heldout")
SCOPES = ("local_region", "global32")
STAGES = ("proposed", "accepted", "dispatch", "fill")
POLICIES = ("one_proposal", "abstention_allowed")


def subset(keys, values):
    if not len(keys):
        return np.zeros(len(values), bool)
    ix = keys.searchsorted(values)
    return (ix < len(keys)) & (keys[np.minimum(ix, len(keys) - 1)] == values)


def native_claims(folder, target):
    """Join matching NoPF-miss consumer keys to their claiming request ID and lifetime."""
    with (folder / "symbols.csv").open() as f:
        symbols = {int(r["id"]): r["value"] for r in csv.DictReader(f)}
    codes = {v: k for k, v in symbols.items()}
    expected = {
        int(r.y8_key): (int(r.y8_line), int(r.y8_cycle), int(r.spp))
        for r in target.itertuples()
        if r.nopf == 2 and r.spp in (6, 9)
    }
    keys = np.array(sorted(expected), dtype=np.uint64)
    native = IndexedStream(folder, "offer")
    claims = {}
    for block in native.chunks():
        stage = block[:, 5] >> 32
        picks = np.isin(
            stage, [codes.get("late", -1), codes.get("timely", -1)]
        ) & subset(keys, block[:, 6])
        for order, trigger, req, line, cycle, code, consumer in block[picks].tolist():
            status = symbols[code >> 32]
            exp_line, arrival, mask = expected[consumer]
            if (
                exp_line != line
                or cycle < arrival
                or mask != (6 if status == "late" else 9)
            ):
                raise ValueError(
                    "Claim line, timestamp or mask disagrees with the sampled target"
                )
            if consumer in claims or req in (0,):
                raise ValueError("Claim lacks a unique operand and request ID")
            claims[consumer] = dict(
                request_id=req,
                spp_trigger=trigger,
                claim_line=line,
                claim_cycle=cycle,
                claim_order=order,
            )
    by_req = {d["request_id"]: key for key, d in claims.items()}
    if len(by_req) != len(claims):
        raise ValueError("One request claimed multiple sampled operands")
    requests = np.array(sorted(by_req), dtype=np.uint64)
    for block in native.chunks():
        for order, trigger, req, line, cycle, code, _ in block[
            subset(requests, block[:, 2])
        ].tolist():
            stage = symbols[code >> 32]
            if stage not in STAGES:
                continue
            dest = claims[by_req[req]]
            if (
                dest["spp_trigger"] != trigger
                or dest["claim_line"] != line
                or stage in dest
            ):
                raise ValueError("Conflicting request lifecycle")
            dest[stage] = cycle
            dest[stage + "_order"] = order
    for consumer, c in claims.items():
        ordered = [s for s in STAGES if s in c]
        if any(
            c[x] > c[y] or c[x + "_order"] >= c[y + "_order"]
            for x, y in zip(ordered, ordered[1:])
        ):
            raise ValueError("Request stages not chronological")
        if (
            expected[consumer][2] == 9
            and "fill" in c
            and c["fill_order"] >= c["claim_order"]
        ):
            raise ValueError("Timely claim preceded its fill")
    return claims


def frozen_choice(model_data, cues, scope, reference):
    vocab, prior, model = model_data
    p = np.broadcast_to(0.05 * prior, (len(cues), len(prior))).copy()
    p[:, model.classes_] += 0.95 * model.predict_proba(features(cues, scope, reference))
    forced = cues.line.to_numpy() + vocab[p[:, 1:].argmax(axis=1)]
    bank = cues[[f"bank{i}" for i in range(8)]].to_numpy()
    return (
        forced,
        (forced >= 0) & ((forced // 64)[:, None] == bank).any(axis=1),
        p.argmax(axis=1) == 0,
    )


def serial_at(folder, ranks):
    src = IndexedStream(folder, "ingress_state")
    serial = np.empty(len(ranks), np.int64)
    pos = 0
    for block in src.chunks():
        lo, hi = ranks.searchsorted([pos, pos + len(block)])
        serial[lo:hi] = block[ranks[lo:hi] - pos, 0]
        pos += len(block)
    if pos <= ranks[-1] or np.any(np.diff(serial) <= 0):
        raise ValueError("Ingress serial alignment failed")
    return serial


def median(data):
    return float(data.median()) if len(data) else ""


def run(case, models, reference):
    folder = STUDY / "raw/capture/runs" / f"{case}__spp"
    if not (folder / "capture_validation.json").is_file():
        raise ValueError("Verified SPP capture required")
    if not (RAW / f"{case}__spp.npz").exists():
        extract(case, "spp")
    with np.load(RAW / f"{case}__spp.npz") as f:
        n = int(f["total_arrivals"][0])
        frame = pd.DataFrame({k: f[k] for k in f.files if k != "total_arrivals"})
    test = frame.loc[splits(frame, n)["test"]].reset_index(drop=True)
    seq = Sequence(STUDY / "raw/labels" / f"{case}__spp")
    ranks = test["index"].to_numpy()
    for offset, fields in (
        (0, ["key", "pc", "line", "cycle"]),
        (8, ["y8_key", "y8_pc", "y8_line", "y8_cycle"]),
    ):
        if not np.array_equal(
            seq.observations(ranks + offset), test[fields].to_numpy()
        ):
            raise ValueError("Indexed/native SPP cue or target does not match")
    labels = seq.outcomes(ranks + 8)
    test["nopf"], test["spp"] = labels[:, 0], labels[:, 3]
    test["cue_serial"] = serial_at(folder, ranks)
    claims = native_claims(folder, test)
    base = test[
        [
            "index",
            "key",
            "pc",
            "line",
            "cycle",
            "cue_serial",
            "y8_key",
            "y8_pc",
            "y8_line",
            "y8_cycle",
            "nopf",
            "spp",
        ]
    ].copy()
    events = pd.DataFrame.from_dict(claims, orient="index").reindex(base.y8_key)
    for name in (
        "request_id",
        "spp_trigger",
        "proposed",
        "proposed_order",
        "accepted",
        "dispatch",
        "fill",
        "claim_cycle",
    ):
        base[name] = (
            pd.array(events[name], dtype="Int64")
            if name in events
            else pd.array([None] * len(base), dtype="Int64")
        )
    intervening = seq.observations((ranks[:, None] + np.arange(1, 8)).ravel()).reshape(
        len(ranks), 7, 4
    )
    intervenes = intervening[:, :, 2] == base.y8_line.to_numpy()[:, None]
    closest = intervenes.argmax(axis=1)
    base["earlier_demand_to_line"] = intervenes.any(axis=1)
    base["first_future_cycle"] = np.where(
        intervenes.any(axis=1),
        intervening[np.arange(len(ranks)), closest, 3],
        base.y8_cycle,
    ).astype(np.int64)
    base["cue_to_demand"] = base.y8_cycle - base.cycle
    base["advance_vs_spp"] = base.proposed - base.cycle
    base["measured_service"] = base["fill"] - base.proposed
    base["service_margin"] = base.cue_to_demand - base.measured_service
    causal = test.drop(
        columns=[
            k for k in test if k.startswith("y") or k in ("spp", "nopf", "cue_serial")
        ]
    )
    all_rows = []
    cohorts = []
    counts = []
    examples = []
    for scope, model in models.items():
        df = base.copy()
        df["predicted_line"], df["legal"], df["other_wins"] = frozen_choice(
            model, causal, scope, reference
        )
        df["exact"] = df.predicted_line == df.y8_line
        df["scope"] = scope
        all_rows.append(df)
        for policy in POLICIES:
            emits = (
                (~df.other_wins)
                if policy == "abstention_allowed"
                else np.ones(len(df), bool)
            )
            legal = emits & df.legal
            exact = legal & df.exact
            original = df.nopf == 2
            info = dict(case=case, scope=scope, policy=policy)
            unc = original & exact & ~df.spp.isin([6, 9])
            counts.append(
                dict(
                    **info,
                    cues=len(df),
                    other_wins=int(df.other_wins.sum()),
                    abstained=int((~emits).sum()),
                    illegal=int((emits & ~df.legal).sum()),
                    legal_issued=int(legal.sum()),
                    exact_legal=int(exact.sum()),
                    no_nopf_tag=int((exact & (df.nopf == 0)).sum()),
                    nopf_hit=int((exact & (df.nopf == 1)).sum()),
                    originals=int(original.sum()),
                    correct_original_miss=int((exact & original).sum()),
                    correct_original_spp_no_tag=int((unc & (df.spp == 0)).sum()),
                    correct_original_spp_ordinary=int((unc & (df.spp == 1)).sum()),
                    correct_original_spp_persistent=int((unc & (df.spp == 2)).sum()),
                    correct_original_spp_other=int(
                        (unc & ~df.spp.isin([0, 1, 2])).sum()
                    ),
                )
            )
            for kind, code in (("late", 6), ("timely", 9)):
                group = original & (df.spp == code)
                elig = group & exact
                matched = df[elig & df.request_id.notna()]
                clock = matched[matched.proposed.notna()]
                dated = clock[clock.measured_service.notna()]
                before = clock.advance_vs_spp > 0
                enough = dated.service_margin > 0
                early = dated.advance_vs_spp > 0
                same = clock.advance_vs_spp == 0
                row = dict(
                    **info,
                    claim=kind,
                    native_claimed_targets=int(group.sum()),
                    correct_legal=int(elig.sum()),
                    joined=len(matched),
                    missing_claim=int(elig.sum()) - len(matched),
                    missing_proposal=len(matched) - len(clock),
                    forecast_before=int(before.sum()),
                    same_cycle=int(same.sum()),
                    same_cycle_cue_callback_first=int(
                        (same & (clock.cue_serial < clock.proposed_order)).sum()
                    ),
                    forecast_after=int((clock.advance_vs_spp < 0).sum()),
                    missing_fill=len(clock) - len(dated),
                    strictly_enough_service=int(enough.sum()),
                    equal_service_deadline=int((dated.service_margin == 0).sum()),
                    before_and_enough=int((early & enough).sum()),
                    before_enough_first_future=int(
                        (
                            early
                            & (
                                (dated.first_future_cycle - dated.cycle)
                                > dated.measured_service
                            )
                        ).sum()
                    ),
                    before_enough_same_page=int(
                        (
                            early & enough & (dated.line // 64 == dated.y8_line // 64)
                        ).sum()
                    ),
                    before_enough_target_is_cue_line=int(
                        (early & enough & (dated.line == dated.y8_line)).sum()
                    ),
                    median_cue_to_demand=median(clock.cue_to_demand),
                    median_advance_vs_spp=median(clock.advance_vs_spp),
                    median_advance_when_earlier=median(
                        clock.loc[before, "advance_vs_spp"]
                    ),
                    median_proposal_to_fill=median(dated.measured_service),
                    median_service_margin=median(dated.service_margin),
                )
                if row["forecast_before"] + row["same_cycle"] + row[
                    "forecast_after"
                ] != len(clock):
                    raise ValueError("Incomplete timing partition")
                cohorts.append(row)
                qualifying = dated[early & enough]
                if policy == "one_proposal" and len(qualifying):
                    ex = qualifying.iloc[0]
                    examples.append(
                        {
                            **info,
                            "claim": kind,
                            **{
                                k: int(ex[k])
                                for k in (
                                    "index",
                                    "key",
                                    "pc",
                                    "line",
                                    "cycle",
                                    "predicted_line",
                                    "y8_key",
                                    "y8_pc",
                                    "y8_cycle",
                                    "request_id",
                                    "spp_trigger",
                                    "proposed",
                                    "accepted",
                                    "dispatch",
                                    "fill",
                                    "claim_cycle",
                                    "advance_vs_spp",
                                    "service_margin",
                                )
                            },
                        }
                    )
    raw = RAW / "spp_clock"
    raw.mkdir(exist_ok=True)
    with gzip.open(raw / f"{case}.csv.gz", "wt", newline="") as f:
        pd.concat(all_rows, ignore_index=True).to_csv(f, index=False)
    raw_frames = {s: x for s, x in zip(SCOPES, all_rows)}
    pairs = []
    for policy in POLICIES:
        for status, code in (("late", 6), ("timely", 9)):
            useful = {}
            for scope, d in raw_frames.items():
                yes = (
                    d.legal
                    & d.exact
                    & (d.nopf == 2)
                    & (d.spp == code)
                    & (d.advance_vs_spp > 0)
                    & (d.service_margin > 0)
                )
                if policy == "abstention_allowed":
                    yes &= ~d.other_wins
                useful[scope] = yes.fillna(False).to_numpy(dtype=bool)
            a, b = (useful[s] for s in SCOPES)
            gain = base[b & ~a]
            delta = (gain.y8_line - gain.line).value_counts()
            pairs.append(
                dict(
                    case=case,
                    policy=policy,
                    claim=status,
                    both=int((a & b).sum()),
                    global_only=int((b & ~a).sum()),
                    local_only=int((a & ~b).sum()),
                    distinct_global_only_deltas=len(delta),
                    most_common_gain_delta=int(delta.index[0]) if len(delta) else "",
                    most_common_gain_count=int(delta.iloc[0]) if len(delta) else 0,
                    distinct_gain_cue_pcs=int(gain.pc.nunique()),
                    distinct_gain_target_pcs=int(gain.y8_pc.nunique()),
                )
            )
    return counts, cohorts, examples, pairs


def main():
    d, n = load("bfs_discovery")
    training = d.loc[splits(d, n)["train"]]
    ref = np.unique(training.pc)
    models = {s: train(training, s, ref, 8, "delta") for s in SCOPES}
    counts = []
    cohorts = []
    examples = []
    pairs = []
    for case in CASES:
        a, b, c, p = run(case, models, ref)
        pairs.extend(p)
        counts.extend(a)
        cohorts.extend(b)
        examples.extend(c)
        print(case, len(a), len(b), len(c), flush=True)
    for name, rows in (
        ("spp_clock_population", counts),
        ("spp_clock_claims", cohorts),
        ("spp_clock_examples", examples),
        ("spp_clock_pairs", pairs),
    ):
        write(HERE / f"{name}.csv", rows)


if __name__ == "__main__":
    main()
