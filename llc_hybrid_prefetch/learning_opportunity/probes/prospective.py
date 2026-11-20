"""Freeze two one-shot probes using only information available at the cue."""

from pathlib import Path

import numpy as np
import pandas as pd

from fit import RAW, features, load, splits, train, write

SOURCE = "soplex06_heldout"
TARGET = "bfs_discovery"


def actions(model_data, cues, scope, reference):
    vocab, prior, model = model_data
    prob = np.broadcast_to(0.05 * prior, (len(cues), len(prior))).copy()
    prob[:, model.classes_] += 0.95 * model.predict_proba(
        features(cues, scope, reference)
    )
    index = prob.argmax(axis=1)
    return index, vocab[np.maximum(index - 1, 0)]


def main():
    donor, n = load(SOURCE)
    training = donor.loc[splits(donor, n)["train"]]
    reference = np.unique(training.pc)
    models = {
        s: train(training, s, reference, 8, "delta") for s in ("global32", "same_pc")
    }
    # The held-out stream contributes only present/past features, including its index.
    with np.load(RAW / f"{TARGET}__stride.npz", allow_pickle=False) as data:
        count = int(data["total_arrivals"][0])
        cues = pd.DataFrame(
            {
                k: data[k]
                for k in data.files
                if not k.startswith("y") and k != "total_arrivals"
            }
        )
    rows = []
    for left, right, require_disagreement in [(80, 85, False), (90, 95, True)]:
        window = cues[
            (cues["index"] >= count * left // 100)
            & (cues["index"] < count * right // 100)
        ]
        rich, delta = actions(models["global32"], window, "global32", reference)
        local, local_delta = actions(models["same_pc"], window, "same_pc", reference)
        line = window.line.to_numpy()
        target = line + delta
        eligible = (rich > 0) & (delta != 0) & (target // 64 == line // 64)
        if require_disagreement:
            eligible &= (local == 0) | (local_delta != delta)
        choices = np.flatnonzero(eligible)
        if choices.size == 0:
            print(f"No eligible cue in preregistered {left}--{right} percent window")
            continue
        j = int(choices[0])
        cue = window.iloc[j]
        key = int(cue.key)
        rows.append(
            dict(
                scenario=f"bfs_h8_at_{left}pct",
                instr_id=key >> 2,
                operand=key & 3,
                cue_line=int(cue.line),
                predicted_line=int(target[j]),
                cue_pc=int(cue.pc),
                cue_ingress_index=int(cue["index"]),
                cue_cycle=int(cue.cycle),
                require_same_pc_disagreement=int(require_disagreement),
                local_prediction=int(line[j] + local_delta[j])
                if local[j]
                else "abstain",
            )
        )
    write(Path(__file__).with_name("prospective_offers.csv"), rows)


if __name__ == "__main__":
    main()
