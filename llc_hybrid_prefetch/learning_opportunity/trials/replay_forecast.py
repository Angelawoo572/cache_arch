"""Replay the first service-qualified MCF forecast identified in the SPP-clock diagnostic."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "one_offer"))
from score import measure
from replay import dump, read, replay


def main():
    case = "mcf17_transfer"
    source = STUDY / "probes/spp_clock_examples.csv"
    examples = [
        r
        for r in read(source)
        if (r["case"], r["scope"], r["policy"], r["claim"])
        == (case, "global32", "one_proposal", "late")
    ]
    if len(examples) != 1:
        raise ValueError("Expected one previously published diagnostic example")
    ex = examples[0]
    anchor, line, pred = (int(ex[k]) for k in ("key", "line", "predicted_line"))
    if line >> 6 != pred >> 6 or int(ex["advance_vs_spp"]) <= 0:
        raise ValueError("Published forecast is outside the intervention interface")
    row = dict(
        scenario="mcf1644_identified_forecast",
        instr_id=anchor >> 2,
        operand=anchor & 3,
        cue_line=line,
        predicted_line=pred,
    )
    dump(
        HERE / "forecast_selection.csv",
        [
            dict(
                **row,
                baseline_request=ex["request_id"],
                baseline_target_key=ex["y8_key"],
                advance_vs_spp=ex["advance_vs_spp"],
            )
        ],
    )
    baseline = STUDY / "raw/capture/runs" / (case + "__spp")
    destination = STUDY / "raw/trials"
    replay(row, baseline, destination)
    effect = measure(row, baseline, destination, case, "spp")
    if effect["mask_nopf"] != 2 or effect["mask_spp"] != 6:
        raise ValueError("Selected dynamic target differs from its diagnosed outcome")
    if effect["actual_eighth_operand_key"] != int(ex["y8_key"]):
        raise ValueError("Baseline eighth successor differs from its diagnosed key")
    dump(HERE / "forecast_effect.csv", [effect])


if __name__ == "__main__":
    main()
