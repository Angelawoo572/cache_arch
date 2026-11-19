"""Recover what each actual cheap control did to the same native first-use witnesses."""

import argparse
from collections import defaultdict
from pattern import HERE, table
from streams import events, keys, location
from score_replay import accepted
from characterize import publish, target_kind


def make(scope):
    examples = table(HERE / (scope + "_native_examples.csv"))
    chosen = {}
    for r in examples:
        if (
            r["actual_arm"] == "shape"
            and r["prior_arm"] == "spp"
            and r["prior_status"] in ("late", "miss")
        ):
            chosen.setdefault(
                (r["case"], int(r["target_instr"]), int(r["target_op"])), r
            )
    groups = defaultdict(list)
    for (case, i, op), r in chosen.items():
        groups[case].append((i, op, r))
    out = []
    for case, witnesses in groups.items():
        ps = (
            (
                "nopf",
                "stride",
                "multi",
                "spp",
                "temporal",
                "look2",
                "look4",
                "sparse1",
                "sparse4",
                "local",
                "ordered",
                "shape",
            )
            if scope == "diagnostic"
            else ("nopf", "stride", "multi", "spp", "shape")
        )
        if scope == "full":
            ps = (
                *ps,
                *(
                    p
                    for p in ("temporal", "look2")
                    if (location(case, p, scope) / "complete.json").is_file()
                ),
            )
        for p in ps:
            z = events(case, p, scope)
            lookup = dict(zip(keys(z), range(len(z["t"]))))
            uses = accepted(location(case, p, scope))
            for i, op, r in witnesses:
                idx = lookup.get((i, op))
                action = uses.get((i, op), {})
                if idx is not None and int(z["pa"][idx]) != int(r["target_line"]):
                    raise ValueError("Witness mapped to different physical line")
                ingress = int(z["t"][idx]) if idx is not None else None
                out.append(
                    dict(
                        case=case,
                        scope=scope,
                        target_instr=i,
                        target_op=op,
                        target_line=r["target_line"],
                        original_nopf_head=r["nopf_head"],
                        arm=p,
                        status=target_kind(z, idx),
                        ingress=ingress,
                        producer_instr=action.get("cue"),
                        producer_line=action.get("line"),
                        source=action.get("source"),
                        generation=action.get("gen"),
                        dispatch=action.get("dispatch"),
                        fill=action.get("fill"),
                        lead=ingress - action["gen"]
                        if ingress is not None and "gen" in action
                        else None,
                    )
                )
    publish(scope + "_witness_alternatives", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("diagnostic", "full"), required=True)
    make(parser.parse_args().scope)
