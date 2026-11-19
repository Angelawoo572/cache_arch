"""Which attributed independent targets were also timely under any matched cheap-control replay?"""

from collections import Counter
from replay import CASES
from streams import location, keys, events, label
from characterize import publish, target_kind
from score_replay import accepted
from pattern import snapshots

CONTROLS = ("stride", "multi", "spp", "temporal", "look2", "sparse1")


def scan(case):
    root = location(case, "shape", "diagnostic")
    arms = (*CONTROLS, "look4", "sparse4", "shape", "nopf")
    if any(
        not (location(case, p, "diagnostic") / "complete.json").is_file() for p in arms
    ):
        return [], [], []
    b = events(case, "nopf", "diagnostic")
    z = label(events(case, "shape", "diagnostic"), b)
    ids = keys(z)
    lookup = {k: i for i, k in enumerate(ids)}
    bm = {k: j for j, k in enumerate(keys(b))}
    references = {}
    for p in (*CONTROLS, "look4", "sparse4"):
        zz = events(case, p, "diagnostic")
        references[p] = (zz, dict(zip(keys(zz), range(len(zz["t"])))))
    obs, seen = snapshots(z)
    outcomes = accepted(root)
    summary = Counter()
    groups = Counter()
    examples = {}
    for j, key in enumerate(ids):
        if z["baseline"][j] != 0 or z["kind"][j] != 1:
            continue
        summary["all_original_timely"] += 1
        rec = outcomes[key]
        if rec["source"] != "frozen":
            continue
        summary["direct_independent_original_timely"] += 1
        status = {p: target_kind(v, mp.get(key)) for p, (v, mp) in references.items()}
        for p, k in status.items():
            if k == "timely":
                summary[p + "_also_timely"] += 1
        if any(status[p] == "timely" for p in CONTROLS):
            continue
        summary["no_timely_in_any_separate_control"] += 1
        if not all(status[p] in ("miss", "late") for p in CONTROLS):
            continue
        k = (rec["cue"], rec["op"])
        i = lookup[k]
        source = int(z["pa"][i])
        target = int(z["pa"][j])
        d = target - source
        summary["all_six_persistent_or_late"] += 1
        summary["all_six_and_positive_nopf_head"] += b["head"][bm[key]] > 0
        summary["all_six_and_fresh_line"] += seen[target] > i
        if all(k in ("miss", "late") for k in status.values()):
            summary["all_eight_persistent_or_late"] += 1
            summary["all_eight_and_positive_nopf_head"] += b["head"][bm[key]] > 0
            summary["all_eight_and_fresh_line"] += seen[target] > i
        shape = obs[i][0][2]
        groups[shape, d] += 1
        if (shape, d) not in examples:
            examples[shape, d] = dict(
                case=case,
                shape_key=" ".join(map(str, shape)),
                delta=d,
                cue_instr=k[0],
                cue_op=k[1],
                past_pcs=" ".join(str(int(x)) for x in z["pc"][max(0, i - 3) : i + 1]),
                cue_line=source,
                target_instr=key[0],
                target_op=key[1],
                target_line=target,
                generation=rec["gen"],
                fill=rec["fill"],
                demand=int(z["t"][j]),
                head_nopf=int(b["head"][bm[key]]),
                fresh=int(seen[target] > i),
                **status,
            )
    groups_ = [
        dict(case=case, shape_key=" ".join(map(str, s)), delta=d, targets=n)
        for (s, d), n in groups.most_common()
    ]
    examples_ = [
        dict(examples[key], group_targets=n) for key, n in groups.most_common(4)
    ]
    return [dict(case=case, **summary)], groups_, examples_


if __name__ == "__main__":
    out = [[], [], []]
    for case in CASES:
        for bucket, rows in zip(out, scan(case)):
            bucket.extend(rows)
    for name, rows in zip(
        (
            "diagnostic_joint_coverage",
            "diagnostic_joint_patterns",
            "diagnostic_joint_examples",
        ),
        out,
    ):
        publish(name, rows)
