"""Audit learned cue observability, actual training/test lead, and shared visibility controls."""

import csv, gzip, statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from cases import ROOT, RAW, TRACES, specs

OUT = ROOT / "workloads"


def table(path):
    with gzip.open(path, "rt") as f:
        yield from csv.DictReader(f)


def cue_trial(spec):
    case = spec["case"]
    path = RAW / "runs" / (case + "-nopf")
    trial = {}
    for lab in table(TRACES / (case + ".labels.csv.gz")):
        if lab["measurement"] == "roi" and lab["label"] in ("cue", "choice_target"):
            trial[int(lab["instr_id"])] = (int(lab["group_index"]), lab["label"])
    obs = defaultdict(dict)
    times = defaultdict(dict)
    for r in table(path / "loads.csv.gz"):
        if r["level"] != "LLC":
            continue
        iid = int(r["instr_id"])
        if iid not in trial:
            continue
        gid, role = trial[iid]
        if r["stage"] == "ingress":
            obs[gid][role] = (iid, int(r["pc"]), int(r["physical_line"]), int(r["cycle"]))
        if role == "choice_target" and r["stage"] in ("dispatch", "fill"):
            times[gid][r["stage"]] = int(r["cycle"])
    mid = spec["warmup"] + spec["roi"] // 2
    rows = [
        (hits["cue"], hits["choice_target"], times[k])
        for k, hits in sorted(obs.items())
        if "cue" in hits and "choice_target" in hits
    ]
    train = [(c, t, tm) for c, t, tm in rows if t[0] <= mid]
    test = [(c, t, tm) for c, t, tm in rows if c[0] > mid]
    per_pc = defaultdict(Counter)
    for cue, tgt, timing in train:
        per_pc[cue[1]][tgt[2] - cue[2]] += 1
    predict = {pc: min(v, key=lambda x: (-v[x], x)) for pc, v in per_pc.items()}
    services = sorted(
        tm["fill"] - tm["dispatch"] for _, _, tm in train if "dispatch" in tm and "fill" in tm
    )
    p90 = services[(9 * len(services) + 9) // 10 - 1]
    gaps = [tgt[3] - cue[3] for cue, tgt, _ in test]
    correct = [(cue, tgt) for cue, tgt, _ in test if tgt[2] - cue[2] == predict.get(cue[1])]
    return dict(
        case=case,
        fit_observed_cue_target=len(train),
        test_source_targets=sum(
            1 for i, (_, role) in trial.items() if i > mid and role == "choice_target"
        ),
        test_observed_cue_target=len(test),
        highest_frequency_delta_correct=len(correct),
        same_4k_observed=sum(c[2] // 64 == t[2] // 64 for c, t, _ in test),
        earlier_half_service_p90=p90,
        heldout_median_lead=statistics.median(gaps),
        correct_with_lead_at_least_p90_plus_one=sum(t[3] - c[3] >= p90 + 1 for c, t in correct),
    )


def two_cues(spec):
    name = spec["case"]
    origin = RAW / "runs" / (name + "-nopf")
    meanings = {
        int(r["instr_id"]): (int(r["group_index"]), r["label"])
        for r in table(TRACES / (name + ".labels.csv.gz"))
        if r["measurement"] == "roi" and r["label"] in ("first_cue", "second_cue", "joint_target")
    }
    trials = defaultdict(dict)
    for r in table(origin / "loads.csv.gz"):
        if r["level"] != "LLC" or r["stage"] != "ingress":
            continue
        ix = int(r["instr_id"])
        if ix in meanings:
            trial, role = meanings[ix]
            trials[trial][role] = (ix, int(r["pc"]), int(r["physical_line"]), int(r["cycle"]))
    window = spec["warmup"] + spec["roi"] // 2
    complete = [
        (x["first_cue"], x["second_cue"], x["joint_target"]) for x in trials.values() if len(x) == 3
    ]
    train = [r for r in complete if r[2][0] <= window]
    test = [r for r in complete if r[0][0] > window]
    answers = []
    for view in ("no_cue", "first_only", "second_only", "both"):

        def key(a, b):
            return (
                a[1] if view in ("first_only", "both") else 0,
                b[1] if view in ("second_only", "both") else 0,
            )

        seen = defaultdict(Counter)
        for a, b, target in train:
            seen[key(a, b)][target[2] - b[2]] += 1
        pick = {k: min(freq, key=lambda delta: (-freq[delta], delta)) for k, freq in seen.items()}
        correct = sum(pick.get(key(a, b)) == t[2] - b[2] for a, b, t in test)
        answers.append(
            dict(
                case=name,
                visible_cues=view,
                fit_trials=len(train),
                heldout_source_targets=sum(
                    1 for i, (_, role) in meanings.items() if role == "joint_target" and i > window
                ),
                heldout_all_three_at_llc=len(test),
                heldout_correct=correct,
                learned_contexts=len(pick),
                median_lead_after_last_cue=statistics.median(t[3] - b[3] for a, b, t in test),
            )
        )
    return answers


def pages_at_boundaries(case):
    keys = {
        int(r["instr_id"]): (r["label"], int(r["cross4k"]))
        for r in table(TRACES / (case + ".labels.csv.gz"))
        if r["measurement"] == "roi"
        and r["label"]
        in ("slab_start", "band_start", "subrow_start", "pair_start", "pair_second_row")
    }
    results = defaultdict(Counter)
    for k in keys.values():
        results[k]["source_loads"] += 1
    for policy in ("nopf", "stride"):
        folder = RAW / "runs" / (case + "-" + policy)
        for r in table(folder / "loads.csv.gz"):
            if r["stage"] == "lookup" and r["level"] == "LLC" and int(r["instr_id"]) in keys:
                obs = results[keys[int(r["instr_id"])]]
                obs[policy + "_misses"] += r["hit"] == "0"
                obs[policy + "_hits"] += r["hit"] == "1"
    return [
        dict(
            case=case,
            stage=stage,
            cross4k=cross,
            **{
                k: count[k]
                for k in (
                    "source_loads",
                    "nopf_misses",
                    "nopf_hits",
                    "stride_misses",
                    "stride_hits",
                )
            },
        )
        for (stage, cross), count in sorted(results.items())
    ]


def profiles(spec):
    seen = defaultdict(Counter)
    deltas = defaultdict(Counter)
    prior = {}
    registers = {}
    for r in table(RAW / "runs" / (spec["case"] + "-nopf") / "loads.csv.gz"):
        if r["stage"] != "ingress" or r["level"] != "LLC":
            continue
        cpu, pc, pl = map(int, (r["cpu"], r["pc"], r["physical_line"]))
        seen[cpu][pc] += 1
        key = (cpu, pc)
        if key in prior:
            deltas[cpu][pl - prior[key]] += 1
        prior[key] = pl
    ans = []
    for cpu, pcs in sorted(seen.items()):
        ds = deltas[cpu]
        dominant, count = ds.most_common(1)[0] if ds else (0, 0)
        ans.append(
            dict(
                case=spec["case"],
                cpu=cpu,
                source_events=sum(pcs.values()),
                distinct_trigger_pcs=len(pcs),
                top_pc_count=pcs.most_common(1)[0][1],
                top_four_pc_count=sum(n for _, n in pcs.most_common(4)),
                within_pc_deltas=sum(ds.values()),
                rank1_delta=dominant,
                rank1_delta_count=count,
                delta_plus1_count=ds[1],
                delta_minus1_count=ds[-1],
            )
        )
    return ans


def write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def maps(case, pol):
    return {
        (int(r["cpu"]), int(r["virtual_page"])): (int(r["page_slot"]), int(r["physical_page"]))
        for r in table(RAW / "runs" / (case + "-" + pol) / "pages.csv.gz")
    }


if __name__ == "__main__":
    cs = list(specs())
    write("cue_visibility.csv", [cue_trial(s) for s in cs if s["case"].startswith("crosspc_")])
    write("joint_cues.csv", [r for s in cs if s["case"].startswith("xor_") for r in two_cues(s)])
    write(
        "boundary_pages.csv",
        [
            r
            for n in ("row_hierarchy_paced", "page_same_paced", "page_cross_paced")
            for r in pages_at_boundaries(n)
        ],
    )
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(profiles, [s for s in cs if s["group"] == "real"]))
    write("real_ingress.csv", [r for group in res for r in group])
    m = {}
    for s in cs:
        case = s["case"]
        m[case] = maps(case, "nopf")
        if m[case] != maps(case, "stride"):
            raise ValueError("policy placement differs: " + case)
    for x, y in (
        ("streams_onepc_paced", "streams_pc_paced"),
        ("crosspc_cue_paced", "crosspc_mask_paced"),
        ("ramp_mlp1", "ramp_mlp4"),
        ("streams_three_switch_paced", "streams_three_identified_paced"),
        ("xor_context_paced", "xor_maskfirst_paced"),
        ("row_period2_paced", "row_period2_tight"),
    ):
        if m[x] != m[y]:
            raise ValueError("control placement differs: " + x)
    mix = m["gap_bfs8_lbm"]
    for core, name in [(0, "gap_bfs8"), (1, "lbm_peer")]:
        placed = {vpn: p for (c, vpn), p in mix.items() if c == core}
        solo = {vpn: p for (c, vpn), p in m[name].items()}
        if placed != solo:
            raise ValueError("co-run placement differs: " + name)
    print(
        f"{len(cs)} paired page maps, six matched counter-controls and both mixed-program mappings verified"
    )
