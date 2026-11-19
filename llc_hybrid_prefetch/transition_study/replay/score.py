"""Map changed executions onto the archived, fixed Stride episode roster. Requires NumPy."""

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from run import CASES, HERE, POLICIES, RUNS, csvs, original, stats

VIEWS = (
    "core",
    "core_pc",
    "core_page",
    "core_region4",
    "ingress",
    "ingress_pc",
    "ingress_page",
    "ingress_region4",
)
TERMINAL = ("timely", "late", "evicted", "redundant", "unresolved")
# Labels when scoring an original NoPF tag miss: 0 has no independent lookup, 1 stays a miss, 2 is an unclaimed hit, 3 is a late merge, 4 is an attributed hit.
LABELS = ("no_llc_lookup", "miss", "other_hit", "late", "timely")
EPISODES = HERE.parent / "episodes/episode_ids.csv.gz"


def save(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def live(folder, zs):
    hits = [{} for z in zs]
    at = [{} for z in zs]
    tmp = [{} for z in zs]
    txn = {}
    counts = [Counter() for z in zs]
    with gzip.open(folder / "loads.csv.gz", "rb") as f:
        for line in f:
            if b",LLC,lookup," not in line and b",LLC,ingress," not in line:
                continue
            r = line.rstrip().split(b",")
            cpu = int(r[0])
            key = int(r[1]) * 256 + int(r[2])
            if r[7] == b"ingress":
                at[cpu][key] = int(r[8])
                continue
            assert key not in hits[cpu]
            hits[cpu][key] = (int(r[9]), int(r[3]), int(r[4]), int(r[5]), int(r[8]))
    with gzip.open(folder / "candidates.csv.gz", "rb") as f:
        next(f)
        for line in f:
            r = line.rstrip().split(b",")
            stage = r[8].decode()
            cpu = int(r[1])
            counts[cpu][stage] += 1
            if stage == "silent":
                continue
            i = int(r[0])
            t = int(r[7])
            if stage == "proposed":
                counts[cpu]["proposed_" + r[9].decode()] += 1
                txn[i] = dict(
                    anchor=int(r[2]) * 256 + int(r[3]),
                    source=r[9].decode(),
                    proposed=t,
                    target=int(r[6]),
                )
            if i not in txn:
                continue
            q = txn[i]
            if stage in ("accepted", "dispatch", "fill", "timely", "late"):
                q[stage] = t
                counts[cpu][stage + "_" + q["source"]] += 1
            if stage in ("timely", "late"):
                core = int(r[10])
                key = int(r[11]) * 256 + int(r[12])
                assert key not in tmp[core]
                tmp[core][key] = (stage, q)
    for c in counts:
        assert c["accepted"] == sum(c[k] for k in TERMINAL)
    head = [{} for z in zs]
    with gzip.open(folder / "head.csv.gz", "rb") as f:
        next(f)
        for b in f:
            c, i, o, wait = map(int, b.split(b","))
            head[c][i * 256 + o] = wait
    result = []
    for z, lookup, demand, claims, h in zip(zs, hits, at, tmp, head):
        keys = np.fromiter(lookup, dtype="int64")
        ix = np.searchsorted(z["ids"], keys)
        assert np.array_equal(keys, z["ids"][ix])
        outcomes = np.zeros(len(z["ids"]), dtype="int8")
        if len(keys):
            events = np.array(list(lookup.values()), dtype="int64")
            assert (
                np.array_equal(events[:, 1], z["pc"][ix])
                and np.array_equal(events[:, 2], z["va"][ix])
                and np.array_equal(events[:, 3], z["pa"][ix])
            )
            outcomes[ix] = 1 + events[:, 0]
        for stage, code in (("timely", 4), ("late", 3)):
            keys = np.fromiter((k for k, (s, _) in claims.items() if s == stage), dtype="int64")
            ix = np.searchsorted(z["ids"], keys)
            assert np.array_equal(keys, z["ids"][ix])
            if stage == "timely":
                assert (outcomes[ix] == 2).all()
            outcomes[ix] = code
        result.append((outcomes, claims, h, demand))
    return result, counts


def episodes(case):
    groups = defaultdict(list)
    with gzip.open(EPISODES, "rt") as f:
        for r in csv.DictReader(f):
            if r["case"] == case:
                groups[int(r["core"]), r["view"], r["failure_class"]].append(r)
    return groups


def identifier(r, role):
    if not r[role + "_instr"]:
        return None
    return int(r[role + "_instr"]) * 256 + int(r[role + "_op"])


def projection(z, view):
    ix = z["ingress"] if view.startswith("ingress") else np.arange(len(z["ids"]))
    group = (
        z["pc"][ix]
        if view.endswith("_pc")
        else z["pa"][ix] // 64
        if view.endswith("_page")
        else z["pa"][ix] // 256
        if view.endswith("_region4")
        else np.zeros(len(ix), dtype=int)
    )
    perm = np.argsort(group, kind="stable")
    ix = ix[perm]
    group = group[perm]
    pos = np.full(len(z["ids"]), -1, dtype="int32")
    pos[ix] = np.arange(len(ix))
    return ix, group, pos


def summarize(job):
    case, policy = job
    folder = RUNS / (case + "__" + policy)
    assert (folder / "complete.json").exists(), folder
    old = RUNS / (case + "__stride")
    if not (old / "complete.json").is_file():
        old = original(case, "stride")
    cores = csvs(folder / "cores.csv")
    baseline = csvs(old / "cores.csv")
    zdir = HERE.parent / "raw/episodes" / case
    zs = []
    for c in range(len(cores)):
        with np.load(zdir / f"c{c}.npz") as f:
            zs.append(
                {k: f[k] for k in ("ids", "pc", "va", "pa", "base_path", "outcome", "ingress")}
            )
    series, counts = live(folder, zs)
    selected = episodes(case)
    sr = []
    ex = []
    duplicates = []
    perf = []
    st, old_st = stats(folder), stats(old)
    total_bytes = sum(st["dram_" + k + "_bytes"] for k in ("demand", "prefetch", "write"))
    old_bytes = sum(old_st["dram_" + k + "_bytes"] for k in ("demand", "prefetch", "write"))
    for cpu, (z, (observed, claims, head, arrival), count) in enumerate(zip(zs, series, counts)):
        ref = z["outcome"]
        bmiss = z["base_path"] == 7
        keys = z["ids"]
        if policy == "stride":
            assert np.array_equal((observed == 4) & bmiss, ref == 5)
            assert np.array_equal((observed == 3) & bmiss & (z["outcome"] != 10), ref == 6)
        time = int(cores[cpu]["cycles"])
        base_time = int(baseline[cpu]["cycles"])
        n = int(cores[cpu]["instructions"])
        max_time = max(int(c["cycles"]) for c in cores)
        max_base = max(int(c["cycles"]) for c in baseline)
        perf.append(
            dict(
                case=case,
                policy=policy,
                core=cpu,
                instructions=n,
                cycles=time,
                base_cycles=base_time,
                saved_cycles=base_time - time,
                ipc=round(n / time, 8),
                makespan=max_time,
                saved_makespan=max_base - max_time,
                time_saved_pct=round(100 * (max_base - max_time) / max_base, 6),
                accepted=count["accepted"],
                timely=count["timely"],
                late=count["late"],
                original_timely=int((ref == 5).sum()),
                original_timely_preserved=int(((ref == 5) & (observed == 4)).sum()),
                original_timely_lost=int(((ref == 5) & (observed != 4)).sum()),
                all_original_late_now_timely=int(((ref == 6) & (observed == 4)).sum()),
                all_original_persistent_now_timely=int(((ref == 7) & (observed == 4)).sum()),
                auxiliary_proposed=count["proposed_auxiliary"],
                auxiliary_accepted=count["accepted_auxiliary"],
                auxiliary_timely=count["timely_auxiliary"],
                auxiliary_late=count["late_auxiliary"],
                unused_evicted=count["evicted"],
                redundant=count["redundant"],
                unresolved=count["unresolved"],
                distance2_proposals=count["proposed_look2"],
                paced_proposals=count["proposed_paced"],
                turn_proposals=count["proposed_turn"],
                head_unique_cycles=st["head_unique_cycles_core_" + str(cpu)],
                dram_bytes_all_cores=total_bytes,
                dram_bytes_delta_all_cores=total_bytes - old_bytes,
            )
        )
        unique_targets = set()
        for view in VIEWS:
            ix, group, pos = projection(z, view)
            future = np.flatnonzero((observed[ix] == 4) & bmiss[ix])
            for kind in ("L", "P"):
                group_rows = selected[cpu, view, kind]
                # Explicit zeros for views and case/cores without qualifying reference episodes.
                ctr = Counter()
                deltas = []
                examples = set()
                for e in group_rows:
                    onset = identifier(e, "first_miss")
                    last = identifier(e, "last")
                    begin = identifier(e, "start")
                    ret = identifier(e, "recovery")
                    b, last_idx, q = np.searchsorted(keys, [begin, last, onset])
                    bp, lp, qp = map(int, pos[[b, last_idx, q]])
                    assert (
                        keys[q] == onset
                        and lp + 1 == qp
                        and group[bp] == group[qp]
                        and qp - bp == int(e["run_length"])
                    )
                    prefix = ix[bp:qp]
                    assert (ref[prefix] == 5).all() and ref[q] == (6 if kind == "L" else 7)
                    status = LABELS[observed[q]]
                    stable = observed[prefix] == 4
                    ctr["episodes"] += 1
                    unique_targets.add(int(q))
                    ctr["prefix_loads"] += len(prefix)
                    ctr["prefix_timely_preserved"] += int(stable.sum())
                    ctr["entire_prefix_preserved"] += int(stable.all())
                    ctr["last_timely_preserved"] += int(observed[last_idx] == 4)
                    ctr[status] += 1
                    if observed[q] == 4 and last in arrival:
                        offer_time = claims[onset][1]["proposed"]
                        ctr["converted_with_last_ingress_observed"] += 1
                        ctr["converted_offer_before_last_ingress"] += int(
                            offer_time < arrival[last]
                        )
                        ctr["converted_offer_at_last_ingress"] += int(offer_time == arrival[last])
                    ctr["converted_with_last_timely_preserved"] += int(
                        observed[q] == 4 and observed[last_idx] == 4
                    )
                    ctr["converted_with_entire_prefix_preserved"] += int(
                        observed[q] == 4 and stable.all()
                    )
                    ctr["first_failure_head_sum"] += head.get(onset, 0)
                    if ret is not None:
                        j = int(np.searchsorted(keys, ret))
                        assert keys[j] == ret and ref[j] == 5
                        ctr["baseline_recoveries"] += 1
                        ctr["recovery_id_still_timely"] += int(observed[j] == 4)
                    fi = int(np.searchsorted(future, qp + 1))
                    end = int(future[fi]) if fi < len(future) else -1
                    if end >= 0 and group[end] == group[qp]:
                        gap = end - qp
                        deltas.append(gap)
                        ctr["new_recovery_known"] += 1
                        ctr["new_recovery_within_4"] += int(gap <= 4)
                    # Examples cover a fixed ordinal and first occurrence of each outcome type.
                    if view == "ingress" and status not in examples:
                        examples.add(status)
                        stage, tx = claims.get(onset, ("", {}))
                        ex.append(
                            dict(
                                case=case,
                                policy=policy,
                                cpu=cpu,
                                reference_failure=kind,
                                first_instr=begin // 256,
                                last_timely_instr=last // 256,
                                last_operand=last % 256,
                                target_instr=onset // 256,
                                target_operand=onset % 256,
                                physical_line=z["pa"][q],
                                pc=z["pc"][q],
                                run_length=len(prefix),
                                prior_tail_preserved=int(observed[last_idx] == 4),
                                prefix_timely=int(stable.sum()),
                                current=status,
                                source=tx.get("source", ""),
                                policy_trigger_instr=tx.get("anchor", -256) // 256 if tx else "",
                                policy_trigger_operand=tx.get("anchor", 0) % 256 if tx else "",
                                proposed=tx.get("proposed", ""),
                                accepted=tx.get("accepted", ""),
                                dispatch=tx.get("dispatch", ""),
                                fill=tx.get("fill", ""),
                                demand_ingress=arrival.get(onset, ""),
                                target_use_cycle=tx.get(stage, ""),
                                baseline_head=e["target_head"],
                                policy_head=head.get(onset, 0),
                                reference_recovery=ret // 256 if ret else "",
                            )
                        )
                sr.append(
                    dict(
                        case=case,
                        policy=policy,
                        core=cpu,
                        projection=view,
                        reference_failure=kind,
                        **{
                            k: ctr[k]
                            for k in (
                                "episodes",
                                "prefix_loads",
                                "prefix_timely_preserved",
                                "entire_prefix_preserved",
                                "last_timely_preserved",
                                *LABELS,
                                "converted_with_last_ingress_observed",
                                "converted_offer_before_last_ingress",
                                "converted_offer_at_last_ingress",
                                "converted_with_last_timely_preserved",
                                "converted_with_entire_prefix_preserved",
                                "first_failure_head_sum",
                                "baseline_recoveries",
                                "recovery_id_still_timely",
                                "new_recovery_known",
                                "new_recovery_within_4",
                            )
                        },
                        new_recovery_median_gap=round(float(np.median(deltas)), 2)
                        if deltas
                        else "",
                    )
                )
        ix = np.array(sorted(unique_targets), dtype="int64")
        duplicates.append(
            dict(
                case=case,
                policy=policy,
                core=cpu,
                unique_original_first_failures=len(ix),
                **{label: int((observed[ix] == i).sum()) for i, label in enumerate(LABELS)},
            )
        )
    return sr, ex, duplicates, perf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    arg = ap.parse_args()
    with ProcessPoolExecutor(arg.jobs) as pool:
        out = list(pool.map(summarize, [(c, p) for c in CASES for p in POLICIES]))
    for j, name in enumerate(("cohorts", "examples", "unique_targets", "performance")):
        save(HERE / (name + ".csv"), [r for sheets in out for r in sheets[j]])
