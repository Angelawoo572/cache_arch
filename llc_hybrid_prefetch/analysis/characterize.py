#!/usr/bin/env python3
"""Pair identical dynamic load operands and measure observation-level outcomes."""
import argparse
import bisect
import csv
import gzip
import json
from array import array
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Event = namedtuple("Event", "key pc va pa cycle hit mshr rq pq dram_rq dram_wq")
Candidate = namedtuple("Candidate", "pid cycle base target key cpu")
Scope = namedtuple("Scope", "case cpu trace")
NULL = ""


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def rows(path):
    if path.is_file():
        stream = path.open()
    elif path.with_suffix(".csv.gz").is_file():
        stream = gzip.open(str(path.with_suffix(".csv.gz")), "rt")
    else:
        raise IOError("Missing instrumentation file {}".format(path))
    with stream:
        for row in csv.DictReader(stream):
            yield row


def operand_key(instr, operand):
    if not 0 <= operand < 256 or instr < 0:
        raise ValueError("Invalid instruction/operand ({},{})".format(instr, operand))
    return 256 * instr + operand


def ratio(a, b):
    return round(a / b, 7) if b else NULL


class Core:
    def __init__(self):
        self.ids = array("Q")
        self.pcs = array("Q")
        self.virtual_lines = array("Q")
        self.upstream = bytearray()
        self.lookup_events = []
        self.by_key = {}
        self.cycles = array("Q")
        self.head = Counter()
        self.stages = Counter()
        self.silent = Counter()
        self.candidates = Counter()
        self.trigger_by_id = {}
        self.accepted_ids = set()
        self.terminal_ids = defaultdict(set)
        self.fill_cycle = {}
        self.use_times = []
        self.timely = {}
        self.late = {}
        self.evidence = defaultdict(list)
        self.llc_disposition = {}

    def register(self, key, pc, va):
        if self.ids and key <= self.ids[-1]:
            raise ValueError("CORE/trace IDs out of order or repeated at {}".format(key))
        self.ids.append(key)
        self.pcs.append(pc)
        self.virtual_lines.append(va)
        self.upstream.append(0)

    def locate(self, key):
        p = bisect.bisect_left(self.ids, key)
        if p == len(self.ids) or self.ids[p] != key:
            raise ValueError("No canonical load {}".format(divmod(key, 256)))
        return p

    def status(self, key):
        i = self.by_key.get(key)
        if i is not None:
            return "hit" if self.lookup_events[i].hit else "miss"
        if key in self.llc_disposition:
            return self.llc_disposition[key]
        flags = self.upstream[self.locate(key)]
        if flags & 1:
            return "L1_hit"
        if flags & 2:
            return "L2_hit"
        if flags & 4:
            return "upstream_merge_or_forward"
        return "no_LLC_lookup"


def read_core_events(run, cores):
    for r in rows(run / "loads.csv"):
        cpu, key = int(r["cpu"]), operand_key(int(r["instr_id"]), int(r["operand"]))
        c = cores[cpu]
        level, stage = r["level"], r["stage"]
        c.stages[level, stage] += 1
        if level == "CORE" and stage == "trace":
            c.register(key, int(r["pc"]), int(r["virtual_line"]))
            continue
        if stage == "lookup":
            hit = int(r["hit"])
            c.stages[level, "hit" if hit else "miss"] += 1
            if level == "LLC":
                if key in c.by_key:
                    raise ValueError("Repeated final LLC lookup {}".format(divmod(key, 256)))
                event = Event(key, int(r["pc"]), int(r["virtual_line"]), int(r["physical_line"]),
                              int(r["cycle"]), hit, int(r["mshr"]), int(r["rq"]), int(r["pq"]),
                              int(r.get("dram_rq", 0)), int(r.get("dram_wq", 0)))
                p = c.locate(key)
                if (c.pcs[p], c.virtual_lines[p]) != (event.pc, event.va):
                    raise ValueError("LLC PC/virtual line differs from canonical load {}".format(divmod(key, 256)))
                if c.cycles and event.cycle < c.cycles[-1]:
                    raise ValueError("LLC events out of temporal order")
                c.by_key[key] = len(c.lookup_events)
                c.lookup_events.append(event)
                c.cycles.append(event.cycle)
            elif hit and level in ("L1", "L2"):
                c.upstream[c.locate(key)] |= 1 if level == "L1" else 2
        elif stage in ("merged", "forwarded"):
            if level == "LLC":
                c.llc_disposition[key] = "LLC_" + stage
            elif level in ("L1", "L2"):
                c.upstream[c.locate(key)] |= 4


def load_run(path):
    spec = read_json(path / "run.json")
    cores = {i: Core() for i in range(len(spec["traces"]))}
    read_core_events(path, cores)
    for r in rows(path / "head.csv"):
        if int(r["cycles"]) > 0:
            cores[int(r["cpu"])].head[operand_key(int(r["instr_id"]), int(r["operand"]))] += int(r["cycles"])
    for r in rows(path / "candidates.csv"):
        c = cores[int(r["cpu"])]
        stage, reason = r["stage"], r["reason"]
        c.candidates[stage] += 1
        if stage == "silent":
            c.silent[reason] += 1
            continue
        pid, cycle, target = int(r["id"]), int(r["cycle"]), int(r["target"])
        if stage == "proposed":
            c.trigger_by_id[pid] = Candidate(pid, cycle, int(r["base"]), target,
                                             operand_key(int(r["instr_id"]), int(r["operand"])), int(r["cpu"]))
        if stage == "accepted":
            c.accepted_ids.add(pid)
        if stage == "fill":
            c.fill_cycle[pid] = cycle
        if stage in ("timely", "late", "evicted", "redundant", "unresolved"):
            c.terminal_ids[stage].add(pid)
        if stage in ("timely","late"):
            c.use_times.append((pid, stage, cycle))
        if stage == "rejected":
            c.candidates["rejected_" + reason] += 1
        if stage in ("proposed", "accepted", "rejected", "dispatch", "fill", "timely", "late", "evicted", "redundant", "unresolved"):
            timeline = c.evidence[target]
            timeline.append((cycle, len(timeline), stage, reason, pid))
        if stage in ("timely", "late"):
            affected = int(r["demand_cpu"])
            if affected >= 0 and int(r["demand_operand"]) >= 0:
                k = operand_key(int(r["demand_instr_id"]), int(r["demand_operand"]))
                lifecycle = cores[affected].timely if stage == "timely" else cores[affected].late
                lifecycle[k] = (int(r["cpu"]), pid, cycle)
    for c in cores.values():
        for timeline in c.evidence.values():
            timeline.sort()
    timings = {int(r["cpu"]): (int(r["instructions"]), int(r["cycles"])) for r in rows(path / "cores.csv")}
    stats = {r["key"]: int(r["value"]) for r in rows(path / "stats.csv")}
    return spec, cores, timings, stats


def failure_reason(key, e, last_same, cores, cpu):
    if key in cores[cpu].late:
        return "demand_merged_late"
    # Include predictions from other cores for a physically shared line. Restrict
    # to the interval since its previous LLC demand lookup on the same core.
    start = last_same.get(e.pa, -1)
    best = None
    for c in cores.values():
        timeline = c.evidence.get(e.pa)
        if not timeline:
            continue
        p = bisect.bisect_left(timeline, (e.cycle,))
        if p:
            item = timeline[p - 1]
            if item[0] > start and (best is None or item[0] > best[0]):
                best = item
    if best is None:
        return "no_observed_target_action_since_previous_lookup"
    if best[2] == "rejected":
        return "target_rejected_" + best[3]
    return "last_target_action_" + best[2]


def pressure_row(result, cohort, e, head, baseline_head):
    s = result[cohort]
    s["loads"] += 1
    s["head_cycles"] += head
    s["baseline_head_cycles"] += baseline_head
    s["head_nonzero"] += int(head > 0)
    s["mshr_sum"] += e.mshr
    s["rq_sum"] += e.rq
    s["pq_sum"] += e.pq
    s["dram_rq_sum"] += e.dram_rq
    s["dram_wq_sum"] += e.dram_wq


def neighbor_class(anchor, event):
    delta = event.va - anchor.va
    return (int(event.pc == anchor.pc), int(event.va // 64 == anchor.va // 64),
            int(abs(delta) <= 4), int(event.hit))


def neighborhoods(c, hits, controls):
    # Exact same PC and core, nearest remaining persistent miss in dynamic
    # instruction distance. Pair each control at most once.
    pools = defaultdict(list)
    for i in controls:
        e = c.lookup_events[i]
        pools[e.pc, (e.va % 64) // 16].append((e.key, i))
    for group in pools.values():
        group.sort()
    groups = defaultdict(list)
    for i in hits:
        e = c.lookup_events[i]
        groups[e.pc, (e.va % 64) // 16].append((e.key, i))
    samples = []
    for pc, items in groups.items():
        pool = pools[pc]
        if not pool:
            continue
        # Disjoint-set successor/predecessor indices skip used controls without
        # quadratic list deletes on highly concentrated PCs.
        forward = list(range(len(pool) + 1))
        reverse = list(range(len(pool) + 1))
        def find(parents, i):
            while i != parents[i]:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        for key, i in sorted(items):
            k = bisect.bisect_left(pool, (key, -1))
            right = find(forward, k)
            left = len(pool) - 1 - find(reverse, len(pool) - k)
            choices = [j for j in (left, right) if 0 <= j < len(pool)]
            if not choices:
                break
            j = min(choices, key=lambda j: (abs((pool[j][0] >> 8) - (key >> 8)), pool[j][0]))
            if abs((pool[j][0] >> 8) - (key >> 8)) > 65536:
                continue
            forward[j] = find(forward, j + 1)
            rev = len(pool) - 1 - j
            reverse[rev] = find(reverse, rev + 1)
            samples.append((i, pool[j][1]))
    return samples


def row(s, **values):
    record = dict(case=s.case, cpu=s.cpu, trace=s.trace)
    record.update(values)
    return record


def compare(no_dir, stride_dir):
    ns, nc, nt, nstats = load_run(no_dir)
    ss, sc, st, sstats = load_run(stride_dir)
    for field in ("traces", "workloads", "warmup", "roi", "skip"):
        if ns[field] != ss[field]:
            raise ValueError("Mismatched {}".format(field))
    result = defaultdict(list)
    if ns.get("environment",{}).get("HYBRID_PAGE_SLOTS") != ss.get("environment",{}).get("HYBRID_PAGE_SLOTS"):
        raise ValueError("NoPF and Stride physical page placements differ")
    case = ss["run_id"].rsplit("-", 1)[0]
    for cpu, c in sorted(sc.items()):
        n = nc[cpu]
        if (n.ids, n.pcs, n.virtual_lines) != (c.ids, c.pcs, c.virtual_lines):
            raise ValueError("{} core {} dynamic load ID/PC/virtual-line mismatch".format(case, cpu))
        s = Scope(case, cpu, ss["traces"][cpu])
        transitions, failures = Counter(), Counter()
        pressure = defaultdict(Counter)
        hit_positions, control_positions = [], []
        last_same = {}
        for i, e in enumerate(c.lookup_events):
            status0 = n.status(e.key)
            status1 = "hit" if e.hit else "miss"
            transitions[status0, status1] += 1
            if status0 == "miss":
                pressure_row(pressure, "NoPF_miss_Stride_" + status1, e, c.head[e.key], n.head[e.key])
                if e.hit:
                    hit_positions.append(i)
                    if e.key in c.timely:
                        transitions["matched_prefetch_fill", "hit"] += 1
                else:
                    if e.key in c.late:
                        transitions["matched_prefetch_late", "miss"] += 1
                    control_positions.append(i)
                    failures[failure_reason(e.key, e, last_same, sc, cpu)] += 1
            last_same[e.pa] = e.cycle
        for e in n.lookup_events:
            if e.key not in c.by_key:
                transitions["hit" if e.hit else "miss", c.status(e.key)] += 1
        ni, nn = nt[cpu]
        si, sn = st[cpu]
        if ni != si or ni != ss["roi"]:
            raise ValueError("{} core {} unequal ROI retirement: {} vs {} vs {}".format(case, cpu, ni, si, ss["roi"]))
        nmiss, smiss = n.stages["LLC", "miss"], c.stages["LLC", "miss"]
        nlook, slook = n.stages["LLC", "lookup"], c.stages["LLC", "lookup"]
        proposal, accepted = c.candidates["proposed"], c.candidates["accepted"]
        admitted = c.accepted_ids
        terminal = set().union(*c.terminal_ids.values())
        timely = len(c.terminal_ids["timely"] & admitted)
        late = len(c.terminal_ids["late"] & admitted)
        direct_first = transitions["matched_prefetch_fill","hit"]
        late_first = transitions["matched_prefetch_late","miss"]
        timely_slack = [t - c.fill_cycle[pid] for pid,stage,t in c.use_times
                        if stage == "timely" and pid in admitted and pid in c.fill_cycle]
        late_slack = [c.fill_cycle[pid] - t for pid,stage,t in c.use_times
                      if stage == "late" and pid in admitted and pid in c.fill_cycle and c.fill_cycle[pid] >= t]
        if accepted != len(admitted) or (terminal-admitted) or admitted != terminal:
            raise ValueError("{} core {} lifecycle loss: {} accepted / {} ended / {} unjoined".format(
                case, cpu, accepted, len(terminal), len(admitted^terminal)))
        if sum(len(ids) for ids in c.terminal_ids.values()) != len(terminal):
            raise ValueError("{} core {} multiple terminal outcomes for a prefetch".format(case, cpu))
        if proposal != accepted + c.candidates["rejected"]:
            raise ValueError("{} core {} proposal dispositions do not sum".format(case, cpu))
        if nmiss != sum(v for (a,_),v in transitions.items() if a=="miss"):
            raise ValueError("Incomplete NoPF baseline miss transitions")
        if smiss != sum(v for (a,b),v in transitions.items() if b=="miss" and not a.startswith("matched_prefetch_")):
            raise ValueError("Incomplete Stride LLC miss transitions")
        head_key = "head_unique_cycles_core_"+str(cpu)
        out = row(s, group=ss["group"], ncores=len(sc), instructions=ni, canonical_loads=len(c.ids),
                  nopf_cycles=nn, stride_cycles=sn, speedup=ratio(nn, sn),
                  nopf_ipc=ratio(ni, nn), stride_ipc=ratio(si, sn), nopf_llc_lookups=nlook,
                  stride_llc_lookups=slook, nopf_llc_misses=nmiss, stride_llc_misses=smiss,
                  nopf_llc_miss_rate=ratio(nmiss, nlook), stride_llc_miss_rate=ratio(smiss, slook),
                  nopf_llc_mpki=ratio(1000*nmiss, ni), stride_llc_mpki=ratio(1000*smiss, si),
                  nopf_l2_lookups=n.stages["L2","lookup"], nopf_l2_hits=n.stages["L2","hit"],
                  nopf_l2_misses=n.stages["L2","miss"], stride_l2_lookups=c.stages["L2","lookup"],
                  stride_l2_hits=c.stages["L2","hit"],stride_l2_misses=c.stages["L2","miss"],
                  nopf_llc_hits=n.stages["LLC","hit"],stride_llc_hits=c.stages["LLC","hit"],
                  nopf_l2_miss_rate=ratio(n.stages["L2","miss"],n.stages["L2","lookup"]),
                  stride_l2_miss_rate=ratio(c.stages["L2","miss"],c.stages["L2","lookup"]),
                  nopf_llc_hit_rate=ratio(n.stages["LLC","hit"],nlook), stride_llc_hit_rate=ratio(c.stages["LLC","hit"],slook),
                  nopf_l2_hit_rate=ratio(n.stages["L2","hit"], n.stages["L2","lookup"]),
                  stride_l2_hit_rate=ratio(c.stages["L2","hit"], c.stages["L2","lookup"]),
                  miss_to_hit=len(hit_positions), miss_to_miss=len(control_positions),
                  hit_to_miss=transitions["hit","miss"], miss_to_first_pf_hit=transitions["matched_prefetch_fill","hit"],
                  paired_coverage=ratio(len(hit_positions), nmiss),
                  direct_timely_coverage=ratio(transitions["matched_prefetch_fill","hit"], nmiss),
                  net_mpki_coverage=ratio(nmiss-smiss, nmiss), proposals=proposal, accepted=accepted,
                  timely_first_uses=timely, late_first_uses=late,
                  miss_to_first_late=late_first,direct_eventual_coverage=ratio(direct_first+late_first,nmiss),
                  prefetch_accuracy=ratio(timely+late, accepted),timely_accuracy=ratio(timely,accepted),
                  timely_fraction_among_uses=ratio(timely,timely+late),
                  matched_fill_lead_events=len(timely_slack),mean_fill_to_demand_cycles=ratio(sum(timely_slack),len(timely_slack)),
                  matched_late_fill_events=len(late_slack),mean_demand_to_late_fill_cycles=ratio(sum(late_slack),len(late_slack)),
                  accepted_without_recorded_terminal=len(admitted-terminal),outcomes_from_non_roi_accepts=len(terminal-admitted),
                  observed_degree=ratio(proposal, proposal+sum(c.silent.values())), nopf_rob_head_wait=nstats.get(head_key, NULL),
                  stride_rob_head_wait=sstats.get(head_key, NULL),
                  nopf_sum_operand_head_wait=sum(n.head.values()),stride_sum_operand_head_wait=sum(c.head.values()))
        result["core_performance"].append(out)
        for (a,b),count in sorted(transitions.items()):
            if not a.startswith("matched_prefetch_") and count:
                result["outcomes"].append(row(s,kind="aligned_lookup",before=a,after=b,count=count))
        for cause,count in sorted(failures.items()):
            result["outcomes"].append(row(s,kind="persistent_miss_evidence",before="miss",after=cause,count=count))
        for tag, cc in (("nopf",n),("stride",c)):
            for (lv,stage),count in sorted(cc.stages.items()):
                if lv!="CORE" and stage not in ("hit","miss"):
                    result["outcomes"].append(row(s,kind="demand_stage_"+tag,before=lv,after=stage,count=count))
        for stage,count in sorted(c.candidates.items()):
            result["outcomes"].append(row(s,kind="candidate",before="stride",after=stage,count=count))
        for why,count in sorted(c.silent.items()):
            result["outcomes"].append(row(s,kind="silent",before="stride",after=why,count=count))
        for label, values in sorted(pressure.items()):
            result["pressure"].append(row(s,cohort=label,loads=values["loads"],head_wait_cycles=values["head_cycles"],
                      paired_nopf_head_wait_cycles=values["baseline_head_cycles"],head_wait_nonzero=values["head_nonzero"],mean_mshr=ratio(values["mshr_sum"],values["loads"]),
                      mean_rq=ratio(values["rq_sum"],values["loads"]),mean_pq=ratio(values["pq_sum"],values["loads"]),
                      mean_dram_rq=ratio(values["dram_rq_sum"],values["loads"]),mean_dram_wq=ratio(values["dram_wq_sum"],values["loads"])))
        contextualize(result,s,c,sc,hit_positions,control_positions)
    scope = Scope(case, "all", "+".join(ss["traces"]))
    for policy,stats in (("nopf",nstats),("stride",sstats)):
        for metric,count in sorted(stats.items()):
            if metric.startswith(("dram_","llc_mshr_","pressure_cycles","multi_operand_head_")):
                result["traffic"].append(row(scope,policy=policy,metric=metric,value=count))
    return result


def contextualize(result, s, c, all_cores, hits, controls):
    matched = neighborhoods(c, hits, controls)
    aggregated = defaultdict(Counter)
    for hit_i, miss_i in matched:
        for cohort, i in (("miss_to_hit", hit_i), ("persistent_miss", miss_i)):
            anchor = c.lookup_events[i]
            for offset in (-4, -2, -1, 1, 2, 4):
                j = i + offset
                if not 0 <= j < len(c.lookup_events):
                    continue
                event = c.lookup_events[j]
                if offset < 0 and event.cycle >= anchor.cycle:
                    # Other same-cycle tag reads are conservatively unavailable.
                    continue
                counter = aggregated[cohort, offset]
                counter["n"] += 1
                p, v, near, h = neighbor_class(anchor, event)
                counter["same_pc"] += p
                counter["same_page"] += v
                counter["near"] += near
                counter["hit"] += h
                counter["delta_t"] += event.cycle - anchor.cycle
                counter["distance"] += abs((c.lookup_events[hit_i].key >> 8) - (c.lookup_events[miss_i].key >> 8))
    for (cohort, offset), a in sorted(aggregated.items()):
        count = a["n"]
        result["hit_context"].append(row(s, cohort=cohort, anchor="stride_lookup", offset=offset,
                    use="pre_demand_descriptive" if offset < 0 else "post_demand_label",
                    matched_pairs=len(matched), observations=count, same_pc_fraction=ratio(a["same_pc"],count),
                    same_page_fraction=ratio(a["same_page"],count), within_4_lines_fraction=ratio(a["near"],count),
                    hit_fraction=ratio(a["hit"],count), mean_cycle_offset=ratio(a["delta_t"],count),
                    mean_abs_pair_instruction_gap=ratio(a["distance"],count)))
    # Keep an actual direct hit, a direct late use, and a persistent miss for each scope.
    for cohort, indices, labels in (("direct_miss_to_hit",hits,c.timely),
                                    ("late_persistent_miss",controls,c.late),
                                    ("persistent_miss",controls,None)):
        selected = next((i for i in indices if (c.lookup_events[i].key not in c.late if labels is None
                         else c.lookup_events[i].key in labels)), None)
        if selected is None:
            continue
        anchor = c.lookup_events[selected]
        trigger = None
        if labels is not None:
            source, pid, _ = labels[anchor.key]
            trigger = all_cores[source].trigger_by_id.get(pid)
        def add(event, role, cycle, cpu, key, pc, va, pa, outcome=NULL, ref=NULL, ident=NULL):
            instr, operand = divmod(key, 256) if key is not None else (NULL, NULL)
            delta = cycle-anchor.cycle if cycle != NULL else NULL
            result["examples"].append(row(s, cohort=cohort, anchor_instr=anchor.key>>8,
                 anchor_operand=anchor.key%256, event=event, relative_to=ref, visibility=role,
                 cpu_of_event=cpu, instr=instr, operand=operand, pc=pc, virtual_line=va,
                 physical_line=pa, cycle=cycle, cycles_from_demand=delta, outcome=outcome, candidate_id=ident))
        if trigger is not None:
            tc = all_cores[trigger.cpu]
            end = bisect.bisect_left(tc.cycles, trigger.cycle)
            for prior in tc.lookup_events[max(0,end-3):end]:
                add("prior_lookup","visible_before_trigger",prior.cycle,trigger.cpu,prior.key,prior.pc,
                    prior.va,prior.pa,"hit" if prior.hit else "miss","generation",trigger.pid)
            trigger_pc = tc.pcs[tc.locate(trigger.key)]
            add("proposed","prediction_instant",trigger.cycle,trigger.cpu,trigger.key,trigger_pc,
                NULL,trigger.target,"target","generation",trigger.pid)
            for t,_,stage,why,pid in tc.evidence.get(trigger.target,()):
                if pid == trigger.pid and stage in ("accepted","dispatch","fill","evicted"):
                    add(stage,"outcome",t,trigger.cpu,None,NULL,NULL,trigger.target,why,"generation",pid)
        for i in range(max(selected-3,0),min(selected+4,len(c.lookup_events))):
            e = c.lookup_events[i]
            if i == selected:
                role = "demand_outcome"
            elif trigger is not None and e.cycle < trigger.cycle:
                role = "visible_before_trigger"
            elif e.cycle < anchor.cycle:
                role = "pre_demand_descriptive"
            else:
                role = "post_demand_or_same_cycle"
            add("demand_anchor" if i == selected else "demand_neighbor",role,e.cycle,s.cpu,e.key,e.pc,e.va,e.pa,
                "hit" if e.hit else "miss","demand",trigger.pid if trigger else NULL)


def combine(files, output):
    result = defaultdict(list)
    for file in sorted(files):
        for name, entries in read_json(file).items():
            result[name].extend(entries)
    if not result:
        return
    matched = {(r["case"], r["cpu"]):r["matched_pairs"] for r in result["hit_context"]}
    for r in result["core_performance"]:
        r["context_matched_controls"] = matched.get((r["case"],r["cpu"]),0)
    solos = {(r["trace"], r["instructions"]):r for r in result["core_performance"] if r["ncores"] == 1}
    grouped = defaultdict(list)
    for r in result["core_performance"]:
        grouped[r["case"]].append(r)
    for case, rr in sorted(grouped.items()):
        for policy in ("nopf","stride"):
            slowdowns = []
            solo_ipcs = []
            for r in rr:
                solo = solos.get((r["trace"], r["instructions"])) if r["group"] == "spec" else None
                cycles = r[policy+"_cycles"]
                r[policy+"_simulated_us_4GHz"] = round(cycles/4000,5)
                si = solo[policy+"_cycles"] if solo else 0
                r[policy+"_slowdown_vs_solo"] = ratio(cycles, si)
                if si:
                    slowdowns.append(cycles/si)
                    solo_ipcs.append(si/cycles)
            mix_cycles = max(r[policy+"_cycles"] for r in rr)
            result["mix_performance"].append(dict(case=case,policy=policy,cores=len(rr),total_instructions=sum(r["instructions"] for r in rr),
                completion_cycles=mix_cycles,simulated_us_4GHz=round(mix_cycles/4000,5),aggregate_ipc=ratio(sum(r["instructions"] for r in rr),mix_cycles),
                weighted_speedup=round(sum(solo_ipcs),7) if len(solo_ipcs)==len(rr) else NULL,
                maximum_slowdown=round(max(slowdowns),7) if len(slowdowns)==len(rr) else NULL))
    output.mkdir(parents=True,exist_ok=True)
    for r in result["traffic"]:
        if r["metric"] in ("dram_demand_bytes","dram_prefetch_bytes","dram_write_bytes"):
            total = sum(x["instructions"] for x in grouped[r["case"]])
            r["bytes_per_1000_retired_instructions"] = ratio(1000*r["value"],total)
        else:
            r["bytes_per_1000_retired_instructions"] = NULL
    for name, entries in sorted(result.items()):
        if not entries:
            continue
        if name == "examples":
            entries.sort(key=lambda r:(r["case"],r["cpu"],r["cohort"],r["cycle"]))
        dest = output / (name+".csv")
        # Atomic publication avoids exposing a partially written result to other readers.
        tmp = dest.with_suffix(".tmp")
        with tmp.open("w",newline="") as fh:
            fields = list(entries[0])
            csvwriter = csv.DictWriter(fh,fieldnames=fields)
            csvwriter.writeheader()
            csvwriter.writerows(entries)
        tmp.replace(dest)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--runs", type=Path, default=ROOT/"raw"/"runs")
    args.add_argument("--cache", type=Path, default=ROOT/"raw"/"final_pair_summaries")
    args.add_argument("--output", type=Path, default=ROOT/"results")
    args.add_argument("--case", action="append", help="Analyze only this named case (without the policy).")
    args.add_argument("--refresh", action="store_true",help="Recompute existing pair summaries.")
    args.add_argument("--combine-only", action="store_true")
    args.add_argument("--pair-only", action="store_true",help="Produce pair summaries independently for parallel processing.")
    opts = args.parse_args()
    opts.cache.mkdir(parents=True,exist_ok=True)
    if not opts.combine_only:
        for run in sorted(opts.runs.glob("*-nopf")):
            case = run.name[:-5]
            if opts.case and case not in opts.case:
                continue
            dest = opts.cache / (case+".json")
            stride = opts.runs / (case+"-stride")
            if not all((p/"complete.json").exists() for p in (run,stride)):
                continue
            if dest.exists() and not opts.refresh:
                completed_at = max((p/"complete.json").stat().st_mtime_ns for p in (run,stride))
                if dest.stat().st_mtime_ns >= completed_at:
                    continue
            out = compare(run,stride)
            tmp = dest.with_suffix(".tmp")
            with tmp.open("w") as fh:
                json.dump(out, fh)
            tmp.replace(dest)
            print("Paired "+case, flush=True)
    if not opts.pair_only:
        combine(opts.cache.glob("*.json"), opts.output)


if __name__ == "__main__":
    main()
