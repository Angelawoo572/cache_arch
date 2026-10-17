#!/usr/bin/env python3
"""Static scientific figures for complete MAC0 width/template supplemental results."""
import sys
sys.dont_write_bytecode=True
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

COLORS={"native":"#777777","h8_student":"#0072B2","h8_template":"#56B4E9","h16_student":"#D55E00","h16_template":"#E69F00","none":"#111111"}
ARMS=((0,"native","Native"),(8,"student","H8 NN"),(8,"template","H8 table"),(16,"student","H16 NN"),(16,"template","H16 table"))


def dump(path,rows):
    if not rows:
        Path(path).write_text("")
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)


def load_csv(path):
    with Path(path).open(newline="") as stream:return list(csv.DictReader(stream))


def num(row,key):
    value=float(row[key])
    if not math.isfinite(value):raise ValueError("Nonfinite "+key+" in "+row["run_id"])
    return value


def key(row):
    return (int(row["roi"]),row["policy"],int(row.get("hidden_size") or 0) if row["method"]!="native" else 0,row["method"],int(row["budget"]))


def arm_color(h,method):
    return COLORS["native" if method=="native" else "h{}_{}".format(h,method)]


def plot(out,metrics,pairs,reference_analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    figures=[];plot_rows=[]
    lookup={}
    for row in metrics:
        if row["stage"]=="test" and row["policy"] in ("streamer","ampm","spp") and row["mode"]=="full" and row["method"] in ("native","student","template"):
            k=key(row)
            if k in lookup:raise ValueError("Duplicate plot arm "+str(k))
            lookup[k]=row
    def save(fig,name):
        for ext in ("png","svg"):fig.savefig(out/(name+"."+ext),dpi=180,bbox_inches="tight")
        plt.close(fig);figures.append(name)
    # Same full ROI, actual realized traffic, no cross-protocol numerical
    # substitution. Origin and original protocol stay in the plot data.
    for roi in (0,1):
        for cap in (0,16144):
            fig,axes=plt.subplots(3,2,figsize=(13,10))
            for column,policy in enumerate(("ampm","spp")):
                rows=[lookup[(roi,policy,h,m,cap)] for h,m,_ in ARMS]
                labels=[label for _,_,label in ARMS]
                colors=[arm_color(h,m) for h,m,_ in ARMS]
                values=((["IPC / matched NoPF"],[num(r,"ipc_speedup_vs_NoPF") for r in rows]),
                        (["Actual allowed requests / M instructions"],[num(r,"requests_per_million_instructions") for r in rows]),
                        (["Actual completed DRAM bus MiB"],[num(r,"raw__dram_total_bus_bytes")/2**20 for r in rows]))
                for index,(_,heights) in enumerate(values):
                    ax=axes[index,column];ax.bar(range(5),heights,color=colors,width=.7)
                    for x,y in enumerate(heights):ax.annotate("{:.4f}".format(y) if index==0 else "{:,.1f}".format(y),(x,y),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
                    ax.set_ylim(0,max(heights)*1.20);ax.set_xticks(range(5),labels,fontsize=9)
                    ax.set_ylabel(values[index][0][0]);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
                    if index==0:ax.axhline(1,color="#222222",lw=.8,ls="--");ax.set_title(policy.upper(),fontweight="bold")
                for r in rows:
                    plot_rows.append(dict(figure="widths_roi{}_cap{}".format(roi,cap),result_id=r["result_id"],run_id=r["run_id"],protocol=r["protocol"],origin=r["evidence_origin"],policy=policy,roi=roi,hidden_size=r.get("hidden_size"),method=r["method"],budget=cap,ipc=num(r,"ipc"),ipc_over_NoPF=num(r,"ipc_speedup_vs_NoPF"),allowed_requests=int(r["raw__request_gate_allowed"]),requests_per_million=num(r,"requests_per_million_instructions"),DRAM_bus_bytes=int(r["raw__dram_total_bus_bytes"])))
            fig.suptitle("Complete ROI{} · {} · synchronous MAC0".format(roi,"uncapped" if cap==0 else "common ceiling {:,}/M".format(cap)),fontsize=14)
            fig.text(.04,.025,"H8/native references retain archived protocol; H16 and capped H8 tables are new runs using the exact same actual simulator binary.",fontsize=9)
            fig.text(.04,.008,"Table = unconditional validation-selected modal K and per-slot delta/fill. Common ceiling does not equalize actual requests or DRAM.",fontsize=9)
            fig.tight_layout(rect=(0,.05,1,.96));save(fig,"widths_roi{}_cap{}".format(roi,cap))
    # H8 template cap supplement includes Streamer as well.
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for roi in (0,1):
        for column,policy in enumerate(("streamer","ampm","spp")):
            rows=[lookup[(roi,policy,8,m,cap)] for cap in (0,16144) for m in ("student","template")]
            labels=["NN\nuncapped","Table\nuncapped","NN\ncap","Table\ncap"]
            ax=axes[roi,column];heights=[num(r,"ipc_speedup_vs_NoPF") for r in rows]
            ax.bar(range(4),heights,color=[COLORS["h8_"+r["method"]] for r in rows])
            ax.axhline(1,color="#222222",lw=.8,ls="--")
            for x,r in enumerate(rows):
                ax.text(x,heights[x]+max(heights)*.025,"{:.4f}×\nR {:,.0f}\nD {:.1f}".format(heights[x],num(r,"requests_per_million_instructions"),num(r,"raw__dram_total_bus_bytes")/2**20),ha="center",fontsize=8)
            ax.set_ylim(0,max(heights)*1.35);ax.set_xticks(range(4),labels,fontsize=9);ax.set_title("{} · ROI{}".format(policy.upper(),roi));ax.set_ylabel("IPC / NoPF");ax.grid(axis="y",alpha=.2)
    fig.suptitle("H8 unconditional tables under the inherited ceiling · complete ROIs · MAC0",fontsize=14)
    fig.text(.04,.02,"R: actual allowed requests/M instructions. D: actual completed DRAM bus MiB. Full results retain original protocols and provenance.",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.96));save(fig,"h8_template_cap")
    # Cost is counted work / observed MAC0 window, not finite-cost IPC.
    costrows=[]
    for roi in (0,1):
        for policy in ("ampm","spp"):
            for h in (8,16):
                r=lookup[(roi,policy,h,"student",0)]
                costrows.append(dict(result_id=r["result_id"],policy=policy,roi=roi,hidden_size=h,
                    dense_MACs=int(r["inference_macs"]),nonlinear_evaluations=int(r["inference_nonlinears"]),
                    measured_MAC0_cycles=int(r["cycles"]),mac4_aggregate_dense_cycles=math.ceil(int(r["inference_macs"])/4),
                    mac4_aggregate_dense_over_MAC0_window=math.ceil(int(r["inference_macs"])/4)/int(r["cycles"]),
                    FP32_weights_bytes=int(r["model_weight_bytes"]),global_hc_bytes=int(r["recurrent_hc_bytes"]),
                    weights_plus_hc_payload_bytes=int(r["model_weight_bytes"])+int(r["recurrent_hc_bytes"]),
                    scope="aggregate necessary throughput condition only; excludes dependency latency, queue, nonlinear throughput, control and scratch; no finite IPC"))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    labels=["{} H{} R{}".format(r["policy"].upper(),r["hidden_size"],r["roi"]) for r in costrows]
    x=range(len(costrows));heights=[r["mac4_aggregate_dense_over_MAC0_window"] for r in costrows]
    axes[0].bar(x,heights,color=[COLORS["h{}_student".format(r["hidden_size"])] for r in costrows]);axes[0].axhline(1,color="#222222",ls="--",lw=1)
    axes[0].set_ylabel("ceil(counted dense MACs / 4) / MAC0 cycles");axes[0].set_xticks(x,labels,rotation=60,ha="right",fontsize=8)
    for i,v in enumerate(heights):axes[0].text(i,v,"{:.2f}×".format(v),ha="center",va="bottom",fontsize=8)
    unique=[r for r in costrows if r["roi"]==0];x2=range(len(unique))
    axes[1].bar(x2,[r["FP32_weights_bytes"] for r in unique],label="FP32 weights",color="#0072B2")
    axes[1].bar(x2,[r["global_hc_bytes"] for r in unique],bottom=[r["FP32_weights_bytes"] for r in unique],label="Global h/c",color="#E69F00")
    axes[1].set_ylabel("Resident weights + h/c payload bytes");axes[1].set_xticks(x2,["{} H{}".format(r["policy"].upper(),r["hidden_size"]) for r in unique],rotation=30,ha="right");axes[1].legend()
    fig.suptitle("Actual operation counts and model payload · not a finite-cost IPC forecast",fontsize=14)
    fig.text(.04,.015,"Native compute is unmeasured (NA), not zero. Storage excludes scratch/output/controller, adapter/STL/allocator and deployment implementation.",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));save(fig,"width_cost")
    dump(out/"cost_plot_data.csv",costrows)
    # All saved startup coordinates are actual, without first-crossing estimates.
    old_snap=load_csv(reference_analysis/"progress_snapshots.csv");new_snap=load_csv(out/"progress_snapshots.csv")
    curves=defaultdict(list)
    for origin,rows in (("archived_64_reused",old_snap),("new_supplemental_run",new_snap)):
        for r in rows:
            if r["stage"]=="test" and int(r["instructions"])>0:
                curves[(origin,r["run_id"])].append(r)
    startup=[]
    for roi in (0,1):
        base=[r for r in metrics if r["stage"]=="test" and r["roi"]==roi and r["method"]=="none"]
        if len(base)!=1:raise ValueError("Need unique archived NoPF for startup")
        for policy in ("ampm","spp"):
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            for ax,cap in zip(axes,(0,16144)):
                group=[lookup[(roi,policy,h,m,cap)] for h,m,_ in ARMS]+base
                for r in group:
                    h=int(r.get("hidden_size") or 0);method=r["method"]
                    series=sorted(curves[(r["evidence_origin"],r["run_id"])],key=lambda v:int(v["instructions"]))
                    label="NoPF" if method=="none" else "Native" if method=="native" else "H{} {}".format(h,"NN" if method=="student" else "table")
                    ax.plot([int(s["instructions"])/1e6 for s in series],[num(s,"cumulative_ipc") for s in series],label=label,color=COLORS["none"] if method=="none" else arm_color(h,method),ls="--" if method=="template" else "-",lw=1.5)
                    for s in series:startup.append(dict(figure="startup_{}_roi{}".format(policy,roi),result_id=r["result_id"],roi=roi,policy=policy,budget=cap,method=method,hidden_size=h,instructions=int(s["instructions"]),cycles=int(s["cycles"]),cumulative_ipc=num(s,"cumulative_ipc"),completed_updates=int(s["completed_updates"])))
                ax.set_xlabel("Actual retired ROI instructions (M)");ax.set_ylabel("Cumulative measured IPC");ax.set_title("Uncapped" if cap==0 else "Ceiling16,144/M");ax.grid(alpha=.2)
            axes[0].legend(fontsize=8,ncol=3)
            fig.suptitle("{} · ROI{} · complete saved startup curves".format(policy.upper(),roi),fontsize=14)
            fig.text(.04,.012,"Every point uses saved actual coordinates. A first cumulative baseline crossing is not a minimum observation/history requirement.",fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.94));save(fig,"startup_{}_roi{}".format(policy,roi))
    dump(out/"startup_plot_data.csv",startup)
    # Full K, explicit top-delta display plus retained 'other' mass, fill targets,
    # and exact phase profiles. No prefix distributions substitute for a ROI.
    all_hist=load_csv(reference_analysis/"action_histograms.csv")+load_csv(out/"action_histograms.csv")
    hist=defaultdict(Counter)
    for r in all_hist:
        if r["stream"]=="raw":
            hist[(r["run_id"],int(r["phase"]),r["statistic"])][int(r["value"])]+=int(r["count"])
    phase_data=[];distribution_data=[]
    for roi in (0,1):
        for policy in ("ampm","spp"):
            runs=[lookup[(roi,policy,h,"student",0)] for h in (8,16)]
            fig,axes=plt.subplots(3,2,figsize=(13,10))
            phasefig,phaseaxes=plt.subplots(1,3,figsize=(13,4))
            for column,r in enumerate(runs):
                rid=r["run_id"];h=int(r["hidden_size"]);phases=sorted({p for name,p,stat in hist if name==rid})
                totals={stat:Counter() for stat in ("K","delta","fill")}
                for phase in phases:
                    for stat in totals:totals[stat].update(hist[(rid,phase,stat)])
                    kh,eh,fh=hist[(rid,phase,"K")],hist[(rid,phase,"emit")],hist[(rid,phase,"fill")]
                    decisions=sum(kh.values());actions=sum(k*v for k,v in kh.items())
                    pd=dict(result_id=r["result_id"],run_id=rid,policy=policy,roi=roi,hidden_size=h,phase=phase,
                            decisions=decisions,actions=actions,emit_fraction=eh[1]/decisions if decisions else None,
                            mean_K=actions/decisions if decisions else None,L2_action_fraction=fh[2]/actions if actions else None,
                            distinct_delta_count=len(hist[(rid,phase,"delta")]),scope="exact callback-assigned phase histogram")
                    phase_data.append(pd)
                local=[q for q in phase_data if q["result_id"]==r["result_id"]]
                for ax,field,label in zip(phaseaxes,("emit_fraction","mean_K","L2_action_fraction"),("Emit fraction","Mean decoded K","L2 fraction of actions")):
                    points=[q for q in local if q[field] is not None];ax.plot([q["phase"] for q in points],[q[field] for q in points],marker="o",label="H{}".format(h),color=COLORS["h{}_student".format(h)]);ax.set_xlabel("Exact 5M-instruction profile phase");ax.set_ylabel(label);ax.grid(alpha=.2)
                for index,stat in enumerate(("K","delta","fill")):
                    counter=totals[stat];total=sum(counter.values())
                    if stat=="delta" and len(counter)>12:
                        shown=sorted(counter,key=lambda v:(-counter[v],v))[:12];shown=sorted(shown)
                        labels=[str(v) for v in shown]+["other"];counts=[counter[v] for v in shown]+[sum(c for v,c in counter.items() if v not in shown)]
                    else:
                        shown=sorted(counter);labels=[str(v) for v in shown];counts=[counter[v] for v in shown]
                    ax=axes[index,column];fractions=[v/total for v in counts] if total else []
                    ax.bar(range(len(labels)),fractions,color=COLORS["h{}_student".format(h)]);ax.set_xticks(range(len(labels)),labels,rotation=45 if stat=="delta" else 0,fontsize=8)
                    if stat=="K" and fractions:ax.set_yscale("log");ax.set_ylim(max(min(fractions)*.5,1e-8),1.5)
                    ax.set_ylabel("Fraction of "+("demand decisions" if stat=="K" else "decoded actions"));ax.set_title("H{} · {} · N={:,}".format(h,stat,total));ax.grid(axis="y",alpha=.2)
                    for value,count in counter.items():distribution_data.append(dict(result_id=r["result_id"],policy=policy,roi=roi,hidden_size=h,statistic=stat,value=value,count=count,fraction=count/total if total else None,scope="entire actual25M ROI"))
            fig.suptitle("{} ROI{} · uncapped full NN raw outputs · whole ROI".format(policy.upper(),roi),fontsize=14)
            fig.text(.04,.012,"K includes every observed support value (log y). Delta displays12 most frequent signed values plus explicit remaining mass; CSV keeps all values. Fill2=L2,4=LLC.",fontsize=8)
            fig.tight_layout(rect=(0,.045,1,.96));save(fig,"distributions_{}_roi{}".format(policy,roi))
            phaseaxes[0].legend();phasefig.suptitle("{} ROI{} · full NN phase distributions".format(policy.upper(),roi),fontsize=13);phasefig.tight_layout(rect=(0,0,1,.93));save(phasefig,"phases_{}_roi{}".format(policy,roi))
    dump(out/"distribution_plot_data.csv",distribution_data);dump(out/"phase_plot_data.csv",phase_data);dump(out/"comparison_plot_data.csv",plot_rows)
    manifest=dict(status="PASS",new_complete_runs=24,reference_complete_runs=64,figures=figures,
        no_prefix_extrapolation=True,raw_inputs_modified=False,source_reference=str(reference_analysis),
        limits=["Existing seed7 only; H16 is a width intervention, not a second seed.",
                "Full original protocols and origins are retained; only explicitly compatible live controls are joined.",
                "Same configured Sacramento and exact actual simulator binary; historical per-run hardware telemetry absent.",
                "No replay/offline IPC enters live comparison figures.",
                "Measured IPC is MAC0; counted-work MAC4 ratios do not predict finite IPC.",
                "Templates were fit on full own-model validation only; same ceiling is not equal actual traffic.",
                "Template failure is not direct evidence of stateful novelty; negative/mixed outcomes retained.",
                "Full distributions and actual saved startup coordinates retained; no first crossing history claim."])
    (out/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest
