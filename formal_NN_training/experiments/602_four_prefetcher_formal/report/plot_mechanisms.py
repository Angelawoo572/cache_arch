#!/usr/bin/env python3
"""Plot only complete, audited full-ROI global-student mechanism interventions."""
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import json
import math
import os
from pathlib import Path

PROTOCOL="four602_frozen_live_20261017_v2"
POLICIES=("streamer","ampm","spp")
CONTROLS=(("student","full","Full student","#0072B2"),
          ("student","fixed_payload","Fixed payload","#D55E00"),
          ("template","full","Unconditional template","#009E73"),
          ("student","reset_state","Reset before each callback","#CC79A7"))


def read(path):
    return json.loads(Path(path).read_text())


def number(row,key):
    value=float(row[key])
    if not math.isfinite(value):raise ValueError("Nonfinite metric "+key+" in "+row["run_id"])
    return value


def select(analysis):
    audit=read(analysis/"audit.json")
    required=(audit.get("status")=="PASS" and audit.get("complete_count")==64 and
              audit.get("expected_count")==64 and audit.get("failure_count")==0 and
              audit.get("callback_logs_checked") is True and audit.get("protocol")==PROTOCOL and
              set(audit.get("stages",[]))=={"test","validation"})
    if not required:raise ValueError("Require complete64/PASS v2 analysis with full callback-log verification; no partial/prefix figures")
    with (analysis/"per_run.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=64 or sum(r["stage"]=="test" for r in rows)!=56 or sum(r["stage"]=="validation" for r in rows)!=8:
        raise ValueError("Actual CSV must contain56 complete test +8 validation rows")
    if any(r["protocol"]!=PROTOCOL or r["status"]!="complete" for r in rows):
        raise ValueError("Mixed protocols or incomplete rows")
    selected=[r for r in rows if r["stage"]=="test" and r["policy"] in POLICIES and
              int(r["budget"])==0 and (r["method"],r["mode"]) in {(a,b) for a,b,_,_ in CONTROLS}]
    by_key={}
    for row in selected:
        key=(int(row["roi"]),row["seed"],row["policy"],row["method"],row["mode"])
        if key in by_key:raise ValueError("Duplicate mechanism arm "+str(key))
        if abs(int(row["instructions"])-25000000)>4 or not row.get("no_pf_run"):
            raise ValueError("Require actual full25M ROI and matched NoPF: "+row["run_id"])
        by_key[key]=row
    rois=sorted({int(r["roi"]) for r in selected});seeds=sorted({r["seed"] for r in selected})
    if rois!=[0,1] or not seeds:raise ValueError("Require both complete measured ROIs")
    groups=[]
    for roi in rois:
        for seed in seeds:
            panels=[]
            for policy in POLICIES:
                group=[]
                for method,mode,label,color in CONTROLS:
                    key=(roi,seed,policy,method,mode)
                    if key not in by_key:raise ValueError("Missing actual intervention "+str(key))
                    row=by_key[key]
                    group.append(dict(run_id=row["run_id"],roi=roi,seed=seed,policy=policy,method=method,mode=mode,label=label,color=color,
                        instructions=int(row["instructions"]),cycles=int(row["cycles"]),ipc=number(row,"ipc"),
                        ipc_over_NoPF=number(row,"ipc_speedup_vs_NoPF"),no_pf_run=row["no_pf_run"],
                        actual_requests_per_million=number(row,"requests_per_million_instructions"),
                        actual_allowed_requests=int(row["raw__request_gate_allowed"]),
                        actual_completed_DRAM_bus_bytes=int(row["raw__dram_total_bus_bytes"])))
                if len({r["no_pf_run"] for r in group})!=1:raise ValueError("Controls have different NoPF comparators")
                full=group[0]
                for item in group:
                    item["delta_ipc_vs_full"]=item["ipc"]-full["ipc"]
                    item["delta_allowed_vs_full"]=item["actual_allowed_requests"]-full["actual_allowed_requests"]
                    item["delta_DRAM_bytes_vs_full"]=item["actual_completed_DRAM_bus_bytes"]-full["actual_completed_DRAM_bus_bytes"]
                panels.append((policy,group))
            groups.append((roi,seed,panels))
    return audit,groups


def configure(out):
    cache=next((p for p in out.resolve().parents if p.name=="cache"),None)
    if cache is None or not out.resolve().is_relative_to((cache/"formal_study_20261017").resolve()):
        raise ValueError("New figures must stay under cache/formal_study_20261017")
    for key,name in (("TMPDIR","tmp"),("PYTHONPYCACHEPREFIX","pycache"),("MPLCONFIGDIR","mpl")):
        path=cache/"formal_study_20261017/runtime_v2"/name
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True,help="Final v2 analysis directory with audit.json and per_run.csv")
    parser.add_argument("--out",type=Path,required=True,help="Fresh output directory")
    args=parser.parse_args()
    audit,groups=select(args.analysis.resolve())
    out=args.out.resolve();configure(out)
    if out.exists():raise ValueError("Preserve prior figures; use a fresh --out")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True)
    data=[];figures=[]
    for roi,seed,panels in groups:
        fig,axes=plt.subplots(3,1,figsize=(12,13))
        for ax,(policy,group) in zip(axes,panels):
            heights=[g["ipc_over_NoPF"] for g in group]
            top=max(1,max(heights))
            ax.bar(range(4),heights,color=[g["color"] for g in group],width=.67)
            ax.axhline(1,color="#243248",lw=1,linestyle="--")
            for index,item in enumerate(group):
                annotation="{:.4f}×\nR {:,.0f}/M\nD {:.1f} MiB\nΔIPC {:+.4f}".format(
                    item["ipc_over_NoPF"],item["actual_requests_per_million"],
                    item["actual_completed_DRAM_bus_bytes"]/2**20,item["delta_ipc_vs_full"])
                ax.text(index,item["ipc_over_NoPF"]+top*.025,annotation,ha="center",va="bottom",fontsize=9)
                data.append({k:v for k,v in item.items() if k!="color"})
            ax.set_ylim(0,top*1.40)
            ax.set_xticks(range(4),[g["label"] for g in group],fontsize=10)
            ax.set_ylabel("IPC / matched NoPF")
            ax.set_title("{} · seed{} · complete ROI{}".format(policy.capitalize(),seed,roi),loc="left",fontweight="bold")
            ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        fig.suptitle("Actual uncapped mechanism interventions · MAC0 synchronous inference",fontsize=15,y=.995)
        fig.text(.045,.044,"R: actual allowed requests per million retired instructions. D: actual completed DRAM bus MiB. ΔIPC: candidate − full student.",fontsize=10)
        fig.text(.045,.024,"Fixed payload preserves the full forward/K/hc at each given callback; cache feedback can change future callbacks and the whole state trajectory.",fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.97))
        name="mechanisms_roi{}_seed{}".format(roi,seed)
        for ext in ("png","svg"):
            fig.savefig(out/(name+"."+ext),dpi=190,bbox_inches="tight")
        plt.close(fig);figures.append(name)
    with (out/"mechanism_plot_data.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,list(data[0]));writer.writeheader();writer.writerows(data)
    manifest=dict(status="PASS",protocol=PROTOCOL,source_analysis=str(args.analysis.resolve()),source_audit_status=audit["status"],
                  actual_formal_runs=64,selected_uncapped_intervention_rows=len(data),figures=figures,
                  seeds=sorted({r["seed"] for r in data}),all_rows_retained=True,averaging=False,
                  limitations=["Only existing seed7; no independent-training robustness claim.",
                    "MAC0 IPC is synchronous; no finite-cost IPC inference.",
                    "No capped mechanism intervention is inferred from uncapped controls.",
                    "Different actions can change future callback streams; whole-run state identity not claimed.",
                    "Template is validation-selected joint per-slot delta/fill plus unconditional modalK; no assumed +2/PC.",
                    "No prefix extrapolation; all displayed rows are complete25M ROI.",
                    "SPP native raw sourcecall K differs from canonical deduplicated training-action K."])
    (out/"figure_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS",selected_rows=len(data),figures=figures,out=str(out))))


if __name__=="__main__":
    main()
