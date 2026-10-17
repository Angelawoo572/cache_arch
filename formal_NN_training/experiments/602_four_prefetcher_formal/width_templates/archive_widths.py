#!/usr/bin/env python3
"""Copy audited width/template CSVs, figures and metadata into a fresh light archive."""
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import time

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=HERE.parents[3]
CACHE=next(p for p in HERE.parents if p.name=="cache")
PROTOCOL="four602_width_templates_20261017_v1"
ANALYSIS_FILES=(
    "audit.json","metric_definitions.json","selection_verification.json","callback_verification.json",
    "binding_verification.json","reference_bindings.json","inventory.json","figure_manifest.json",
    "cost_attachment_verification.json","cost_fields_h16.json","cost_fields_h8_original.json",
    "reference_audit_original.json","FINDINGS.md","PAPER_STORY.md",
    "per_run_new.csv","per_run.csv","reference_per_run_original.csv","raw_flat_new.csv",
    "paired_comparisons.csv","action_histograms.csv","action_histograms_full_roi.csv",
    "classified_requests.csv","progress_snapshots.csv","phase_progress.csv","storage_cost.csv",
    "cost_plot_data.csv","startup_plot_data.csv","distribution_plot_data.csv","phase_plot_data.csv",
    "comparison_plot_data.csv","route_status.csv")
RAW_FILES=("bindings.json","reference_bindings.json","correctness_gate.json","selection.json","matrix.json")
TEMPLATES=("ampm_h16.txt","spp_h16.txt","streamer_h8.txt","ampm_h8.txt","spp_h8.txt")


def read(p):return json.loads(Path(p).read_text())


def mapped(path):
    text=str(path);prefix="/home/qianruw/cache/"
    if text.startswith(prefix):
        text=text[len(prefix):]
        for old,new in (("formal_study_20261016/","formal_study/archive/20261016/"),
                        ("formal_study_20261017/","formal_study/archive/20261017/")):
            if text.startswith(old):text=new+text[len(old):];break
        return CACHE/text
    return Path(path)


def rows(path):
    with Path(path).open(newline="") as stream:return list(csv.DictReader(stream))


def write_csv(path, data):
    fields=list(dict.fromkeys(key for row in data for key in row))
    with Path(path).open("w",newline="") as stream:
        if not fields:return
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(data)


def subset_plans(analysis, reference, combined, destination):
    plans=[]
    by_id={row["run_id"]:row for row in combined}
    if len(by_id)!=88:raise ValueError("Original and supplemental run_ids must be disjoint")
    shared=("action_histograms.csv","action_histograms_full_roi.csv","classified_requests.csv",
            "progress_snapshots.csv","phase_progress.csv","storage_cost.csv")
    for policy,expected in (("streamer",21),("ampm",30),("spp",30)):
        folder=EXP.parent/("602_lstm_"+policy+"_live_inference")/"results/width_templates_20261017"
        if folder.exists():raise ValueError("Fresh sibling subset required: "+str(folder))
        selected=[row for row in combined if row["policy"] in (policy,"none")]
        if len(selected)!=expected or sum(row["policy"]=="none" for row in selected)!=3:
            raise ValueError("Unexpected policy + three matched NoPF subset count: "+policy)
        ids={row["run_id"] for row in selected}
        data={"per_run.csv":selected}
        for name in shared:
            joined=[]
            for source in (reference/"analysis_final"/name,analysis/name):
                for row in rows(source):
                    if row["run_id"] not in ids:continue
                    label=by_id[row["run_id"]]
                    for field in ("result_id","protocol","evidence_origin","stage","roi","policy","method","mode","budget","seed","hidden_size"):
                        row.setdefault(field,label.get(field,""))
                    row["source_table"]=str(source)
                    joined.append(row)
            data[name]=joined
        data["paired_comparisons.csv"]=[row for row in rows(analysis/"paired_comparisons.csv") if row["policy"]==policy]
        model=folder.parents[1]/"models"/("h16/original_checkpoint.pt" if policy in ("ampm","spp") else "original_checkpoint.pt")
        if not model.is_file():raise ValueError("Retained real sibling checkpoint missing: "+str(model))
        plans.append(dict(policy=policy,folder=folder,rows=expected,data=data,ids=sorted(ids),model=model))
    return plans


def write_subsets(plans, destination, analysis, raw):
    summaries=[]
    for plan in plans:
        folder=plan["folder"];folder.mkdir(parents=True)
        for name,data in plan["data"].items():write_csv(folder/name,data)
        aggregate=Path(os.path.relpath(destination,folder))
        model=Path(os.path.relpath(plan["model"],folder))
        text=["# "+plan["policy"].upper()+" width/template formal subset", "",
              "Complete policy rows plus three exact matched NoPF references: **{} rows**. Original protocols and result identities are retained; references are not new runs.".format(plan["rows"]), "",
              "[Aggregate findings]({}/FINDINGS.md), [one-page story]({}/PAPER_STORY.md), [all88 labeled rows]({}/per_run.csv), [strict audit]({}/audit.json), [method]({}/method.svg).".format(aggregate,aggregate,aggregate,aggregate,aggregate), "",
              "[This policy + matched NoPF](per_run.csv), [paired contrasts](paired_comparisons.csv), [entire-ROI action histograms](action_histograms_full_roi.csv), [phase histograms](action_histograms.csv), [classified request outcomes](classified_requests.csv), [actual startup snapshots](progress_snapshots.csv), [phase progress](phase_progress.csv), [storage/cost](storage_cost.csv).", "",
              "[Retained real {} checkpoint]({}) is the source model asset; no online training or new seed is implied.".format("H16" if plan["policy"] in ("ampm","spp") else "H8",model), "",
              "CSV profiles join the unmodified original64-source tables and supplemental24-source tables, filter actual run_ids, and add protocol/origin/result labels plus source_table. This is a transparent subset, not a new measurement or an average. All rows use actual complete ROIs; no prefixes are extrapolated.", "",
              "MAC0 synchronous inference only. Same ceiling does not imply equal realized requests/DRAM. L2/LLC target cohorts remain separate; unused is not pollution. SPP native keeps prefetch-fill FILTER feedback while the NN sees all raw L2 fills. Exact actual binary/configured Sacramento controls are reused; independent historical hardware telemetry is absent.", ""]
        (folder/"README.md").write_text("\n".join(text))
        manifest=dict(status="PASS",policy=plan["policy"],source_analysis=str(analysis),raw_root=str(raw),aggregate_archive=str(destination),
                      selected_rows=plan["rows"],selected_run_ids=plan["ids"],matched_NoPF_rows=3,
                      source_protocols=["four602_frozen_live_20261017_v2",PROTOCOL],
                      outputs={name:len(data) for name,data in plan["data"].items()},raw_inputs_modified=False,
                      original_values_preserved=True,added_fields="explicit protocol/origin/result_id and source_table where applicable")
        (folder/"subset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        summaries.append(dict(policy=plan["policy"],directory=str(folder),rows=plan["rows"],files=len(plan["data"])+2))
    return summaries


def archive(args):
    analysis,raw,destination=args.analysis.resolve(),args.raw.resolve(),args.destination.resolve()
    for name,path in (("analysis",analysis),("raw",raw),("destination",destination)):
        if not path.is_relative_to(CACHE.resolve()):raise ValueError(name+" must be inside cache")
    if destination.exists():raise ValueError("Fresh destination required; prior archives are preserved")
    if destination.is_relative_to(analysis) or destination.is_relative_to(raw):
        raise ValueError("Archive must be separate from original raw/analysis")
    audit=read(analysis/"audit.json")
    required=(audit.get("status")=="PASS" and audit.get("protocol")==PROTOCOL
              and audit.get("complete_count")==24 and audit.get("expected_count")==24
              and audit.get("new_test_count")==22 and audit.get("new_validation_count")==2
              and audit.get("reference_count")==64 and audit.get("failure_count")==0
              and audit.get("callback_logs_checked") is True and audit.get("plot_status")=="generated"
              and audit.get("raw_inputs_modified") is False and audit.get("reference_data_rewritten") is False)
    if not required:raise ValueError("Only strict PASS24/22test2val + original64 + fully generated figures may be archived")
    if mapped(audit["raw_root"]).resolve()!=raw:raise ValueError("Audit/raw input mismatch")
    new=rows(analysis/"per_run_new.csv");combined=rows(analysis/"per_run.csv")
    if len(new)!=24 or sum(r["stage"]=="test" for r in new)!=22 or sum(r["stage"]=="validation" for r in new)!=2:
        raise ValueError("Actual new CSV does not contain24 complete expected runs")
    if len(combined)!=88 or any(r["status"]!="complete" for r in combined):
        raise ValueError("Expected24 new +64 explicitly labeled completed reference CSV rows")
    if len({r["result_id"] for r in combined})!=88:
        raise ValueError("Result identities must retain distinct original protocols")
    reference=mapped(audit["reference_raw"])
    if (analysis/"reference_per_run_original.csv").read_bytes()!=(reference/"analysis_final/per_run.csv").read_bytes():
        raise ValueError("Original64 CSV is not byte-equal")
    callback=read(analysis/"callback_verification.json")
    required_ids={r["run_id"] for r in new if r["method"]=="student" and r["mode"]=="full" and int(r["budget"])==0}
    by_callback={r["run_id"]:r for r in callback}
    if (len(callback)!=24 or len(by_callback)!=24 or set(by_callback)!={r["run_id"] for r in new}
            or len(required_ids)!=6 or not all(by_callback[rid].get("complete_log") is True for rid in required_ids)):
        raise ValueError("Require six actual uncapped H16 full callback logs, all complete")
    cost=read(analysis/"cost_attachment_verification.json")
    if cost.get("available") is not True or len(cost.get("attached_runs",[]))!=10:
        raise ValueError("Require exact H16 cost attachment for10 actual full-student rows")
    figure=read(analysis/"figure_manifest.json")
    if figure.get("status")!="PASS" or figure.get("new_complete_runs")!=24 or figure.get("reference_complete_runs")!=64:
        raise ValueError("Wrong complete-figure manifest")
    plan=[(analysis/name,Path(name)) for name in ANALYSIS_FILES]
    names=figure.get("figures",[])
    if len(names)!=18 or len(set(names))!=18:raise ValueError("Require all18 actual scientific figure pairs")
    for name in names:
        if not re.fullmatch(r"[a-z0-9_]+",name):raise ValueError("Unsafe figure filename")
        for extension in ("png","svg"):plan.append((analysis/(name+"."+extension),Path(name+"."+extension)))
    for name in RAW_FILES:plan.append((raw/name,Path("metadata")/name))
    plan.append((raw/"protocol_source/config.json",Path("metadata/config.json")))
    plan.append((reference/"build.json",Path("metadata/reference_actual_build.json")))
    for name in TEMPLATES:plan.append((raw/"templates"/name,Path("metadata/templates")/name))
    if (raw/"pipeline_result.json").is_file():plan.append((raw/"pipeline_result.json",Path("metadata/pipeline_result.json")))
    method=args.method.resolve()
    for extension in ("svg","png"):
        source=analysis/("method."+extension)
        if source.read_bytes()!=method.with_suffix("."+extension).read_bytes():
            raise ValueError("Analyzed/rendered method bytes changed")
        plan.append((source,Path("method."+extension)))
    targets=[str(relative) for _,relative in plan]
    if len(targets)!=len(set(targets)):raise ValueError("Duplicate archive target")
    for source,relative in plan:
        if not source.is_file() or not source.resolve().is_relative_to(CACHE.resolve()):
            raise ValueError("Missing/outside-cache archive source "+str(source))
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Invalid relative archive path")
        if source.name in ("raw_snapshots.json","run.log") or source.suffix in (".bin",".pt",".xz",".gz"):
            raise ValueError("Large raw/model/log material must stay outside light archive")
    subsets=subset_plans(analysis,reference,combined,destination)
    destination.mkdir(parents=True)
    copied=[]
    for source,relative in plan:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        if target.read_bytes()!=source.read_bytes():raise RuntimeError("Copy not byte-equal "+str(relative))
        copied.append(dict(source=str(source.resolve()),relative_destination=str(relative),bytes=source.stat().st_size))
    notes=["# Width/template formal light archive","",
           "This archive contains24 new complete runs (22 test +2 validation),64 explicitly labeled compatible references, all18 scientific PNG/SVG figure pairs, full-ROI/phase tables and original metadata. Every copied source is byte-equal; no raw source is edited.","",
           "[Findings](FINDINGS.md), [per-run data](per_run.csv), [new24](per_run_new.csv), [unaltered original64](reference_per_run_original.csv), [audit](audit.json), [method](method.svg), [copy manifest](archive_manifest.json).","",
           "Large snapshots, simulator logs, traces, checkpoints and portable model binaries remain in the complete cache/formal_study raw/archive trees. Their source paths are retained in metadata and the manifest. No replay/offline IPC is combined with live IPC.","",
           "All measured IPC assumes synchronous MAC0. H16 is an existing seed7 model-width comparison, not an additional training seed. The inherited ceiling does not equalize actual requests or DRAM. Old reference protocols remain distinct. Historical hardware telemetry is unavailable beyond the configured Sacramento environment and exact simulator-byte reuse.",""]
    (destination/"README.md").write_text("\n".join(notes))
    subset_summaries=write_subsets(subsets,destination,analysis,raw)
    manifest=dict(status="PASS",protocol=PROTOCOL,created_ns=time.time_ns(),source_analysis=str(analysis),raw_root=str(raw),
                  reference_raw=str(reference),destination=str(destination),new_complete_runs=24,reference_runs=64,
                  copied_files=len(copied),copied_bytes=sum(r["bytes"] for r in copied),files=copied,
                  byte_equal_all_copies=True,raw_inputs_modified=False,reference_data_rewritten=False,subsets=subset_summaries,
                  excluded=["raw_snapshots.json","run.log","full callback streams","simulator binary","model.bin","model.pt","traces"])
    (destination/"archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("status","destination","copied_files","copied_bytes","new_complete_runs","reference_runs")}))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis",type=Path,required=True);p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--destination",type=Path,default=EXP/"results/width_templates_20261017")
    p.add_argument("--method",type=Path,default=ROOT/"docs/602_width_templates/method.svg")
    return archive(p.parse_args())


if __name__=="__main__":raise SystemExit(main())
