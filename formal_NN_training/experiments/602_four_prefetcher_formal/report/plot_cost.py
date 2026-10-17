#!/usr/bin/env python3
"""Plot complete measured MAC0 IPC against counted work and partial payload."""
import argparse,csv,json,math,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
POLICIES=('stride','streamer','ampm','spp')
COLORS=dict(stride='#4c78a8',streamer='#f58518',ampm='#54a24b',spp='#b279a2')
LABELS=dict(stride='Stride',streamer='Streamer',ampm='AMPM',spp='SPP')
def numeric(row,key):
    value=row.get(key)
    if value in (None,'','None','null','NA'):return None
    result=float(value)
    if not math.isfinite(result):raise ValueError('nonfinite '+key)
    return result
def read_csv(path):
    with Path(path).open(newline='') as handle:return list(csv.DictReader(handle))
def derive(rows,storage,cost):
    output=[]
    for raw in rows:
        if raw.get('status')!='complete':continue
        row=dict(storage.get(raw['run_id'],{}),**raw);method=row['method'];neural=method in ('student','stride_student')
        ins=numeric(row,'instructions');ipc=numeric(row,'ipc');cycles=numeric(row,'cycles')
        if not ins or ipc is None:raise ValueError('missing measured IPC/instructions '+row['run_id'])
        mac=numeric(row,'inference_macs');nl=numeric(row,'inference_nonlinears')
        work_available=neural and mac is not None and mac>0
        weights=numeric(row,'model_weight_bytes');hc=numeric(row,'recurrent_hc_bytes');scope='not_applicable'
        if method=='student':
            model=cost.get('nn_models',{}).get(row['policy'],{}).get('model',{})
            if model:
                if weights is not None and weights!=model['fp32_weight_bytes']:raise ValueError('model weight mismatch '+row['run_id'])
                if hc is not None and hc!=model['global_hc_bytes']:raise ValueError('global h/c mismatch '+row['run_id'])
                weights=model['fp32_weight_bytes'];hc=model['global_hc_bytes']
            scope='FP32 weights + one global h/c pair; excludes scratch/output/control'
        elif method=='stride_student':
            # state_bytes_configured contains metadata; do not mislabel it as h/c.
            h=numeric(row,'raw__h');capacity=numeric(row,'state_capacity')
            if h is not None and capacity is not None:hc=8*h*capacity
            scope='FP32 weights + configured per-PC h/c payload; excludes PC tags/metadata/FIFOs/scratch/output/control'
        out={k:row.get(k) for k in ('run_id','protocol','stage','roi','policy','method','mode','budget','seed','build_id','instructions','cycles','ipc','completed_updates')}
        out.update(counted_nn_macs=mac if work_available else None,
            counted_nn_nonlinears=nl if work_available else None,
            counted_nn_macs_per_1k_retired=1000*mac/ins if work_available else None,
            counted_nn_nonlinears_per_1k_retired=1000*nl/ins if work_available and nl is not None else None,
            analytic_mac4_dense_cycles_per_1k_retired=1000*math.ceil(mac/4)/ins if work_available else None,
            analytic_mac4_dense_plus_4cycle_nonlinear_cycles_per_1k_retired=1000*(math.ceil(mac/4)+4*nl)/ins if work_available and nl is not None else None,
            required_dense_macs_per_measured_mac0_roi_cycle=mac/cycles if work_available and cycles else None,
            mac4_dense_work_cycles_div_measured_mac0_roi_cycles=math.ceil(mac/4)/cycles if work_available and cycles else None,
            mac4_dense_work_exceeds_measured_mac0_roi_window=math.ceil(mac/4)>cycles if work_available and cycles else None,
            model_weight_bytes=weights if neural else None,hc_payload_bytes=hc if neural else None,
            weights_plus_hc_payload_bytes=weights+hc if neural and weights is not None and hc is not None else None,
            payload_scope=scope,work_scope=('existing Stride runtime counters; separate implementation definition' if method=='stride_student' else 'shared scalar actual dense MAC/nonlinear counters') if work_available else 'NA: native compute not instrumented or no measured NN work',
            cost_scope='Measured IPC is synchronous MAC0. MAC4 columns are aggregate operation arithmetic, not finite-cost IPC/latency/PPA. Ratio>1 cannot serve all counted ROI work at4 MAC/core-cycle within the same measured MAC0 callback timeline/window even ignoring all other costs. Ratio<=1 is not sufficient for feasible latency/queues/state dependencies. Finite scheduling can change arrivals, drops, pending work and IPC for the same program trace. Native compute not zeroed or guessed.')
        output.append(out)
    return output
def main():
    p=argparse.ArgumentParser();p.add_argument('--per-run',type=Path,required=True);p.add_argument('--storage-cost',type=Path)
    p.add_argument('--cost-fields',type=Path,default=Path(__file__).with_name('cost_fields.json'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=('test','validation'),default='test');a=p.parse_args()
    cache=next((x for x in a.out.resolve().parents if x.name=='cache'),None)
    if cache is None or not a.out.resolve().is_relative_to(cache/'formal_study_20261017'):raise ValueError('output must stay in cache/formal_study_20261017')
    runtime=cache/'formal_study_20261017/runtime_v2'
    for key,name in [('TMPDIR','tmp'),('PYTHONPYCACHEPREFIX','pycache'),('MPLCONFIGDIR','mpl')]:
        directory=runtime/name;directory.mkdir(parents=True,exist_ok=True);os.environ[key]=str(directory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    a.out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(a.per_run);storage={r['run_id']:r for r in read_csv(a.storage_cost)} if a.storage_cost else {}
    data=derive(rows,storage,json.loads(a.cost_fields.read_text()))
    with (a.out/'cost_derived.csv').open('w',newline='') as handle:
        if not data:raise ValueError('no complete rows')
        writer=csv.DictWriter(handle,list(data[0]));writer.writeheader();writer.writerows(data)
    selected=[r for r in data if r['stage']==a.stage and r['method'] in ('student','stride_student') and r['mode']=='full']
    rois=sorted({int(r['roi']) for r in selected})
    if not rois:raise ValueError('no completed full student rows for selected stage')
    protocols={r['protocol'] for r in data};builds={r['build_id'] for r in data}
    if len(protocols)!=1 or len(builds)!=1:raise ValueError('do not mix protocols/builds')
    fig,axes=plt.subplots(len(rois),2,figsize=(11,4.3*len(rois)),squeeze=False)
    unavailable=[]
    for rowindex,roi in enumerate(rois):
        candidates=[r for r in selected if int(r['roi'])==roi]
        baselines=[r for r in data if r['stage']==a.stage and int(r['roi'])==roi and r['method']=='none' and r['policy']=='none']
        if len(baselines)!=1:raise ValueError('need exactly one complete same-ROI NoPF')
        baseline=float(baselines[0]['ipc'])
        for ax in axes[rowindex]:
            ax.axhline(baseline,color='#777777',ls='--',lw=1);ax.text(.98,.03,f'NoPF IPC {baseline:.4f}',ha='right',transform=ax.transAxes,fontsize=8,color='#666666',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            ax.set_ylabel('Measured IPC (MAC0)');ax.grid(alpha=.18);ax.set_axisbelow(True)
        for row in candidates:
            policy=row['policy'];budget=float(row['budget']);marker='^' if budget else 'o';ipc=float(row['ipc'])
            work=row['counted_nn_macs_per_1k_retired'];payload=row['weights_plus_hc_payload_bytes']
            if work is None:unavailable.append(row['run_id'])
            else:
                axes[rowindex,0].scatter(work,ipc,c=COLORS[policy],marker=marker,s=65,edgecolor='white',linewidth=.6,zorder=3)
                if not budget and row['mac4_dense_work_cycles_div_measured_mac0_roi_cycles'] is not None:
                    offset={'stride':(6,7),'streamer':(6,7),'ampm':(6,-14),'spp':(-5,7)}[policy]
                    axes[rowindex,0].annotate('r={:.2f}'.format(row['mac4_dense_work_cycles_div_measured_mac0_roi_cycles']),(work,ipc),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if policy=='spp' else 'left',color=COLORS[policy])
            if payload is not None:axes[rowindex,1].scatter(payload/1024,ipc,c=COLORS[policy],marker=marker,s=65,edgecolor='white',linewidth=.6,zorder=3)
        axes[rowindex,0].set_title(f'{a.stage.title()} ROI {roi}: counted work');axes[rowindex,0].set_xlabel('Counted NN MACs / 1K retired instructions')
        axes[rowindex,1].set_title(f'{a.stage.title()} ROI {roi}: partial model payload');axes[rowindex,1].set_xlabel('FP32 weights + h/c payload (KiB)\n(partial; excludes control, queues, scratch/output)')
        for ax in axes[rowindex]:ax.margins(x=.12,y=.17)
    legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[p],markersize=7,label=LABELS[p]) for p in POLICIES if any(r['policy']==p for r in selected)]
    if any(float(r['budget'])==0 for r in selected):legend += [Line2D([0],[0],marker='o',color='none',markerfacecolor='#555555',label='Uncapped')]
    if any(float(r['budget'])>0 for r in selected):legend += [Line2D([0],[0],marker='^',color='none',markerfacecolor='#555555',label='Common ceiling')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.09),ncol=6,frameon=False)
    fig.suptitle('Frozen students: measured MAC0 effect versus counted work / partial storage',fontsize=13)
    fig.text(.5,.052,'r = ceil(dense MACs / 4) / measured MAC0 ROI cycles (uncapped labels). Native compute is not instrumented.',ha='center',fontsize=8.5)
    fig.text(.5,.018,'r > 1 exceeds throughput in the same MAC0 callback timeline/window; r ≤ 1 is not sufficient. No finite-cost IPC/PPA prediction.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.18,1,.95))
    fig.savefig(a.out/'cost_effect.png',dpi=180);fig.savefig(a.out/'cost_effect.svg');plt.close(fig)
    note=dict(status='complete',protocols=sorted(protocols),builds=sorted(builds),stage=a.stage,plotted_full_student_rows=len(selected),derived_complete_rows=len(data),
        missing_work_run_ids=unavailable,source_per_run=str(a.per_run),source_storage_cost=str(a.storage_cost),source_cost_fields=str(a.cost_fields),
        interpretation='MAC0 measured IPC only; counted operation totals and partial weights+h/c payload. Native compute NA. r=ceil(MAC/4)/measured MAC0 cycles is a necessary aggregate-throughput test for serving all counted work in the same MAC0 callback timeline/window. r<=1 is not sufficient; r>1 does not predict finite IPC, and finite scheduling can change arrivals for the same program trace. Mechanism rows retained in derived CSV but main effect/cost plot shows full models only.')
    (a.out/'cost_plot_manifest.json').write_text(json.dumps(note,indent=2)+'\n');print(json.dumps(note))
if __name__=='__main__':main()
