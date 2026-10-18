#!/usr/bin/env python3
"""Rebuild reader-only tables/figures from completed, labeled runs; never simulates.

Run from repository root with Python + matplotlib. All outputs remain under
formal_study/results/602_width_templates. Source CSVs are read-only.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import csv
import math
import os

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'formal_study/results/602_width_templates'
# A direct reader rebuild must obey the same project-only runtime contract.
for variable, relative in (('TMPDIR', '.runtime/tmp'), ('TMP', '.runtime/tmp'),
                           ('TEMP', '.runtime/tmp'), ('MPLCONFIGDIR', '.runtime/mpl'),
                           ('XDG_CACHE_HOME', '.runtime/xdg')):
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
import tempfile
tempfile.tempdir = str(ROOT / '.runtime/tmp')
POLICIES = ('stride', 'streamer', 'ampm', 'spp')
NAMES = dict(stride='Stride', streamer='Streamer', ampm='AMPM', spp='SPP', none='NoPF')
REFERENCE = 'four602_frozen_live_20261017_v2'
SUPPLEMENT = 'four602_width_templates_20261017_v1'


def load():
    with (OUT / 'per_run.csv').open(newline='') as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for s in source:
        if s['stage'] != 'test' or s['mode'] != 'full':
            continue
        arm = 'nopf' if s['policy'] == 'none' and s['method'] == 'none' else 'native' if s['method'] == 'native' else 'nn' if s['method'] in ('student', 'stride_student') else ''
        if not arm:
            continue
        h = int(s.get('hidden_size') or s.get('raw__h') or 0) if arm == 'nn' else 0
        expected = SUPPLEMENT if h == 16 else REFERENCE
        assert s['protocol'] == expected, s['result_id']
        r = {k:s[k] for k in ('result_id', 'run_id', 'protocol', 'evidence_origin', 'source_raw', 'source_run', 'policy')}
        r.update(arm=arm, hidden_size=h)
        for k in ('roi', 'budget', 'seed', 'instructions', 'cycles', 'l2_load_misses'):
            r[k] = int(s[k]) if s[k] else ''
        r.update(ipc=float(s['ipc']), ipc_speedup_vs_NoPF=float(s['ipc_speedup_vs_NoPF']),
                 requests_per_million_instructions=float(s['requests_per_million_instructions']),
                 allowed_requests=int(s['raw__request_gate_allowed']), dram_bus_bytes=int(s['raw__dram_total_bus_bytes']))
        for k in ('l2_load_mpki', 'net_l2_load_miss_reduction', 'net_llc_load_miss_reduction'):
            r[k] = float(s[k]) if s[k] else None
        assert math.isclose(r['ipc'], r['instructions']/r['cycles'], rel_tol=1e-12)
        assert math.isclose(r['requests_per_million_instructions'], 1e6*r['allowed_requests']/r['instructions'], rel_tol=1e-12, abs_tol=1e-12)
        for level in ('l2', 'llc'):
            available = s[level+'_target_metrics_available'].lower() == 'true' and arm != 'nopf'
            values = dict(enqueued=int(s[level+'_target_enqueued']), timely_load=int(s[level+'_target_timely_load']), late_load=int(s[level+'_target_late_load']), useful_load=int(s[level+'_target_LOAD_useful']))
            assert values['useful_load'] == values['timely_load'] + values['late_load']
            r.update({level+'_'+k:v for k,v in values.items()})
            den = int(s['no_pf_'+level.upper()+'_LOAD_misses'])
            r[level+'_available'] = available
            r[level+'_accuracy'] = values['useful_load']/values['enqueued'] if available and values['enqueued'] else None
            r[level+'_coverage_proxy'] = values['useful_load']/den if available and den else None
            r[level+'_timeliness'] = values['timely_load']/values['useful_load'] if available and values['useful_load'] else None
        rows.append(r)
    assert len(rows) == 42
    assert sum(r['arm']=='nopf' for r in rows) == 2
    assert sum(r['arm']=='nn' and r['hidden_size']==16 for r in rows) == 8
    assert len({r['result_id'] for r in rows}) == len(rows)
    return sorted(rows, key=lambda r:(r['roi'],r['budget'],POLICIES.index(r['policy']) if r['policy'] in POLICIES else -1,r['hidden_size'],r['arm']))


def dump(name, rows):
    with (OUT/name).open('w', newline='') as stream:
        writer=csv.DictWriter(stream, list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def dominates(a, b, cost):
    return a['ipc'] >= b['ipc'] and a[cost] <= b[cost] and (a['ipc'] > b['ipc'] or a[cost] < b[cost])


def relation(native, nn, cost):
    if dominates(nn,native,cost): return 'nn_dominates'
    if dominates(native,nn,cost): return 'native_dominates'
    if nn['ipc']==native['ipc'] and nn[cost]==native[cost]: return 'exact_tie'
    return 'tradeoff'


def paired(rows):
    pairs=[]
    for nn in rows:
        if nn['arm']!='nn':continue
        native, = [r for r in rows if r['arm']=='native' and all(r[k]==nn[k] for k in ('policy','roi','budget'))]
        p={k:nn[k] for k in ('policy','hidden_size','roi','budget')}
        p.update(native_result_id=native['result_id'],nn_result_id=nn['result_id'],native_ipc=native['ipc'],nn_ipc=nn['ipc'],ipc_gain_pct=100*(nn['ipc']/native['ipc']-1),
                 native_requests_per_million=native['requests_per_million_instructions'], nn_requests_per_million=nn['requests_per_million_instructions'],requests_change_pct=100*(nn['requests_per_million_instructions']/native['requests_per_million_instructions']-1),
                 native_dram_bus_bytes=native['dram_bus_bytes'],nn_dram_bus_bytes=nn['dram_bus_bytes'],dram_change_pct=100*(nn['dram_bus_bytes']/native['dram_bus_bytes']-1),
                 paired_requests_relation=relation(native,nn,'requests_per_million_instructions'),paired_dram_relation=relation(native,nn,'dram_bus_bytes'))
        for level in ('l2','llc'):
            for metric in ('enqueued','accuracy','coverage_proxy','timeliness'):
                for arm,r in (('native',native),('nn',nn)):
                    p[arm+'_'+level+'_'+metric]=r[level+'_'+metric]
        pairs.append(p)
    assert len(pairs)==24
    return pairs


def frontiers(rows):
    result=[]
    for candidate_set,max_h in (('H8_primary',8),('H8_plus_H16_supplement',16)):
        for roi in (0,1):
            for cap in (0,16144):
                selected=[r for r in rows if r['roi']==roi and r['hidden_size']<=max_h and (r['budget']==cap or r['arm']=='nopf')]
                assert len(selected)==(9 if max_h==8 else 11)
                for cost in ('requests_per_million_instructions','dram_bus_bytes'):
                    for r in selected:
                        result.append(dict(candidate_set=candidate_set,roi=roi,budget=cap,cost_axis=cost,result_id=r['result_id'],actual_run_budget=r['budget'],x_cost=r[cost],ipc=r['ipc'],is_pareto=not any(dominates(s,r,cost) for s in selected)))
    return result


def pct(v):return 'NA' if v is None else f'{v*100:.1f}'
def number(v):return f'{v:,.0f}'
def gain(v):return f'{v:+.2f}%'
def budget(cap):return 'uncapped' if cap==0 else 'cap 16,144/M'

def document(rows,pairs,fronts):
    lookup={(r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget']):r for r in rows}
    def pair(policy,h,roi,cap):return next(p for p in pairs if (p['policy'],p['hidden_size'],p['roi'],p['budget'])==(policy,h,roi,cap))
    out=['# 602: native prefetchers versus frozen NN students','',
         '- **Start here:** paired IPC, accuracy, coverage proxy and timeliness for Stride, Streamer, AMPM and SPP. [Project overview](../../../RESULTS_INDEX.md).',
         '- **Measured scope:** complete 25M-instruction ROIs; original seed7 models; frozen weights with live state updates; synchronous zero-cost inference (MAC0). These are operating points, not a finite-hardware efficiency ranking.',
         '- **Main comparison:** H8 NN versus its own native teacher. H16 AMPM/SPP is a separate width supplement. NoPF is the same matched reference in both budget panels.',
         '- **Evidence:** [42 selected run rows](reader_metrics.csv) · [24 native/NN pairs](reader_pairs.csv) · [Pareto membership](reader_pareto.csv) · [reader audit](reader_audit.json) · [all 88 source rows](per_run.csv). Selection excludes validation, templates, reset-state and fixed-payload interventions.','',
         '## Main result: H8, common request ceiling','',
         '- **IPC:** NN wins 6/8 capped pairs: both Stride, Streamer and AMPM ROIs; SPP loses both. Across capped and uncapped cases, NN wins 9/16.',
         '- **Quality:** higher accuracy does not imply higher IPC. SPP H8 mainly targets LLC; its L2-only figures do not describe the whole policy.',
         '- **Traffic:** all eight capped NN cases transfer more DRAM bytes than their own native baseline. The common ceiling is not equal actual traffic.',
         '- Values below are **native → NN**. IPC gain is relative to that native, not NoPF. Quality values are percentages; **C is a coverage proxy**, defined below.','',
         '| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for roi in (0,1):
        for pol in POLICIES:
            n=lookup[(pol,'native',0,roi,16144)];s=lookup[(pol,'nn',8,roi,16144)];p=pair(pol,8,roi,16144)
            out.append(f"| {roi} | {NAMES[pol]} | {n['ipc']:.4f} → {s['ipc']:.4f} | {gain(p['ipc_gain_pct'])} | {pct(n['l2_accuracy'])} → {pct(s['l2_accuracy'])} | {pct(n['l2_coverage_proxy'])} → {pct(s['l2_coverage_proxy'])} | {pct(n['l2_timeliness'])} → {pct(s['l2_timeliness'])} |")
    out+=['','- SPP NN L2 enqueues: **ROI0 = 0; ROI1 = 14** under the cap (13 useful). ROI0 accuracy/timeliness are NA; ROI1’s high accuracy is a tiny cohort. Use the LLC table below.',
          '','![H8 native and NN quality and IPC, capped](reader_quality_cap16144.png)','','## SPP: LLC-intent quality','',
          '- Completely tracked LLC-intent cohorts are reported separately from L2 intent. These are the first enqueue’s intent, not a guarantee of final fill level.',
          '- Coverage proxy can exceed 100% because useful requests are not a count of unique baseline misses eliminated. Do not add L2 and LLC proxies.','',
          '| Budget | ROI | Arm | LLC enqueued | Accuracy % | C % | Timeliness % |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for cap in (16144,0):
        for roi in (0,1):
            for arm in ('native','nn'):
                r=lookup[('spp',arm,0 if arm=='native' else 8,roi,cap)]
                out.append(f"| {budget(cap)} | {roi} | {'Native' if arm=='native' else 'H8 NN'} | {number(r['llc_enqueued'])} | {pct(r['llc_accuracy'])} | {pct(r['llc_coverage_proxy'])} | {pct(r['llc_timeliness'])} |")
    out+=['','## IPC versus actual requests and DRAM','',
          '- **Pareto axes:** higher IPC; fewer actual allowed cache calls/M or fewer completed DRAM bus bytes. A point is dominated only when another has no lower IPC and no higher cost, with at least one strict improvement.',
          '- **Candidate set:** the four natives, their four H8 students and NoPF; each ROI and cap is evaluated separately. Finite compute, training cost and native hardware area are not included.',
          '- **Capped own-native pairs:** Stride, Streamer and AMPM trade higher IPC for more DRAM; native SPP dominates its H8 NN on both IPC and DRAM in both ROIs. No capped H8 NN is a paired IPC/DRAM Pareto improvement.',
          '- Frontiers are descriptive for these measured points. They are not statistical significance or a global optimum; coincident points remain tied. NoPF in the capped panel is a reused reference, not a separate capped run.','',
          '![H8/native/NoPF IPC versus actual requests and DRAM, capped](reader_pareto_cap16144.png)','',
          '| ROI | Policy | Actual requests/M native → NN | DRAM MiB native → NN | DRAM change | Net L2 miss reduction % native → NN |',
          '| --- | --- | --- | --- | --- | --- |']
    for roi in (0,1):
        for pol in POLICIES:
            n=lookup[(pol,'native',0,roi,16144)];s=lookup[(pol,'nn',8,roi,16144)];p=pair(pol,8,roi,16144)
            out.append(f"| {roi} | {NAMES[pol]} | {number(n['requests_per_million_instructions'])} → {number(s['requests_per_million_instructions'])} | {n['dram_bus_bytes']/2**20:.3f} → {s['dram_bus_bytes']/2**20:.3f} | {gain(p['dram_change_pct'])} | {pct(n['net_l2_load_miss_reduction'])} → {pct(s['net_l2_load_miss_reduction'])} |")
    out+=['','- For example, capped ROI0 Stride emits about twice as many allowed requests under NN. The shared ceiling alone cannot establish an equal-traffic benefit.',
          '- [Exact requests, bytes and paired changes](reader_pairs.csv) · [Frontier flags for both H8-only and H8+H16 candidate sets](reader_pareto.csv).','',
          '## Uncapped H8 comparison','',
          '- NN wins 3/8 uncapped pairs. Streamer loses ROI1; AMPM and SPP lose both. This is retained alongside the capped gains.',
          '- The same definitions and complete-ROI scope apply; these are different operating points from the capped rows.','',
          '| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for roi in (0,1):
        for pol in POLICIES:
            n=lookup[(pol,'native',0,roi,0)];s=lookup[(pol,'nn',8,roi,0)];p=pair(pol,8,roi,0)
            out.append(f"| {roi} | {NAMES[pol]} | {n['ipc']:.4f} → {s['ipc']:.4f} | {gain(p['ipc_gain_pct'])} | {pct(n['l2_accuracy'])} → {pct(s['l2_accuracy'])} | {pct(n['l2_coverage_proxy'])} → {pct(s['l2_coverage_proxy'])} | {pct(n['l2_timeliness'])} → {pct(s['l2_timeliness'])} |")
    out+=['','- [Uncapped quality figure](reader_quality_cap0.png) · [Uncapped Pareto figure](reader_pareto_cap0.png).','',
          '## H16 width supplement','',
          '- H16 AMPM/SPP improves IPC over its own native in 8/8 measured pairs; this is not an additional training seed or a replacement for the four-route H8 comparison.',
          '- Only capped ROI1 SPP H16 improves both IPC and DRAM versus its own native. The other seven H16 pairs exchange higher IPC for more DRAM.',
          '- Simple templates reproduce nearly all measured H16 IPC gain. These runs do not establish a need for a learned payload or recurrent state. [Template pairs](paired_comparisons.csv) · [mechanism findings](FINDINGS.md).','',
          '| Policy | ROI | Uncapped IPC gain vs native | Capped IPC gain vs native |',
          '| --- | --- | --- | --- |']
    for pol in ('ampm','spp'):
        for roi in (0,1):out.append(f"| {NAMES[pol]} H16 | {roi} | {gain(pair(pol,16,roi,0)['ipc_gain_pct'])} | {gain(pair(pol,16,roi,16144)['ipc_gain_pct'])} |")
    out+=['','- [H16 IPC and accuracy summary](reader_h16.png) · [H8+H16 Pareto supplement, capped](reader_pareto_h16_cap16144.png) · [uncapped](reader_pareto_h16_cap0.png).','',
          '## Metric definitions and limits','',
          '- **IPC / performance:** actual retired ROI instructions ÷ actual ROI cycles. Pair NN with its own native in the same ROI and budget. Speedup versus NoPF is a separate CSV field.',
          '- **Reader CSV availability:** NoPF has no prefetch quality, so its reader-level availability flags are false and its ratios are NA. These are presentation flags, not replacements for source tracking flags.',
          '- **Accuracy:** LOAD-useful ÷ enqueued requests for the fully tracked intent cohort. Includes ROI-end pending/resident-unused in the denominator; merges are not new enqueues.',
          '- **Coverage proxy (C):** LOAD-useful cohort requests ÷ same-ROI NoPF LOAD misses at that level. It is not causal miss elimination. **Net miss reduction** separately uses (NoPF misses − candidate misses) ÷ NoPF misses.',
          '- **Timeliness:** timely LOAD-useful ÷ (timely + late LOAD-useful). Zero useful denominator gives NA, not 0%. RFO usefulness is excluded from these quality metrics.',
          '- **Actual requests/M:** allowed cache calls ÷ actual retired instructions × 1M; includes later cache acceptance, merge or rejection. This is not generated proposals or queue insertions.',
          '- **DRAM:** actual completed read + write controller bus bytes, all traffic; MiB = bytes ÷ 2²⁰. Frozen resident NN weight fetches have no modeled DRAM traffic. Requests × 64 is not used.',
          '- **Common protocol:** validation 20–25M; ROI0 25–50M and ROI1 75–100M; each test has 5M cache/state warmup. ROI0 was previously inspected. Page/fill rules remain those of the retained live protocol.',
          '- **Fairness limits:** Stride uses per-PC state and a different training budget from global-state Streamer/AMPM/SPP. Native SPP uses prefetch-fill FILTER feedback; its student observes all raw L2 fill callbacks. These are own-teacher route comparisons, not a matched learning-algorithm ranking.',
          '- **Cost limits:** MAC0 IPC gives no compute/area/energy Pareto result. [Actual fields and partial storage](storage_cost.csv) · [full metric contract](metric_definitions.json) · [method limitations](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).','',
          '## Provenance and complete evidence','',
          '- **Main H8/native/NoPF:** 34 selected completed test rows reused from `four602_frozen_live_20261017_v2`. **H16:** 8 selected completed test rows from `four602_width_templates_20261017_v1`. No new simulations or training were run to produce this reader view.',
          '- The complete source bundle has 64 reused rows plus 24 later runs (including validation and templates). Reader selection is not an additional experiment or replication.',
          '- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/` (later runs); `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/` (H8/native/NoPF).',
          '- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Models/builds/logs retain their original versions; large raw data is not on GitHub.',
          '- **Secondary reading:** [old replay versus live](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [full-ROI action distributions](action_histograms_full_roi.csv) · [stage curves](phase_progress.csv) · [paper story](PAPER_STORY.md) · [archive manifest](archive_manifest.json).',
          '- **Rebuild only this view:** run `python3 formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/build_reader_results.py` from the repository root with matplotlib available. Source run CSVs remain unchanged.','']
    (OUT/'README.md').write_text('\n'.join(out))


def plot(rows):
    # Keep every generated plotting cache inside the project.
    os.environ['MPLCONFIGDIR'] = str(ROOT/'.runtime/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=dict(stride='#0072B2',streamer='#009E73',ampm='#D55E00',spp='#CC79A7',none='#333333')
    def save(fig,name):
        for ext in ('png','svg'):
            destination = OUT/(name+'.'+ext)
            fig.savefig(destination,dpi=160,bbox_inches='tight')
            if ext == 'svg':
                destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
        plt.close(fig)
    def get(pol,arm,h,roi,cap):return next(r for r in rows if (r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget'])==(pol,arm,h,roi,cap))
    for cap in (16144,0):
        fig,axes=plt.subplots(2,4,figsize=(15,7.7))
        for roi in (0,1):
            for col,(metric,label) in enumerate((('ipc','IPC'),('l2_accuracy','L2 accuracy (%)'),('l2_coverage_proxy','L2 coverage proxy (%)'),('l2_timeliness','L2 timeliness (%)'))):
                ax=axes[roi,col]
                for i,pol in enumerate(POLICIES):
                    for offset,arm,h,color in ((-.18,'native',0,'#999999'),(.18,'nn',8,'#0072B2')):
                        value=get(pol,arm,h,roi,cap)[metric]
                        if value is None:
                            ax.text(i+offset,1,'NA',ha='center',va='bottom',fontsize=8,color='#555555')
                        else:
                            y=value if metric=='ipc' else value*100
                            ax.bar(i+offset,y,.33,color=color,label=arm if i==0 else None)
                            ax.text(i+offset,y,f'{y:.3f}' if metric=='ipc' else '<0.01' if 0 < y < .01 else f'{y:.1f}',ha='center',va='bottom',fontsize=8,rotation=90)
                ax.set_xticks(range(4),['Stride','Streamer','AMPM','SPP'],rotation=20,ha='right');ax.set_title(f'ROI{roi} · {label}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                ax.set_ylim(0,.50 if metric=='ipc' else 125)
        fig.legend(handles=[Line2D([0],[0],color='#999999',lw=8,label='Native'),Line2D([0],[0],color='#0072B2',lw=8,label='H8 NN')],loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False,fontsize=10)
        fig.suptitle(f'Four prefetchers: native vs H8 NN · {budget(cap)} · complete ROIs · seed7 · MAC0',fontsize=14)
        fig.text(.035,.028,'C = LOAD-useful cohort requests / matched NoPF misses; not net miss elimination. Accuracy uses enqueued requests.',fontsize=10)
        fig.text(.035,.008,'SPP H8 mainly targets LLC. Its L2 ROI0 cohort is empty; ROI1 is tiny. See the separate LLC table before interpreting quality.',fontsize=10)
        fig.tight_layout(rect=(0,.065,1,.93));save(fig,f'reader_quality_cap{cap}')
        for h16 in (False,True):
            fig,axes=plt.subplots(2,2,figsize=(12.5,8.8))
            for roi in (0,1):
                selected=[r for r in rows if r['roi']==roi and r['hidden_size']<=(16 if h16 else 8) and (r['budget']==cap or r['arm']=='nopf')]
                for col,cost in enumerate(('requests_per_million_instructions','dram_bus_bytes')):
                    ax=axes[roi,col];scale=1 if col==0 else 2**20
                    front=[r for r in selected if not any(dominates(s,r,cost) for s in selected)]
                    sorted_front=sorted(front,key=lambda r:r[cost]);ax.plot([r[cost]/scale for r in sorted_front],[r['ipc'] for r in sorted_front],color='#555555',ls='--',lw=1,zorder=1)
                    for r in selected:
                        marker='X' if r['arm']=='nopf' else 'o' if r['arm']=='native' else 'D' if r['hidden_size']==16 else '^'
                        ax.scatter(r[cost]/scale,r['ipc'],c=colors[r['policy']],marker=marker,s=74,edgecolors='black' if r in front else 'white',linewidths=1.2 if r in front else .5,zorder=3)
                    ax.set_xlabel('Actual allowed requests / M instructions' if col==0 else 'Completed DRAM bus MiB');ax.set_ylabel('IPC ↑');ax.set_title(f'ROI{roi} · lower cost ←');ax.grid(alpha=.18)
            handles=[Line2D([0],[0],color=colors[p],marker='X' if p=='none' else 'o',ls='',label=NAMES[p]) for p in (*POLICIES,'none')]
            handles += [Line2D([0],[0],color='#555555',marker=m,ls='',label=l) for m,l in (('o','Native'),('^','H8 NN'))]
            if h16:handles.append(Line2D([0],[0],color='#555555',marker='D',ls='',label='H16 NN'))
            fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.035))
            fig.suptitle(f'IPC–traffic Pareto view · {budget(cap)} · '+('H8 + H16 supplement' if h16 else 'four native/H8 routes'),fontsize=14)
            fig.text(.035,.02,'Black outline / dashed line: nondominated within each panel; NoPF is the same reused reference. Seed7, MAC0 only.',fontsize=9)
            fig.text(.035,.003,'Actual requests include subsequent cache merges/rejections. DRAM is actual read + write bus traffic; compute/area are not axes.',fontsize=9)
            fig.tight_layout(rect=(0,.13,1,.96));save(fig,f"reader_pareto_{'h16_' if h16 else ''}cap{cap}")
    fig,axes=plt.subplots(2,3,figsize=(13,7.8))
    for roi in (0,1):
        labels=[];selected=[]
        for p in ('ampm','spp'):
            for arm,h in (('native',0),('nn',16)):
                labels.append(f"{NAMES[p]}\n{'Native' if arm=='native' else 'H16 NN'}")
                selected.append(get(p,arm,h,roi,16144))
        for col,(metric,label) in enumerate((('ipc','IPC'),('l2_accuracy','L2 accuracy (%)'),('llc_accuracy','LLC accuracy (%)'))):
            ax=axes[roi,col]
            for x,r in enumerate(selected):
                value=r[metric]
                if value is None:ax.text(x,1,'NA',ha='center',fontsize=9);continue
                y=value if metric=='ipc' else value*100
                ax.bar(x,y,color='#999999' if r['arm']=='native' else '#D55E00');ax.text(x,y,f'{y:.4f}' if metric=='ipc' else f'{y:.1f}',ha='center',va='bottom',fontsize=9)
            ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(f'ROI{roi} · {label}');ax.set_ylim(0,.5 if metric=='ipc' else 110);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Separate H16 supplement · cap 16,144/M · seed7 · MAC0',fontsize=14)
    fig.text(.035,.015,'Not a replacement for the four-route H8 comparison. Both-level coverage and timeliness, uncapped values and traffic remain in reader_metrics.csv.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));save(fig,'reader_h16')


def main():
    rows=load();pairs=paired(rows);fronts=frontiers(rows)
    dump('reader_metrics.csv',rows);dump('reader_pairs.csv',pairs);dump('reader_pareto.csv',fronts)
    document(rows,pairs,fronts);plot(rows)
    print('Reader outputs: 42 complete-run rows; 24 native/NN pairs; 160 panel memberships; 7 figures (PNG + SVG). No simulation or training.')

if __name__=='__main__':main()
