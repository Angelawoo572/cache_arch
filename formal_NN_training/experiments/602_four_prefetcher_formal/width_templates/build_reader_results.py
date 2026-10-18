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
    for candidate_set,max_h in (('H8_only',8),('H8_H16',16)):
        for roi in (0,1):
            for cap in (0,16144):
                selected=[r for r in rows if r['roi']==roi and r['hidden_size']<=max_h and (r['budget']==cap or r['arm']=='nopf')]
                assert len(selected)==(9 if max_h==8 else 11)
                for cost in ('requests_per_million_instructions','dram_bus_bytes'):
                    for r in selected:
                        result.append(dict(candidate_set=candidate_set,roi=roi,budget=cap,cost_axis=cost,result_id=r['result_id'],actual_run_budget=r['budget'],x_cost=r[cost],ipc=r['ipc'],is_pareto=not any(dominates(s,r,cost) for s in selected)))
    return result


def pct(v):
    if v is None:return 'NA'
    return '<0.01' if 0 < v*100 < .01 else f'{v*100:.1f}'
def number(v):return f'{v:,.0f}'
def gain(v):return f'{v:+.2f}%'
def budget(cap):return 'uncapped' if cap==0 else 'cap 16,144/M'


def document(rows,pairs,fronts):
    lookup={(r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget']):r for r in rows}
    def get(pol,arm,h,roi,cap):return lookup[(pol,arm,h,roi,cap)]
    def pair(pol,h,roi,cap):return next(p for p in pairs if (p['policy'],p['hidden_size'],p['roi'],p['budget'])==(pol,h,roi,cap))
    def width_rows(h,cap,compact=False,policies=None,rois=(0,1)):
        out=[]
        for pol in (policies or (POLICIES if h==8 else ('ampm','spp'))):
            for roi in rois:
                n=get(pol,'native',0,roi,cap);s=get(pol,'nn',h,roi,cap);p=pair(pol,h,roi,cap)
                quality=[f'{pct(n["l2_"+metric])} → {pct(s["l2_"+metric])}' for metric in ('accuracy','coverage_proxy','timeliness')]
                if compact:
                    out.append(f'| {NAMES[pol]} H{h} | {roi} | {gain(p["ipc_gain_pct"])} | '+ ' | '.join(quality)+' |')
                else:
                    out.append(f'| {roi} | {NAMES[pol]} | {n["ipc"]:.4f} → {s["ipc"]:.4f} | {gain(p["ipc_gain_pct"])} | '+' | '.join(quality)+' |')
        return out
    prefix='formal_study/results/602_width_templates/'
    root=['# 602 results: native prefetchers versus H8 and H16','',
          '- **Compare:** Stride, Streamer, AMPM and SPP with their own frozen NN students on IPC, accuracy, coverage, timeliness and traffic Pareto trade-offs.',
          '- **Available live results:** H8 for all four routes; H16 for AMPM and SPP. Stride/Streamer H16 has no completed result in this two-ROI/cap protocol. [Earlier Stride H16](formal_NN_training/experiments/602_lstm_stride_dynamic_inference/results/summary.csv) is retained separately under a different protocol.',
          '- **Capped IPC result:** H8 wins 6/8 native pairs; H16 AMPM/SPP wins 4/4. Seed7, complete ROIs, frozen weights, live state and zero-cost inference (**MAC0**). [Exact run metrics]('+prefix+'reader_metrics.csv).','',
          '## H8 and H16: quality and IPC','',
          '- **Common ceiling: 16,144 allowed requests/M.** Each quality cell is **native → NN**, in percent. IPC change is relative to the matching native.',
          '- **C = coverage proxy:** useful LOAD requests ÷ matched NoPF LOAD misses; it is not unique misses eliminated. [Definitions and net miss reduction]('+prefix+'README.md#definitions-and-scope).','',
          '| Model | ROI | NN IPC change | L2 accuracy | L2 C | L2 timeliness |',
          '| --- | --- | --- | --- | --- | --- |']
    for pol in POLICIES:
        for roi in (0,1):
            for h in (8,16):
                if h==8 or pol in ('ampm','spp'):
                    root+=width_rows(h,16144,True,(pol,),(roi,))
    root+=['',
           '- **H8:** IPC wins 6/8 capped pairs and 9/16 across both budgets. **H16:** wins 4/4 capped pairs and 8/8 across both budgets, on AMPM/SPP only. [All pairs, including losses]('+prefix+'reader_pairs.csv).',
           '- **SPP differs by width:** H8 mostly targets LLC; its capped L2 cohort is empty in ROI0 and only 14 enqueues in ROI1 (13 useful). H16 improves IPC despite lower L2 accuracy than native in all four cases. Quality and IPC must be assessed together.',
           '- **Both budgets:** [full H8/H16 tables]('+prefix+'README.md) include uncapped results, actual IPC, counts and traffic. Values rounded to 100.0% are not always exactly 100%.','',
           '![Native, H8 and H16: IPC and L2 quality under the common ceiling]('+prefix+'reader_quality_cap16144.svg)','',
           '## SPP: keep L2 and LLC quality separate','',
           '- Higher LLC accuracy in H8 coexists with lower IPC; H16 shifts usefulness toward L2 and improves IPC. These observations do not isolate a causal mechanism.',
           '- LLC accuracy, coverage proxy and timeliness below use fully tracked first-enqueue-intent cohorts. A proxy above 100% is not more than 100% of unique misses eliminated. [LLC counts and both budgets]('+prefix+'README.md#spp-llc-quality).','',
           '![SPP native, H8 and H16 LLC quality under the common ceiling]('+prefix+'reader_spp_llc_cap16144.svg)','',
           '## Pareto: IPC versus actual requests and DRAM','',
           '- **Shown together:** four natives, four H8 students, two H16 students and NoPF. Each ROI/cap has its own frontier. Higher IPC and lower actual traffic are preferred; compute, area and energy are not measured axes.',
           '- **H8 capped:** all eight NN cases use more DRAM than their own native. Native SPP dominates H8 NN on IPC/DRAM; the other routes trade more DRAM for higher IPC.',
           '- **H16:** all eight pairs dominate their own native on IPC/actual requests. On IPC/DRAM, only capped ROI1 SPP dominates native; the other seven pairs use more DRAM. A common ceiling does not equalize actual traffic. [Exact pair classifications]('+prefix+'reader_pairs.csv).','',
           '![Native, H8 and H16 IPC–traffic Pareto frontiers under the common ceiling]('+prefix+'reader_pareto_h16_cap16144.svg)','',
           '## Conclusions and limits','',
           '- **Supported here:** mixed H8 outcomes; positive AMPM/SPP H16 IPC outcomes; different request/DRAM trade-offs. Better accuracy alone does not establish better performance.',
           '- **Not supported:** a universal NN advantage or a necessary learned payload/state. Validation-frozen templates reproduce nearly all measured H16 IPC gain. [Mechanism evidence]('+prefix+'FINDINGS.md).',
           '- **Still open:** more training seeds/workloads and finite-cost deployment. Route training/state budgets differ; native/student SPP fill feedback differs. ROI0 was previously inspected. [Method limits](formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md) · [cost assumptions]('+prefix+'storage_cost.csv).','',
           '## Evidence and locations','',
           '- **Protocol:** validation 20–25M; ROI0 25–50M; ROI1 75–100M; 5M cache/state warmup each. Old replay and earlier finite-engine studies remain separate. [Replay alignment](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [earlier gate/finite results](docs/602_gate_mechanism/REPORT.md).',
           '- **Completed evidence:** original 64-row H8 study + 24 later H16/template runs. This reader view selects 34 reused H8/native/NoPF test rows and 8 H16 test rows; no new training or simulation. [Labeled source]('+prefix+'per_run.csv) · [paper story]('+prefix+'PAPER_STORY.md).',
           '- **Shared figures/CSVs:** [formal_study/results/602_width_templates]('+prefix+'README.md). Main figures use SVG; PNG exports remain available.',
           '- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`; original H8/native raw: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.',
           '- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; original H8/native raw: `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Large logs, models/builds and traces stay on the machines.',
           '- **Branches/recovery:** `main` is the shared reading branch; experiment/archive branches are preserved. [Layout and preservation](docs/602_width_templates/LAYOUT_AND_RECOVERY.md#repository-and-documentation-preservation).','']
    (ROOT/'RESULTS_INDEX.md').write_text('\n'.join(root))
    out=['# 602: H8 and H16 versus native prefetchers','',
         '- **Read:** [project overview](../../../RESULTS_INDEX.md) · [run metrics](reader_metrics.csv) · [native/NN pairs](reader_pairs.csv) · [Pareto flags](reader_pareto.csv) · [reader audit](reader_audit.json).',
         '- **Coverage of this matrix:** H8 Stride/Streamer/AMPM/SPP; H16 AMPM/SPP. Missing Stride/Streamer H16 results are **not measured under this protocol**, not zero or failed results.',
         '- **Scope:** original seed7 weights; complete two-ROI live inference; frozen weights, updating state; MAC0. Quality cells are **native → NN (%)**; C is the useful/NoPF-miss proxy.','']
    for cap,title in ((16144,'Capped results'),(0,'Uncapped results')):
        out+=['## '+title,'',f'- **Budget:** {budget(cap)}. H8 IPC wins {"6/8" if cap else "3/8"}; H16 AMPM/SPP wins 4/4. Each NN is paired with its own native.','']
        for h in (8,16):
            out+=['### H'+str(h),'',
                  '| ROI | Policy | IPC native → NN | NN IPC gain | L2 accuracy | L2 C | L2 timeliness |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
            out+=width_rows(h,cap)+['']
        out += ['- SPP H8 L2 enqueues: **0 / '+('14' if cap else '66')+'** (ROI0 / ROI1). ROI1 useful: '+('13' if cap else '65')+'. Its L2-only quality cannot describe its predominantly LLC-targeted policy.',
                '- H16 SPP has lower L2 accuracy but higher IPC than native in both ROIs; a metric gain/loss is not a performance mechanism by itself.','',
                f'![Native versus H8/H16 IPC and L2 quality, {budget(cap)}](reader_quality_cap{cap}.svg)','',
                '| Model | ROI | Actual requests/M native → NN | DRAM MiB native → NN | DRAM change | Net L2 miss reduction % native → NN |',
                '| --- | --- | --- | --- | --- | --- |']
        for h in (8,16):
            for pol in (POLICIES if h==8 else ('ampm','spp')):
                for roi in (0,1):
                    n=get(pol,'native',0,roi,cap);s=get(pol,'nn',h,roi,cap);p=pair(pol,h,roi,cap)
                    out.append(f'| {NAMES[pol]} H{h} | {roi} | {n["requests_per_million_instructions"]:,.2f} → {s["requests_per_million_instructions"]:,.2f} | {n["dram_bus_bytes"]/2**20:.3f} → {s["dram_bus_bytes"]/2**20:.3f} | {gain(p["dram_change_pct"])} | {pct(n["net_l2_load_miss_reduction"])} → {pct(s["net_l2_load_miss_reduction"])} |')
        out+=['']
    out+=['## SPP LLC quality','',
          '- Keep LLC and L2 first-enqueue intent separate. They are not final fill destinations; do not add their ratios.',
          '- Empty useful/enqueued denominators give NA. Useful/NoPF-miss ratios can exceed 100%; they are not unique baseline misses eliminated.','',
          '| Budget | ROI | Arm | LLC enqueued | Accuracy % | C % | Timeliness % |',
          '| --- | --- | --- | --- | --- | --- | --- |']
    for cap in (16144,0):
        for roi in (0,1):
            for arm,h in (('native',0),('nn',8),('nn',16)):
                r=get('spp',arm,h,roi,cap);label='Native' if arm=='native' else f'H{h} NN'
                out.append(f'| {budget(cap)} | {roi} | {label} | {number(r["llc_enqueued"])} | {pct(r["llc_accuracy"])} | {pct(r["llc_coverage_proxy"])} | {pct(r["llc_timeliness"])} |')
    out+=['','![SPP native/H8/H16 LLC accuracy, coverage proxy and timeliness, capped](reader_spp_llc_cap16144.svg)',
          '- [Uncapped LLC quality figure](reader_spp_llc_cap0.svg). AMPM has no LLC-intent enqueues: accuracy/timeliness are NA and the coverage proxy is 0.','',
          '## Pareto frontiers','',
          '- **Axes:** maximize IPC; minimize actual requests/M or completed DRAM bytes. Strict nondominance uses unrounded values. Ties remain tied; no significance or global-optimum claim is made.',
          '- **Combined candidate set:** four natives + four H8 NNs + two H16 NNs + NoPF, separately per ROI and budget. H8-only membership is also retained in `reader_pareto.csv`.',
          '- **H8 capped:** all eight NN cases increase DRAM. Native SPP dominates H8 NN on IPC/DRAM; Stride/Streamer/AMPM trade higher IPC for more DRAM.',
          '- **H16:** all eight pairs dominate native on IPC/actual requests; only capped ROI1 SPP dominates native on IPC/DRAM. Native compute/area and finite NN service costs are outside these plots.',
          '- NoPF is the same reused reference in both budget panels, not a new capped run. Requests below the final ceiling do not prove that cumulative credit never limited earlier output.','',
          '![Native, H8 and H16 Pareto comparison, capped](reader_pareto_h16_cap16144.svg)',
          '- [Uncapped H8/H16 Pareto](reader_pareto_h16_cap0.svg) · [H8-only capped](reader_pareto_cap16144.svg) · [H8-only uncapped](reader_pareto_cap0.svg).','',
          '## Definitions and scope','',
          '- **IPC:** actual retired instructions ÷ cycles. **Net miss reduction:** (NoPF LOAD misses − candidate misses) ÷ NoPF misses; distinct from C.',
          '- **Accuracy:** LOAD-useful ÷ enqueued requests in a fully tracked intent cohort, including ROI-end pending/resident-unused. Cache merges are not new enqueues.',
          '- **Coverage proxy (C):** LOAD-useful cohort requests ÷ same-ROI NoPF LOAD misses at that level. It is not causal miss elimination.',
          '- **Timeliness:** timely LOAD-useful ÷ (timely + late LOAD-useful). Zero denominator gives NA. RFO usefulness is excluded.',
          '- **Requests/M:** credit-allowed cache calls ÷ actual retired instructions × 1M, including later cache merges/rejections. **DRAM:** actual completed read + write controller bus bytes; MiB = bytes ÷ 2²⁰. Requests × 64 is never used.',
          '- **NoPF:** prefetch quality is not applicable; reader availability is false and ratios are NA, independently of source empty-cohort tracking flags.',
          '- **Protocol:** validation 20–25M; ROI0 25–50M and ROI1 75–100M; 5M cache/state warmup each. ROI0 was previously inspected. Different route training/state budgets and native/student SPP fill inputs limit claims.',
          '- **Cost:** synchronous MAC0 IPC; resident NN weights with no modeled weight DRAM fetches. No hardware compute/area/energy Pareto is established. [Storage fields](storage_cost.csv) · [full contract](metric_definitions.json) · [method limits](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).','',
          '## Evidence and recovery','',
          '- **42 selected test rows:** 34 H8/native/NoPF references + 8 H16 runs. The source bundle retains 64 reused rows + 24 later validation/test/template runs. No new simulation or training is needed for this view. [All labeled rows](per_run.csv).',
          '- **Mechanisms/history:** [template findings](FINDINGS.md) · [paper story](PAPER_STORY.md) · [replay alignment](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [full-ROI actions](action_histograms_full_roi.csv) · [stages](phase_progress.csv).',
          '- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`; `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.',
          '- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. [Layout and preservation](../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).',
          '- **Rebuild figures/tables only:** `python3 formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/build_reader_results.py` with matplotlib. [Independent checker](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/verify_reader_results.py). All runtime files stay inside cache.','']
    (OUT/'README.md').write_text('\n'.join(out))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    colors=dict(stride='#0072B2',streamer='#009E73',ampm='#D55E00',spp='#CC79A7',none='#333333')
    arm_colors=('#999999','#0072B2','#D55E00')
    arms=(('native',0,'Native'),('nn',8,'H8 NN'),('nn',16,'H16 NN'))
    def save(fig,name):
        for ext in ('svg','png'):
            destination=OUT/(name+'.'+ext)
            fig.savefig(destination,dpi=160,bbox_inches='tight')
            if ext=='svg':destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
        plt.close(fig)
    def get(pol,arm,h,roi,cap):return next((r for r in rows if (r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget'])==(pol,arm,h,roi,cap)),None)
    def annotate(ax,x,y,label):ax.text(x,y,label,ha='center',va='bottom',fontsize=8,rotation=90)
    for cap in (16144,0):
        fig,axes=plt.subplots(2,4,figsize=(15,7.7))
        for roi in (0,1):
            for col,(metric,label) in enumerate((('ipc','IPC'),('l2_accuracy','L2 accuracy (%)'),('l2_coverage_proxy','L2 coverage proxy (%)'),('l2_timeliness','L2 timeliness (%)'))):
                ax=axes[roi,col]
                for i,pol in enumerate(POLICIES):
                    for j,(arm,h,_) in enumerate(arms):
                        x=i+(j-1)*.24;r=get(pol,arm,h,roi,cap)
                        if r is None:
                            ax.text(x,1 if metric!='ipc' else .006,'n/m',ha='center',va='bottom',fontsize=7,rotation=90,color='#777777');continue
                        value=r[metric]
                        if value is None:
                            ax.text(x,1,'NA',ha='center',va='bottom',fontsize=8,color='#555555');continue
                        y=value if metric=='ipc' else value*100
                        ax.bar(x,y,.22,color=arm_colors[j])
                        annotate(ax,x,y,f'{y:.3f}' if metric=='ipc' else '<0.01' if 0<y<.01 else f'{y:.1f}')
                ax.set_xticks(range(4),[NAMES[p] for p in POLICIES],rotation=20,ha='right');ax.set_title(f'ROI{roi} · {label}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,.5 if metric=='ipc' else 125)
        fig.legend(handles=[Line2D([0],[0],color=c,lw=8,label=a[2]) for a,c in zip(arms,arm_colors)],loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
        fig.suptitle(f'Native vs H8 and H16 · {budget(cap)} · complete ROIs · seed7 · MAC0',fontsize=14)
        fig.text(.035,.028,'n/m: no compatible current-protocol H16 run. C = useful LOAD requests / matched NoPF misses; not net miss elimination.',fontsize=9)
        fig.text(.035,.008,'SPP H8 mainly targets LLC: L2 ROI0 is empty; ROI1 is tiny. Compare the separate LLC-intent figure as well.',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.93));save(fig,f'reader_quality_cap{cap}')
        fig,axes=plt.subplots(2,3,figsize=(12,6.8))
        for roi in (0,1):
            for col,(metric,label) in enumerate((('accuracy','LLC accuracy (%)'),('coverage_proxy','LLC coverage proxy (%)'),('timeliness','LLC timeliness (%)'))):
                ax=axes[roi,col]
                for j,(arm,h,name) in enumerate(arms):
                    r=get('spp',arm,h,roi,cap);v=r['llc_'+metric]
                    if v is None:ax.text(j,1,'NA',ha='center');continue
                    y=v*100;ax.bar(j,y,color=arm_colors[j],width=.65);ax.text(j,y,f'{y:.2f}',ha='center',va='bottom',fontsize=9)
                ax.set_xticks(range(3),[a[2] for a in arms]);ax.set_title(f'ROI{roi} · {label}');ax.set_ylim(0,130 if metric=='coverage_proxy' else 115);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(f'SPP: native vs H8 and H16 · LLC-intent cohorts · {budget(cap)} · MAC0',fontsize=13)
        fig.text(.035,.012,'Accuracy = useful/enqueued; C = useful/NoPF LLC misses; timeliness = timely/useful. Separate from L2; not causal miss elimination.',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.95));save(fig,f'reader_spp_llc_cap{cap}')
        for all_widths in (False,True):
            fig,axes=plt.subplots(2,2,figsize=(12.5,8.8))
            for roi in (0,1):
                selected=[r for r in rows if r['roi']==roi and r['hidden_size']<=(16 if all_widths else 8) and (r['budget']==cap or r['arm']=='nopf')]
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
            if all_widths:handles.append(Line2D([0],[0],color='#555555',marker='D',ls='',label='H16 NN'))
            fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.035))
            fig.suptitle(f'IPC–traffic Pareto · {budget(cap)} · '+('native, H8 and H16' if all_widths else 'native and H8 only'),fontsize=14)
            fig.text(.035,.02,'Black outline / dashed line: nondominated within each panel. NoPF is reused. Seed7, MAC0; missing H16 routes are not zero.',fontsize=9)
            fig.text(.035,.003,'Requests include subsequent cache merges/rejections. DRAM is actual read + write bus traffic; compute/area are not axes.',fontsize=9)
            fig.tight_layout(rect=(0,.13,1,.96));save(fig,f"reader_pareto_{'h16_' if all_widths else ''}cap{cap}")


def main():
    rows=load();pairs=paired(rows);fronts=frontiers(rows)
    dump('reader_metrics.csv',rows);dump('reader_pairs.csv',pairs);dump('reader_pareto.csv',fronts)
    document(rows,pairs,fronts);plot(rows)
    print('Reader outputs: H8/H16, 42 run rows, 24 native/NN pairs, 160 panel memberships, 8 SVG + PNG figure pairs. No simulation or training.')

if __name__=='__main__':main()
