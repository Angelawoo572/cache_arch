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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
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
COMPLETION = 'four602_all_h16_20261017_v1'


def load():
    with (OUT / 'per_run.csv').open(newline='') as stream:
        source = list(csv.DictReader(stream))
    with (OUT/'per_run_h16_completion.csv').open(newline='') as stream:
        source += list(csv.DictReader(stream))
    rows = []
    for s in source:
        if s['stage'] != 'test' or s['mode'] != 'full':
            continue
        arm = 'nopf' if s['policy'] == 'none' and s['method'] == 'none' else 'native' if s['method'] == 'native' else 'nn' if s['method'] in ('student', 'stride_student') else ''
        if not arm:
            continue
        h = int(s.get('hidden_size') or s.get('raw__h') or 0) if arm == 'nn' else 0
        expected = COMPLETION if h == 16 and s['policy'] in ('stride','streamer') else SUPPLEMENT if h == 16 else REFERENCE
        assert s['protocol'] == expected, s['result_id']
        r = {k:s[k] for k in ('result_id', 'run_id', 'protocol', 'evidence_origin', 'source_raw', 'source_run', 'policy')}
        r.update(arm=arm, hidden_size=h)
        for k in ('roi', 'budget', 'seed', 'instructions', 'cycles', 'l2_load_misses'):
            r[k] = int(s[k]) if s[k] else ''
        r.update(ipc=float(s['ipc']), ipc_speedup_vs_NoPF=float(s['ipc_speedup_vs_NoPF']),
                 requests_per_million_instructions=float(s['requests_per_million_instructions']),
                 allowed_requests=int(s['raw__request_gate_allowed']), dram_bus_bytes=int(s['raw__dram_total_bus_bytes']))
        for k in ('inference_macs','inference_nonlinears','completed_updates','model_weight_bytes','recurrent_hc_bytes'):
            r[k] = int(s[k]) if s.get(k) else None
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
    assert len(rows) == 50
    assert sum(r['arm']=='nopf' for r in rows) == 2
    assert sum(r['arm']=='nn' and r['hidden_size']==16 for r in rows) == 16
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
    assert len(pairs)==32
    return pairs


def frontiers(rows):
    result=[]
    for candidate_set,max_h in (('H8_only',8),('H8_H16',16)):
        for roi in (0,1):
            for cap in (0,16144):
                selected=[r for r in rows if r['roi']==roi and r['hidden_size']<=max_h and (r['budget']==cap or r['arm']=='nopf')]
                assert len(selected)==(9 if max_h==8 else 13)
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
        for pol in (policies or POLICIES):
            for roi in rois:
                n=get(pol,'native',0,roi,cap);s=get(pol,'nn',h,roi,cap);p=pair(pol,h,roi,cap)
                quality=[f'{pct(n["l2_"+metric])} → {pct(s["l2_"+metric])}' for metric in ('accuracy','coverage_proxy','timeliness')]
                if compact:
                    out.append(f'| {NAMES[pol]} H{h} | {roi} | {gain(p["ipc_gain_pct"])} | '+ ' | '.join(quality)+' |')
                else:
                    out.append(f'| {roi} | {NAMES[pol]} | {n["ipc"]:.4f} → {s["ipc"]:.4f} | {gain(p["ipc_gain_pct"])} | '+' | '.join(quality)+' |')
        return out
    def count_wins(h,cap=None):
        values=[p for p in pairs if p['hidden_size']==h and (cap is None or p['budget']==cap)]
        return f"{sum(p['ipc_gain_pct']>0 for p in values)}/{len(values)}"
    def dominance(h,axis):
        values=[p for p in pairs if p['hidden_size']==h]
        return f"{sum(p['paired_'+axis+'_relation']=='nn_dominates' for p in values)}/{len(values)}"
    h16_traffic=f'H16 dominates its own native in {dominance(16,"requests")} IPC/request pairs and {dominance(16,"dram")} IPC/DRAM pairs.'
    prefix='formal_study/results/602_width_templates/'
    root=['# 602 results: native prefetchers versus H8 and H16','',
          '- **Compare:** Stride, Streamer, AMPM and SPP with their own frozen NN students on IPC, accuracy, coverage, timeliness and traffic Pareto trade-offs.',
          '- **Completed models:** H8 and H16 for all four routes. Compatible retained Stride/Streamer H16 checkpoints were recovered and verified; no retraining was needed. Older replay and live protocols remain separate.',
          f'- **Capped IPC result:** H8 wins {count_wins(8,16144)} native pairs; H16 wins {count_wins(16,16144)}. Seed7, complete ROIs, frozen weights, live state and zero-cost inference (**MAC0**). [Exact run metrics]('+prefix+'reader_metrics.csv).','',
          '## H8 and H16: quality and IPC','',
          '- **Common ceiling: 16,144 allowed requests/M.** Each quality cell is **native → NN**, in percent. IPC change is relative to the matching native.',
          '- **C = coverage proxy:** useful LOAD requests ÷ matched NoPF LOAD misses; it is not unique misses eliminated. [Definitions and net miss reduction]('+prefix+'README.md#definitions-and-scope).','',
          '| Model | ROI | NN IPC change | L2 accuracy | L2 C | L2 timeliness |',
          '| --- | --- | --- | --- | --- | --- |']
    for pol in POLICIES:
        for roi in (0,1):
            for h in (8,16):
                root+=width_rows(h,16144,True,(pol,),(roi,))
    root+=['',
           f'- **H8:** IPC wins {count_wins(8,16144)} capped pairs and {count_wins(8)} across both budgets. **H16:** wins {count_wins(16,16144)} capped pairs and {count_wins(16)} across both budgets. [All pairs, including losses]('+prefix+'reader_pairs.csv).',
           '- **SPP differs by width:** H8 mostly targets LLC; its capped L2 cohort is empty in ROI0 and only 14 enqueues in ROI1 (13 useful). H16 improves IPC despite lower L2 accuracy than native in all four cases. Quality and IPC must be assessed together.',
           '- **Both budgets:** [full H8/H16 tables]('+prefix+'README.md) include uncapped results, actual IPC, counts and traffic. Values rounded to 100.0% are not always exactly 100%.','',
           '![Native, H8 and H16: IPC and L2 quality under the common ceiling]('+prefix+'reader_quality_cap16144.svg)','',
           '## SPP: keep L2 and LLC quality separate','',
           '- Higher LLC accuracy in H8 coexists with lower IPC; H16 shifts usefulness toward L2 and improves IPC. These observations do not isolate a causal mechanism.',
           '- LLC accuracy, coverage proxy and timeliness below use fully tracked first-enqueue-intent cohorts. A proxy above 100% is not more than 100% of unique misses eliminated. [LLC counts and both budgets]('+prefix+'README.md#spp-llc-quality).','',
           '![SPP native, H8 and H16 LLC quality under the common ceiling]('+prefix+'reader_spp_llc_cap16144.svg)','',
           '## Pareto: IPC versus actual requests and DRAM','',
           '- **Shown together:** four natives, four H8 students, four H16 students and NoPF. Each ROI/cap has its own frontier. Higher IPC and lower actual traffic are preferred; compute, area and energy are not measured axes.',
           '- **H8 capped:** all eight NN cases use more DRAM than their own native. Native SPP dominates H8 NN on IPC/DRAM; the other routes trade more DRAM for higher IPC.',
           '- **H16:** '+h16_traffic+' A common request ceiling does not equalize actual traffic. [Exact pair classifications]('+prefix+'reader_pairs.csv).','',
           '![Native, H8 and H16 IPC–traffic Pareto frontiers under the common ceiling]('+prefix+'reader_pareto_h16_cap16144.svg)','',
           '## Conclusions and limits','',
           '- **Supported here:** route- and width-dependent quality/IPC/traffic outcomes. Read each own-native pair; better accuracy alone does not establish better performance.',
           '- **Not established:** a universal NN advantage or a necessary learned payload/state. Full, fixed-payload, reset-state and NN-free controls constrain those claims. [Mechanism evidence]('+prefix+'FINDINGS.md).',
           '- **Still open:** more training seeds/workloads and finite-cost deployment. Route training/state budgets differ; native/student SPP fill feedback differs. ROI0 was previously inspected. [Method limits](formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md) · [cost assumptions]('+prefix+'README.md#inference-work-and-width).','',
           '## Evidence and locations','',
           '- **Protocol:** validation 20–25M; ROI0 25–50M; ROI1 75–100M; 5M cache/state warmup each. Old replay and earlier finite-engine studies remain separate. [Replay alignment](formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [earlier gate/finite results](docs/602_gate_mechanism/REPORT.md).',
           '- **Completed evidence:** 64 original H8 rows + 24 AMPM/SPP width/template rows + 30 H16-completion runs (2 validation, 28 full tests). The main view selects 50 test rows; reused references are not extra replications. [Earlier labeled rows]('+prefix+'per_run.csv) · [30 new rows]('+prefix+'per_run_h16_completion.csv).',
           '- **Size: parameters → bytes:** all eight models are tensor-verified. [Model size and inference state]('+prefix+'README.md#model-size-and-inference-state) separates FP32 weights, shared/per-PC state and implementation overhead.',
           '- **Shared figures/CSVs:** [formal_study/results/602_width_templates]('+prefix+'README.md). Main figures use SVG; PNG exports remain available.',
           '- **New complete raw, local:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; earlier width raw is its parent; original H8/native raw: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.',
           '- **New complete raw, Sacramento:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; earlier width raw is its parent; original H8/native raw: `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. Large logs, models/builds and traces stay on the machines.',
           '- **Branches/recovery:** `main` is the shared reading branch; experiment/archive branches are preserved. [Layout and preservation](docs/602_width_templates/LAYOUT_AND_RECOVERY.md#repository-and-documentation-preservation).','']
    (ROOT/'RESULTS_INDEX.md').write_text('\n'.join(root))
    out=['# 602: H8 and H16 versus native prefetchers','',
         '- **Read:** [project overview](../../../RESULTS_INDEX.md) · [run metrics](reader_metrics.csv) · [native/NN pairs](reader_pairs.csv) · [Pareto flags](reader_pareto.csv) · [reader audit](reader_audit.json).',
         '- **Completed widths:** H8 and H16 for Stride, Streamer, AMPM and SPP. All are frozen during inference. The same cache/ROI/page protocol and actual binary are used; each checkpoint retains its own training provenance.',
         '- **Scope:** original seed7 weights; complete two-ROI live inference; frozen weights, updating state; MAC0. Quality cells are **native → NN (%)**; C is the useful/NoPF-miss proxy.','']
    for cap,title in ((16144,'Capped results'),(0,'Uncapped results')):
        out+=['## '+title,'',f'- **Budget:** {budget(cap)}. H8 IPC wins {count_wins(8,cap)}; H16 wins {count_wins(16,cap)}. Each NN is paired with its own native.','']
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
            for pol in POLICIES:
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
          '- **Combined candidate set:** four natives + four H8 NNs + four H16 NNs + NoPF, separately per ROI and budget. H8-only membership is also retained in `reader_pareto.csv`.',
          '- **H8 capped:** all eight NN cases increase DRAM. Native SPP dominates H8 NN on IPC/DRAM; Stride/Streamer/AMPM trade higher IPC for more DRAM.',
          '- **H16:** '+h16_traffic+' Native compute/area and finite NN service costs are outside these plots.',
          '- NoPF is the same reused reference in both budget panels, not a new capped run. Requests below the final ceiling do not prove that cumulative credit never limited earlier output.','',
          '![Native, H8 and H16 Pareto comparison, capped](reader_pareto_h16_cap16144.svg)',
          '- [Uncapped H8/H16 Pareto](reader_pareto_h16_cap0.svg) · [H8-only capped](reader_pareto_cap16144.svg) · [H8-only uncapped](reader_pareto_cap0.svg).','',
          '## Actions and live progress','',
          '- **Full-ROI actions:** all 32 H8/H16 NN tests. K includes silent demand decisions; delta/fill count decoded slots before current-line/page/credit gates. Rare support is retained, not rounded away. [Exact action counts](action_plot_data_h16_completion.csv) · [per-slot support](slot_support_h16_completion.csv).',
          '![All four H8/H16 routes: complete-ROI K, delta and fill distributions, capped ROI1](actions_all_widths_roi1_cap16144.svg)',
          '- **Other conditions:** [ROI0 capped](actions_all_widths_roi0_cap16144.svg) · [ROI0 uncapped](actions_all_widths_roi0_cap0.svg) · [ROI1 uncapped](actions_all_widths_roi1_cap0.svg) · [slot/delta/fill, capped ROI1](slots_all_widths_roi1_cap16144.svg).',
          '- **Stages:** each phase is 5M measured instructions. [Capped ROI1 phase emit/K/fill](phase_actions_all_widths_roi1_cap16144.svg) · [all phase values](phase_plot_data_h16_completion.csv). SPP fill callbacks advance state but are excluded from demand-action denominators.',
          '![H16 live progress by measured instructions, demand callbacks and completed inference, capped ROI1](history_all_h16_roi1_cap16144.svg)',
          '- **History axes differ:** measured instructions, demand callbacks and completed inference are runtime progress, not training data or minimum history. SPP completions include demand and fill callbacks. Curves do not identify a minimum observation count by first baseline crossing.',
          '- **Other histories:** [ROI0 capped](history_all_h16_roi0_cap16144.svg) · [ROI0 uncapped](history_all_h16_roi0_cap0.svg) · [ROI1 uncapped](history_all_h16_roi1_cap0.svg). [Saved snapshots](startup_plot_data_h16_completion.csv) · [all figure/source links](plot_manifest_h16_completion.json).','',
          '## Inference work and width','',
          '- Measured MAC/nonlinear counts come from full completed ROIs. `ceil(MACs/4) / MAC0 cycles` is an aggregate necessary throughput condition, not simulated finite-cost IPC. Native compute is unmeasured, not zero.',
          '![Counted inference work across all eight models](reader_cost.svg)',
          '- [Width, IPC and actual traffic figures](reader_widths_roi1_cap16144.svg) · [complete cost points](reader_cost.csv).','']
    import runpy
    out += runpy.run_path(str(HERE/'model_size.py'))['reader_section']()+['']
    out += ['## Definitions and scope','',
          '- **IPC:** actual retired instructions ÷ cycles. **Net miss reduction:** (NoPF LOAD misses − candidate misses) ÷ NoPF misses; distinct from C.',
          '- **Accuracy:** LOAD-useful ÷ enqueued requests in a fully tracked intent cohort, including ROI-end pending/resident-unused. Cache merges are not new enqueues.',
          '- **Coverage proxy (C):** LOAD-useful cohort requests ÷ same-ROI NoPF LOAD misses at that level. It is not causal miss elimination.',
          '- **Timeliness:** timely LOAD-useful ÷ (timely + late LOAD-useful). Zero denominator gives NA. RFO usefulness is excluded.',
          '- **Requests/M:** credit-allowed cache calls ÷ actual retired instructions × 1M, including later cache merges/rejections. **DRAM:** actual completed read + write controller bus bytes; MiB = bytes ÷ 2²⁰. Requests × 64 is never used.',
          '- **NoPF:** prefetch quality is not applicable; reader availability is false and ratios are NA, independently of source empty-cohort tracking flags.',
          '- **Protocol:** validation 20–25M; ROI0 25–50M and ROI1 75–100M; 5M cache/state warmup each. ROI0 was previously inspected. Different route training/state budgets and native/student SPP fill inputs limit claims.',
          '- **Cost:** synchronous MAC0 IPC; resident NN weights with no modeled weight DRAM fetches. No hardware compute/area/energy Pareto is established. [Storage fields](storage_cost.csv) · [full contract](metric_definitions.json) · [method limits](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).','',
          '## Evidence and recovery','',
          '- **50 selected test rows:** 16 native + 16 H8 + 16 H16 + 2 NoPF. Original 88 labeled rows remain unchanged; [30 H16-completion rows](per_run_h16_completion.csv) retain their new protocol. No references are relabeled as new runs. [Earlier source](per_run.csv).',
          '- **New complete raw:** `formal_study/runs/602_width_templates_20261017/stride_streamer_h16/` under both cache roots, including checkpoints/bindings, correctness, runs, validation selections and analysis.',
          '- **Mechanisms/history:** [template findings](FINDINGS.md) · [paper story](PAPER_STORY.md) · [replay alignment](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md) · [full-ROI actions](action_histograms_full_roi.csv) · [stages](phase_progress.csv) · [new completion stages](phase_progress_h16_completion.csv).',
          '- **Local raw:** `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/`; `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/archive/20261017/raw_v2/`.',
          '- **Sacramento raw:** `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/`; `/home/qianruw/cache/formal_study/archive/20261017/raw_v2/`. [Layout and preservation](../../../docs/602_width_templates/LAYOUT_AND_RECOVERY.md).',
          '- **Rebuild figures/tables only:** `python3 formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/build_reader_results.py` with matplotlib. [Independent checker](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/verify_reader_results.py). [Action/history plot source](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/plot_h16_completion.py) rebuilds those figures from retained CSVs only. All runtime files stay inside cache.','']
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
        fig.text(.035,.028,'All four routes have H8/H16 results. C = useful LOAD requests / matched NoPF misses; not net miss elimination.',fontsize=9)
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
            fig.text(.035,.02,'Black outline / dashed line: nondominated within each panel. NoPF is reused. Seed7, MAC0; '+('all eight models' if all_widths else 'H8 models only')+'.',fontsize=9)
            fig.text(.035,.003,'Requests include subsequent cache merges/rejections. DRAM is actual read + write bus traffic; compute/area are not axes.',fontsize=9)
            fig.tight_layout(rect=(0,.13,1,.96));save(fig,f"reader_pareto_{'h16_' if all_widths else ''}cap{cap}")


def plot_width_cost(rows):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names=('Native','H8 NN','H16 NN');colors=('#999999','#0072B2','#D55E00')
    def save(fig,name):
        for ext in ('svg','png'):
            path=OUT/(name+'.'+ext);fig.savefig(path,dpi=160,bbox_inches='tight')
            if ext=='svg':path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        plt.close(fig)
    points=[]
    for roi in (0,1):
        baseline=next(r for r in rows if r['arm']=='nopf' and r['roi']==roi)
        for cap in (0,16144):
            fig,axes=plt.subplots(3,4,figsize=(15,9))
            for col,pol in enumerate(POLICIES):
                selected=[next(r for r in rows if (r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget'])==(pol,a,h,roi,cap)) for a,h in (('native',0),('nn',8),('nn',16))]
                for row,(key,title,scale) in enumerate((('ipc','IPC',1),('requests_per_million_instructions','Actual requests / M',1),('dram_bus_bytes','Completed DRAM MiB',2**20))):
                    ax=axes[row,col];values=[r[key]/scale for r in selected];ax.bar(range(3),values,color=colors,width=.65)
                    for x,y in enumerate(values):ax.text(x,y,f'{y:.4f}' if row==0 else f'{y:,.1f}',ha='center',va='bottom',fontsize=8)
                    ax.set_xticks(range(3),names,fontsize=8);ax.set_title(NAMES[pol]+' · '+title);ax.set_ylim(0,max(values)*1.18);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                    if row==0:ax.axhline(baseline['ipc'],color='#444444',ls='--',lw=.8)
                for r in selected:
                    points.append({k:r[k] for k in ('result_id','protocol','policy','arm','hidden_size','roi','budget','ipc','requests_per_million_instructions','dram_bus_bytes')})
            fig.suptitle(f'Native, H8 and H16 · ROI{roi} · {budget(cap)} · seed7 · MAC0',fontsize=14)
            fig.text(.035,.012,'Dashed IPC line: matched NoPF. Actual requests are credit-allowed cache calls; DRAM is measured read + write bus traffic.',fontsize=9)
            fig.tight_layout(rect=(0,.04,1,.96));save(fig,f'reader_widths_roi{roi}_cap{cap}')
    dump('reader_width_points.csv',points)
    with (OUT/'model_size.csv').open(newline='') as stream:size={(r['policy'],int(r['hidden_size'])):r for r in csv.DictReader(stream)}
    cost=[]
    for r in rows:
        if r['arm']!='nn' or r['budget']!=0:continue
        assert r['completed_updates'] and r['inference_macs'] is not None and r['inference_nonlinears'] is not None
        item={k:r[k] for k in ('result_id','protocol','policy','hidden_size','roi','cycles','completed_updates','inference_macs','inference_nonlinears')}
        item.update(dense_MACs_per_completed_update=r['inference_macs']/r['completed_updates'],nonlinears_per_completed_update=r['inference_nonlinears']/r['completed_updates'],aggregate_MAC4_dense_work_over_MAC0_cycles=math.ceil(r['inference_macs']/4)/r['cycles'],FP32_parameter_bytes=int(size[(r['policy'],r['hidden_size'])]['FP32_parameter_bytes']),configured_hc_payload_bytes=int(size[(r['policy'],r['hidden_size'])]['configured_hc_payload_bytes']),scope='actual ROI counted work and partial storage; no finite-cost IPC, hardware area or weight DRAM prediction')
        cost.append(item)
    dump('reader_cost.csv',cost)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for col,(key,title) in enumerate((('dense_MACs_per_completed_update','Counted dense MACs / completed update'),('aggregate_MAC4_dense_work_over_MAC0_cycles','ceil(counted MACs / 4) / MAC0 ROI cycles'))):
        ax=axes[col]
        for j,(h,roi,color,hatch) in enumerate(((8,0,colors[1],''),(16,0,colors[2],''),(8,1,colors[1],'//'),(16,1,colors[2],'//'))):
            values=[next(r[key] for r in cost if (r['policy'],r['hidden_size'],r['roi'])==(p,h,roi)) for p in POLICIES]
            x=[i+(j-1.5)*.19 for i in range(4)];ax.bar(x,values,.18,color=color,hatch=hatch,label=f'H{h} ROI{roi}')
        ax.set_xticks(range(4),[NAMES[p] for p in POLICIES]);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if col==1:ax.axhline(1,color='#444444',ls='--',lw=.8)
    axes[0].legend(ncol=2,fontsize=8);fig.suptitle('All eight models: measured uncapped inference work · synchronous MAC0',fontsize=14)
    fig.text(.035,.065,'Aggregate MAC4 work is a necessary throughput condition only; scheduling, dependency latency, nonlinear units and bandwidth are excluded.',fontsize=9)
    fig.text(.035,.020,'Native compute is unmeasured (NA). Parameter/state bytes are separate in model_size.csv; neither plot measures hardware area or finite-cost IPC.',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.94));save(fig,'reader_cost')



def document_mechanisms(rows,pairs):
    with (OUT/'paired_controls_h16_completion.csv').open(newline='') as f:controls=list(csv.DictReader(f))
    assert len(controls)==28
    def control(pol,roi,cap,mode='NN_free_template'):
        return next(r for r in controls if (r['policy'],int(r['roi']),int(r['budget']),r['control'])==(pol,roi,cap,mode))
    def changes(pol,cap,mode='NN_free_template'):
        return ' / '.join(f'{float(control(pol,roi,cap,mode)["control_relative_IPC_pct"]):+.4f}%' for roi in (0,1))
    sizes={r['policy']:r for r in csv.DictReader((OUT/'model_size.csv').open()) if r['hidden_size']=='16'}
    h16=[p for p in pairs if p['hidden_size']==16]
    width=[]
    for n in rows:
        if n['arm']!='nn' or n['hidden_size']!=16:continue
        h8=next(r for r in rows if (r['policy'],r['arm'],r['hidden_size'],r['roi'],r['budget'])==(n['policy'],'nn',8,n['roi'],n['budget']))
        width.append((n,100*(n['ipc']/h8['ipc']-1)))
    wins=sum(v>0 for _,v in width);losses=[(n,v) for n,v in width if v<0]
    assert len(width)==16
    findings=['# 602 mechanisms: H8 and H16','',
      '- **Main comparison:** [IPC, accuracy, coverage, timeliness and traffic Pareto](README.md). This page isolates width and mechanism controls; it does not replace those metrics.',
      '- **Completed extension:** 30/30 new conditions: 2 validation + 28 full test ROIs. [Audit](audit_h16_completion.json) · [new rows](per_run_h16_completion.csv) · [28 paired controls](paired_controls_h16_completion.csv). Earlier 64 H8 + 24 width/template rows retain their original identities.',
      '- **Contract:** seed7; frozen weights, live h/c; 5M cache/state warmup; ROI0 25–50M, ROI1 75–100M; MAC0; uncapped or cap16144. Templates are frozen using each model’s validation 20–25M only. No checkpoint was retrained.','',
      '## Width: performance improves in most measured pairs','',
      f'- H16 beats its own native in **{sum(p["ipc_gain_pct"]>0 for p in h16)}/16 IPC pairs** and H8 in **{wins}/16**. [Native pairs](reader_pairs.csv) · [complete run metrics](reader_metrics.csv).',
      '- **Negative width result:** capped Streamer H16 is below H8 by '+ ' / '.join(f'{abs(v):.4f}%' for n,v in sorted(losses,key=lambda x:x[0]['roi']))+' in ROI0 / ROI1. Wider is not uniformly better.',
      f'- **Traffic cost:** {sum(p["dram_change_pct"]>0 for p in h16)}/16 H16/native pairs increase actual DRAM. Only capped ROI1 SPP lowers it. Request and DRAM Pareto conclusions differ; [exact classifications](reader_pairs.csv).','',
      '## H16: can a frozen NN-free template reproduce it?','',
      '- Below, **template IPC change = 100 × (template/full NN − 1)**. Positive favors the template. Each cell lists ROI0 / ROI1. [Exact IPC, actual requests/M and DRAM](paired_controls_h16_completion.csv).',
      '- Fixed-payload keeps full NN gate/forward/state but replaces the final list. NN-free template executes no NN. Reset-state clears h/c before each callback; all other rules stay fixed. These are distinct interventions.','',
      '| H16 route | Own validation-frozen list | Uncapped template IPC change | Capped template IPC change |',
      '| --- | --- | --- | --- |']
    payload={'stride':'K2: (0,L2), (+2,L2); filter 0 before credit','streamer':'K3: (+1,+2,+3), L2','ampm':'K2: (+9,+8), L2','spp':'K5: (+1,L2), (+2..+5,LLC)'}
    for pol in POLICIES:findings.append(f'| {NAMES[pol]} | {payload[pol]} | {changes(pol,0)} | {changes(pol,16144)} |')
    findings+=['',
      '- **Streamer and SPP:** templates exactly match cycles, instructions, allowed requests, DRAM and LOAD misses in all four cases. This supports replication of the measured outcomes, not policy identity on unseen inputs.',
      '- **AMPM:** template residuals are below 0.125% IPC; the NN wins slightly uncapped and loses slightly capped. These small residuals do not isolate state or payload value.',
      '- **Stride:** the effective +2 template wins 3/4 IPC cases while issuing more requests. Full NN wins capped ROI1, with fewer requests and less DRAM. This does not establish recurrent necessity; a matched static PC gate is not evaluated in this protocol.','',
      '## H16: fixed payload versus reset state','',
      '- **Uncapped fixed payload:** Streamer/SPP exactly reproduce the full NN in both ROIs; AMPM matches its template outcomes above. [All interventions](paired_controls_h16_completion.csv).',
      '- **Reset-state IPC change versus full NN, ROI0 / ROI1:** '+ '; '.join(f'{NAMES[p]} {changes(p,0,"reset_state")}' for p in ('streamer','ampm','spp'))+'.',
      '- Reset-state reduces requests and DRAM as well as IPC. A recurrence-sensitive output is not evidence that recurrence is necessary for the gain: the NN-free templates still match or nearly match those outcomes.','',
      '## H8 under the same cap','',
      '- **Streamer:** template/full IPC +0.0004% / −0.0051%; **AMPM:** +0.8939% / +0.9542%; **SPP:** 0.0000% / −0.0006% (ROI0 / ROI1). [Original exact pairs](paired_comparisons.csv).',
      '- Streamer/SPP outcomes are nearly reproduced; AMPM’s template improves IPC with fewer requests. Stride’s earlier gate controls remain a [separate protocol](../../../docs/602_gate_mechanism/REPORT.md). Native-relative H8 losses remain visible in the main reader.','',
      '## Size, cost and limits','',
      '- [Model size and inference state](README.md#model-size-and-inference-state): all eight checkpoints and tensor exports verified; FP32 weights = 4 × parameters; h/c storage is separate. [Layer counts](model_size_layers.csv) · [tensor counts](model_size_tensors.csv).',
      '- MAC0 IPC does not establish finite-compute benefit. [Counted full-ROI work](reader_cost.csv) and host allocation probes are neither synthesized area nor measured energy.',
      '- One workload/seed, independently trained widths, differing route training/state budgets, previously inspected ROI0 and different SPP fill feedback limit generalization. A failed template is not proof of stateful novelty.',
      '- Historical 24-row report/figures are preserved under `formal_study/runs/602_width_templates_20261017/analysis_final/`; original H8 evidence is under `formal_study/archive/20261017/raw_v2/`. Old replay remains [separate](../../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md).',
      '- New full raw on local: `/Users/angelawoo/Documents/CMU/ece/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`; Sacramento: `/home/qianruw/cache/formal_study/runs/602_width_templates_20261017/stride_streamer_h16/`.','']
    (OUT/'FINDINGS.md').write_text('\n'.join(findings).replace('\n','\n'))
    story=['# 602 paper story: H8 and H16','',
      '- **Question:** Can small recurrent students trained on traditional prefetcher actions improve IPC and request quality under a shared request ceiling, and do the gains need NN inference?',
      '- **Scope:** four native/student routes, H8/H16, retained seed7, two complete 25M ROIs, 5M warmup, frozen weights and live state, MAC0. [Protocol and all quality metrics](README.md).',
      '- **Supported — performance is route/width dependent:** H8 wins 9/16 own-native IPC pairs; H16 wins 16/16. H16 beats H8 in 14/16, losing both capped Streamer cases. These are measured pairs, not seed-robust guarantees. [Exact pairs](reader_pairs.csv).',
      '- **Supported — IPC and quality/traffic can disagree:** SPP H8 improves LLC accuracy while losing IPC; H16 improves IPC despite lower L2 accuracy than native. H16 increases DRAM in 15/16 native pairs. [IPC/quality and traffic Pareto](README.md).',
      '- **Not supported — universal need for learned payload/state:** H16 Streamer/SPP NN-free templates match all measured IPC/traffic cases. AMPM is within 0.125% IPC. Stride +2 wins 3/4; full NN wins capped ROI1. Reset-state losses alone do not establish necessity. [Actual interventions](FINDINGS.md).',
      '- **Supported — explicit software storage accounting:** all eight tensor counts and checkpoint/export bytes agree; weights, recurrent state and observed host overhead are separate. Shorter history alone does not shrink fixed streaming LSTM storage. [Parameters → bytes](README.md#model-size-and-inference-state).',
      '- **Unverified — finite-resource advantage and generality:** analytical work counts and MAC0 IPC do not establish hardware area/energy/finite-cost Pareto. One seed/workload, differing route recipes and SPP feedback, plus prior ROI0 inspection, remain limits. [Cost points](reader_cost.csv) · [method limits](../../../formal_NN_training/experiments/602_four_prefetcher_formal/report/METHOD_LIMITATIONS.md).','']
    (OUT/'PAPER_STORY.md').write_text('\n'.join(story).replace('\n','\n'))


def main():
    rows=load();pairs=paired(rows);fronts=frontiers(rows)
    dump('reader_metrics.csv',rows);dump('reader_pairs.csv',pairs);dump('reader_pareto.csv',fronts)
    document(rows,pairs,fronts);document_mechanisms(rows,pairs);plot(rows);plot_width_cost(rows)
    print('Reader outputs: H8/H16, 50 run rows, 32 native/NN pairs, 176 panel memberships, 13 SVG + PNG figure pairs. No simulation or training.')

if __name__=='__main__':main()
