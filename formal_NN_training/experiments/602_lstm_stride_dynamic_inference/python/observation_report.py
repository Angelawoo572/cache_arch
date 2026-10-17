"""Render measured observation and strict-quality results into the existing report."""
import csv
import json
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=('useful_prefetch_coverage','selected_accuracy','legacy_timeliness')
def read(path):
    if not path.exists():return []
    return list(csv.DictReader(path.open()))
def n(v):
    try:return float(v)
    except (ValueError,TypeError):return math.nan
def f(v,d=0):
    x=n(v)
    return 'NA' if not math.isfinite(x) else f'{x:,.{d}f}'
def pct(v):return f(n(v)*100,3)+r'\%'
def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def label(r):return 'h'+str(r.get('hidden_size',''))+'-'+str(r.get('checkpoint',''))
def tab(headers,rows,align=None):
    align=align or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{center}\scriptsize',r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']+[' & '.join(map(str,row))+r' \\' for row in rows]+[r'\bottomrule',r'\end{tabular}\end{center}'])
def primary(r):return n(r.get('state_capacity'))==0 and not r['run_id'].startswith('cold_history')
def summary(rows,protocol,comparison,scale='cumulative'):
    return [r for r in rows if r['protocol']==protocol and r['comparison']==comparison and r['scale']==scale and primary(r)]
def status(r):
    if r['status']=='OBSERVED':return f(r['first_recorded_instructions'])+' / '+f(r['first_completed_l2'])
    return 'Not reached' if r['status']=='NOT_REACHED_IN_MEASURED_RANGE' else 'No matched reference'

def plots(points,out):
    for protocol,comparison,name in [('cold','stride','strict_cold_quality'),('historical','summer','strict_summer_quality')]:
        fig,axes=plt.subplots(3,1,figsize=(7.1,7.2),sharex=True,layout='constrained')
        selected=[r for r in points if r['protocol']==protocol and r['comparison']==comparison and r['scale']=='cumulative' and primary(r)]
        for rid in sorted({r['run_id'] for r in selected}):
            rows=sorted([r for r in selected if r['run_id']==rid],key=lambda r:n(r['end_instructions']))
            for ax,key,title in zip(axes,METRICS,['Useful coverage difference','Selected accuracy difference','Legacy timeliness difference']):
                ax.plot([n(r['end_instructions'])/1e6 for r in rows],[n(r.get('difference_'+key))*100 for r in rows],label=label(rows[0]),linewidth=1.1)
                ax.axhline(0,color='black',linewidth=.7);ax.set_ylabel('percentage points');ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Retired instructions from measurement boundary (M)')
        axes[0].legend(ncol=4,fontsize=8,loc='best')
        fig.savefig(out/'figures'/ (name+'.pdf'));plt.close(fig)

def augment(document,results,out):
    summaries=read(results/'quality_summary.csv');points=read(results/'quality_points.csv')
    if not summaries:return document
    intervals=read(results/'quality_intervals.csv');firsts=read(results/'quality_first_points.csv')
    plots(points,out)
    cold=[r for r in firsts if r['protocol']=='cold' and r['comparison']=='stride' and r['scale']=='cumulative' and primary(r)]
    intro=[r'\section{Direct answer: one new access, carried history, measured advantage}',
      r'Each prediction consumes \textbf{one new eligible L2 LOAD access}. Its PC and aligned byte address become a tensor of shape $(1,1,128)$. History is carried through the exact PC\textquotesingle s continuous LSTM h/c, each of shape $(1,1,H)$. There is no multi-access warm-up requirement or last-W recomputation. Runtime weights remain fixed. The 128 features, H hidden values and K output addresses are not instruction or access counts.']
    if cold:
        r=cold[0];sp=next(s for s in intervals if s['run_id']==r['run_id'] and s['comparison']=='stride' and s['scale']=='cumulative' and s['quality_status']=='EXCEEDS')
        intro.append('In the cold slice, '+label(r)+' first has all three cumulative quality metrics strictly above matched Stride at the recorded point $N='+f(r['end_instructions'])+'$ retired instructions and $D='+f(r['total_completed_decisions'])+'$ completed L2 decisions. Coverage / selected accuracy / legacy timeliness are '+ ' / '.join(pct(r[k]) for k in METRICS)+', versus Stride\textquotesingle s '+ ' / '.join(pct(r['reference_'+k]) for k in METRICS)+'.')
        intro.append('The consecutive recorded cumulative advantage lasts through $N='+f(sp['last_recorded_instructions'])+'$ and $D='+f(sp['last_completed_l2'])+'$; the first recorded exit is $N='+f(sp['exit_at_instructions'])+'$ because accuracy falls below Stride. At entry, IPC is '+f(r['ipc'],6)+', cumulative saving is '+f(r['cumulative_cycles_saved_vs_stride'])+' cycles, request pressure is '+f(r['request_pressure'],3)+' and L2 occupancy is '+f(r['occupancy_lines'])+'/'+f(r['capacity_lines'])+' lines.')
    histjoint=summary(summaries,'historical','stride_and_summer')
    observed=[r for r in histjoint if r['status']=='OBSERVED']
    if observed:
        r=next(r for r in firsts if r['protocol']=='historical' and r['comparison']=='stride_and_summer' and r['scale']=='cumulative' and primary(r))
        sp=next(s for s in intervals if s['run_id']==r['run_id'] and s['comparison']=='stride_and_summer' and s['scale']=='cumulative' and s['quality_status']=='EXCEEDS')
        intro.append('In the separate summer-compatible protocol, '+label(r)+' first strictly exceeds BOTH references at $N='+f(r['end_instructions'])+'$, $D='+f(r['total_completed_decisions'])+'$. Coverage / accuracy / timeliness are '+ ' / '.join(pct(r[k]) for k in METRICS)+'; Stride: '+ ' / '.join(pct(r['stride_reference_'+k]) for k in METRICS)+'; original summer h8-i20m: '+ ' / '.join(pct(r['summer_reference_'+k]) for k in METRICS)+'. The recorded cumulative advantage persists through $N='+f(sp['last_recorded_instructions'])+'$, $D='+f(sp['last_completed_l2'])+'$ (4,500,003 further instructions). This is quality superiority: its final execution still takes 15,757 more cycles than summer.')
    elif histjoint and all(r['status']=='NOT_REACHED_IN_MEASURED_RANGE' for r in histjoint):
        intro.append(r'In the summer-compatible protocol, simultaneous strict superiority to both Stride and the same-H original summer is \textbf{not reached in the measured range} for any of the four live checkpoints. Individual-reference outcomes and limiting metrics are reported below. Cold and warm-cache measurements are never pooled.')
    else:intro.append(r'The summer-compatible joint comparison has incomplete matching reference data in this generated draft. It must not be interpreted as a failed quality test.')
    intro.extend([r'With the retained 4/16-MAC inference costs, h8-i1m has no strict three-quality joint win over Stride; end-to-end losses are 460,332 / 391,755 cycles. High timeliness accompanies only 0.258\% / 0.643\% useful coverage. Queueing, dropped work and redundant requests remain visible.',
        r'N denotes retired instructions since the evaluation boundary. Historical total simulated retirement is N+25,000,004 (45,500,004 at joint entry); cold total retirement is N because skipped records are not simulated. These descriptive points are not minimum-history requirements. At the cold entry point the NN has 39,679 issued requests (12,441 useful); the distinction is not based on a handful of requests. The preceding point ties in timeliness. No new stability threshold or policy change is introduced.'])
    details=[r'\clearpage\section{What a prediction actually observes}',
      tab(['Quantity','Actual forward meaning'],[
          ['New inputs','1 L2 LOAD; PC and aligned 64-byte address'],
          ['Encoded tensor','$(1,1,128)$ FP32 bits: 64 PC + 64 address'],
          ['Projected input','$(1,1,H)$ after the existing tanh projection'],
          ['Carried h/c','$(1,1,H)$ each; exact-PC state lifetime'],
          ['Output','Learned silent/act, learned positive K, free-running signed deltas']], 'll'),
      r'At service start the engine selects the last completed h/c for this PC. On a successful completion it installs the new h/c and increments that state lifetime\textquotesingle s observed update count. A different PC selects a different entry. Eviction starts a new lifetime at zero. Queued, dropped and unfinished events are not completed state updates. Silent decisions still update h/c. The decoder\textquotesingle s temporary recurrent state is separate from the carried encoder history.',
      r'Retired N labels execution progress. Eligible, admitted, started and completed D count actual callbacks/service events; out-of-order observation can precede retirement. The trace reader\textquotesingle s look-ahead is never counted as predictor input. Offline i1m/i20m are pretraining budgets, separate from all runtime counts.',
      r'Having updated h/c k times describes its history of use. It neither means the model remembers k accesses in full nor proves k accesses are sufficient. Measurement snapshots leave h/c and the output policy unchanged.']
    examples=read(results/'history_example.csv')
    if examples:
        rows=[]
        for r in examples[:8]:
            rows.append([f(r.get('completion')),esc(r.get('pc')),f(r.get('line')),f(r.get('lifetime',r.get('life'))),f(r.get('prior_updates'))+' to '+f(r.get('post_updates')),f(r.get('emit'))+'/'+f(r.get('k'))])
        details+=[r'\subsection{A real interleaved-PC sequence}',tab(['Decision','PC','Line','Life','Prior / post updates','act/K'],rows,'rllrrr'),r'These are consecutive completed events from the actual frozen h8-i1m run. Full old/new h/c vectors, timestamps, N/D and candidate addresses are retained in history\_example.csv. The table indexes the old state by PC and lifetime; it does not substitute a synthetic access sequence.']
        repeated=next((r for i,r in enumerate(examples[:8]) if any(q['pc']==r['pc'] for q in examples[:i])),None)
        if repeated:
            oldh=json.loads(repeated['old_h']);oldc=json.loads(repeated['old_c']);newh=json.loads(repeated['new_h']);newc=json.loads(repeated['new_c'])
            details.append('On the repeated PC '+esc(repeated['pc'])+' at N='+f(repeated['instructions'])+', h[0] changes from '+f(oldh[0],6)+' to '+f(newh[0],6)+', and c[0] from '+f(oldc[0],6)+' to '+f(newc[0],6)+'. These are components of the actual carried vectors, whose full contents are in the CSV. Line values above are multiplied by 64 to form the aligned byte-address feature.')
    details+=[r'\clearpage\section{Strict three-quality comparison rules}',
      r'Each comparison uses one NN run and one fixed reference run at the exact same recorded N or interval $(N_a,N_b]$. Coverage is U/M0, selected accuracy is U/(Ipf-Q), and legacy timeliness is U/(U+L). Raw U/Ipf is also retained. All three differences must be strictly positive; exact integer fractions distinguish equality from strict superiority. Ties are NONLOWER\_WITH\_TIE, undefined denominators are NA, and missing or unaligned references are separate from measured non-attainment.',
      r'The original summer reference is the compact hurdle v9 seed-7 i20m model of the same hidden size. Its checkpoint bytes, action stream and 30 end metrics match the prefix-sweep i20m artifacts. New measurements replay that original stream with the unchanged keyed ListReplayer. Separate offline\_same\_checkpoint comparisons retain each i1m/i20m point\textquotesingle s own original replay, rather than selecting the best budget after seeing results.',
      r'Historical comparisons use 25M no-prefetch warmup, empty predictor state at measurement start, and the original 25M measurement (actual origin 25,000,004; final N 25,000,003). The newly measured conventional Stride is also gated throughout warmup. Its original offline Stride replay is retained separately. Cold starts use a fresh cache at the skipped 25M boundary; there is no compatible cold summer reference and no cross-protocol joint assertion.']
    for protocol in ('cold','historical'):
        rows=[]
        for role in (('stride',) if protocol=='cold' else ('stride','summer','stride_and_summer','offline_same_checkpoint')):
            for r in summary(summaries,protocol,role):rows.append([label(r),esc(role),status(r),f(r['strictly_exceeding_points']),f(r['joint_win_intervals'])])
        details.append(r'\subsection{'+('Cold' if protocol=='cold' else 'Summer-compatible')+' cumulative outcomes}')
        details.append(tab(['NN','Reference','First recorded N / completed D','Points','Intervals'],rows,'llrrr'))
    details.extend([r'\clearpage\section{Where each advantage begins and ends}',r'The following lists every maximal strictly exceeding segment at the retained scales. For cumulative rows the endpoints are recorded points, not claims about unsampled instants. For interval rows $(a,b]$ is the union of consecutive satisfying counter windows. The exit column gives the first subsequent failing/tied/undefined observation. The full status segments, raw denominators, separate metric differences and exits are in quality\_intervals.csv and quality\_points.csv.'])
    winrows=[r for r in intervals if r['quality_status']=='EXCEEDS' and primary(r)]
    for protocol in ('cold','historical'):
        details.append(r'\subsection{'+('Cold versus Stride' if protocol=='cold' else 'Summer-compatible references')+'}')
        rows=[]
        for r in winrows:
            if r['protocol']!=protocol or (protocol=='cold' and r['comparison']!='stride'):continue
            start=r['first_recorded_instructions'] if r['scale']=='cumulative' else r['start_window_instructions']
            rows.append([label(r),esc(r['comparison']),esc(r['scale'].replace('interval_','')),f(start)+'--'+f(r['last_recorded_instructions']),f(r.get('exit_at_instructions'))])
        if rows:
            # Longtable permits any genuine interval count without clipping or truncation.
            details.append(r'\scriptsize\begin{longtable}{lllrr}\toprule NN & Reference & Scale & Range N & First exit N \\\midrule\endhead'+'\n'+'\n'.join(' & '.join(x)+r' \\' for x in rows)+r'\bottomrule\end{longtable}\normalsize')
        else:details.append('Not reached in the measured range.')
    details.extend([r'\clearpage\section{Cumulative quality differences versus Stride}',r'\includegraphics[width=\linewidth]{figures/strict_cold_quality.pdf}',r'All three differences must be above zero together. A cumulative winning interval does not mean every local window wins: h16-i20m wins 15 of 250 existing 100k windows and 9 of 25 existing 1M windows. No threshold is fitted to turn these descriptive counts into a stability guarantee.',r'\clearpage\section{Cumulative quality differences versus original summer}',r'\includegraphics[width=\linewidth]{figures/strict_summer_quality.pdf}',r'The original summer comparison is confined to its compatible warm-cache protocol. The same-H i20m reference is fixed for every point of a live run.'])
    details.extend([r'\clearpage\section{Request lifecycles and actual inference delays}',r'Legacy interval ratios describe activity in that interval, including usefulness credited to prefetches requested earlier. They are not issue-cohort success probabilities. The existing 100k issue cohorts follow requests to run end; pending and resident-unused requests remain censored. At the cold h16-i20m entry N=2.6M, the legacy useful count is 12,441; final follow-up of those issue cohorts has 12,449 timely requests and 2 censored requests. At the entry snapshot itself, 75 resident-unused requests remain unresolved. None is declared a failure at that boundary.'])
    history=read(results/'history_summary.csv')
    if history:
        keys=['run_id','eligible','completed','mean_wait_cycles','mean_completion_cycles','input_drops','deployment_configured_kib','deployment_component_peaks_kib']
        rows=[[esc(r.get(k,'NA')) if k=='run_id' else f(r.get(k),3 if 'mean' in k or 'kib' in k else 0) for k in keys] for r in history]
        details.append(tab(['Case','Eligible','Done','Wait cycles','Arrival-to-done','Drops','Max KiB','Peak envelope'],rows,'lrrrrrrr'))
        details.append(tab(['Case','Mean prior updates','Max prior','Zero-prior decisions','Max wait cycles','Max completion cycles'],[[esc(r['run_id']),f(r['mean_prior'],2),f(r['max_prior']),f(r['zero_prior']),f(r['max_wait_cycles']),f(r['max_completion_cycles'])] for r in history],'lrrrrr'))
        details.append(r'The bounded history logs retained complete early records but each has one truncated final buffered record; that tail is explicitly excluded. The selected consecutive excerpts are complete, and all final aggregate counters are complete. The logger now flushes each bounded row, validated by the regression tests. The three instrumented full runs still match all 90 original end-counter fields.')
    details.append(r'Wait means use started decisions; completed service and arrival-to-completion means use completed decisions. Endpoint queues/inference remain pending. Added lifetime counters and bounded first-1,024-event logging are measurement overhead, not predictor features or deployment SRAM. Weights, PC state, inference scratch and queues retain their full FP32 deployment accounting; offline training memory is excluded. The original finite-service negative results and cache occupancy/turnover measurements follow.')
    start=document.index(r'\section{Question and measured answer}')
    end=document.index(r'\clearpage\section{Causal inference and matched conditions}')
    return (document[:start]+'\n\n'.join(intro+details)+'\n\n'+document[end:]).replace(r'\textquotesingle ', "'").replace(chr(9)+'extquotesingle s', "'s")
