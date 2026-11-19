"""Follow every frozen page-offset exclusive over next-line into the native arms."""
import csv
import gzip
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from evaluate import HERE, grant, guarded, openz, writer
from page_association import signals, targets

STUDY = HERE.parent
DATA = STUDY / 'raw/page_memory'
BASE = STUDY / 'raw/runs'


def freeze():
    z = openz('gap_bfs3_late-c0')
    end = int(.6 * len(z['t']))
    service_mask = (z['dispatch'] >= 0) & (z['fill'] >= z['dispatch']) & (z['fill'] < z['t'][end])
    service = int(np.median((z['fill'] - z['dispatch'])[service_mask]))
    signals_by_name, links = signals(z)
    future = targets(z, links, service)
    fit = guarded(z,0,end)
    fit = fit[grant(fit,2,131)]
    keys = signals_by_name['page_offset_only']
    # Every context/outcome count uses only the frozen prefix and its guarded labels.
    fit_offset = np.where(future[fit]>=0, z['pa'][np.maximum(0,future[fit])]%64,-1)
    rows = {}
    with (DATA / 'shadow_reference.csv').open() as f:
        for r in csv.DictReader(f):
            i = int(r['index'])
            q = int(future[i])
            if q < 0:
                continue
            line = int(z['pa'][q])
            if int(r['page_offset_only']) != line or int(r['base']) + 1 == line:
                continue
            key = (str(z['instr'][q]), str(z['op'][q]))
            if key in rows:
                raise ValueError('duplicate qualifying target')
            base = int(r['base'])
            same = [j for j in range(max(0,i-64),i) if int(z['pa'][j])//64==base//64 and 0 <= int(z['t'][i]-z['t'][j]) <= 4096]
            previous = same[-1] if same else None
            training = keys[fit]==keys[i]
            current_train = signals_by_name['current'][fit]==signals_by_name['current'][i]
            evidence = 'same_page_context' if training.sum()>=8 else ('current_backoff' if current_train.sum()>=8 else 'global_backoff')
            repeat = 2*base-int(z['pa'][previous]) if previous is not None else None
            rows[key] = dict(cue_instr=r['instr'], cue_operand=r['operand'], cue_pc=int(z['pc'][i]),
                             cue_line=base,cue_page_offset=base%64,
                             previous_instr=int(z['instr'][previous]) if same else '',
                             previous_operand=int(z['op'][previous]) if same else '',
                             previous_pc=int(z['pc'][previous]) if same else '',
                             previous_offset=int(z['pa'][previous])%64 if same else 64,
                             previous_age_ingresses=i-previous if same else '',
                             previous_age_cycles=int(z['t'][i]-z['t'][previous]) if same else '',
                             training_observations=int(training.sum()),training_chosen_offset_positives=int((training&(fit_offset==line%64)).sum()),
                             inference_context=evidence,current_training_observations=int(current_train.sum()),
                             current_training_target_positives=int((current_train&(fit_offset==line%64)).sum()),
                             current_chosen_offset=int(r['current'])%64,
                             repeat_offset=repeat%64 if repeat is not None and repeat//64==base//64 and repeat!=base else '',
                             target_instr=key[0], target_operand=key[1], target_pc=int(z['pc'][q]),
                             target_virtual_line=int(z['va'][q]), target_line=line, target_page_offset=line%64,
                             stride_lead_cycles=int(z['t'][q]-z['t'][i]))
    return rows


def open_log(d, name):
    p = d / (name + '.csv')
    return p.open() if p.exists() else gzip.open(str(p) + '.gz', 'rt')


def native(mode, d, cases):
    # Keep only the wanted dynamic demands. Read the full first-use ledger so that
    # service from a different trigger or conventional candidate counts separately.
    out = {k:dict(llc='no_lookup', head_wait=0, first_use='none', first_use_instr='', at_frozen_cue='') for k in cases}
    tokens = tuple(',' + k[0] + ',' + k[1] + ',' for k in cases)
    with open_log(d, 'loads') as f:
        hdr = f.readline().rstrip().split(',')
        for line in f:
            if ',LLC,lookup,' not in line or not any(t in line for t in tokens):
                continue
            r = dict(zip(hdr, line.rstrip().split(',')))
            k = r['instr_id'], r['operand']
            if r['cpu'] != '0' or k not in out:
                continue
            case = cases[k]
            if (r['pc'],r['physical_line'],r['virtual_line']) != tuple(str(case[c]) for c in ('target_pc','target_line','target_virtual_line')):
                raise ValueError((mode,k,'identity mismatch'))
            assert out[k]['llc'] == 'no_lookup'
            out[k]['llc'] = 'hit' if r['hit']=='1' else 'miss'
    with open_log(d,'head') as f:
        for r in csv.DictReader(f):
            k = r['instr_id'],r['operand']
            if r['cpu']=='0' and k in out:
                out[k]['head_wait'] += int(r['cycles'])
    source = {}
    interesting = {str(case['target_line']) for case in cases.values()}
    with open_log(d,'candidates') as f:
        for r in csv.DictReader(f):
            if r['target'] not in interesting:
                continue
            if r['stage']=='proposed':
                source[r['id']] = r['reason']
            k = r['demand_instr_id'], r['demand_operand']
            if r['demand_cpu']!='0' or k not in out or r['stage'] not in ('timely','late'):
                continue
            assert int(r['target'])==cases[k]['target_line'] and out[k]['first_use']=='none'
            kind = source[r['id']]
            out[k]['first_use'] = kind + '_' + r['stage']
            out[k]['first_use_instr'] = r['instr_id']
            out[k]['at_frozen_cue'] = int((r['instr_id'],r['operand'])==(cases[k]['cue_instr'],cases[k]['cue_operand']))
    return mode,out


def main():
    cases = freeze()
    runs = [('nopf',BASE/'gap_bfs3_late-nopf'), ('stride',BASE/'gap_bfs3_late-stride'),
            *((mode,DATA/'runs'/mode) for mode in ('current','repeat','offset16','offset','next'))]
    for mode in ('spp','multi'):
        directory = STUDY/('raw/conventional/runs/gap_bfs3_late__'+mode+'__four_targets')
        if (directory/'complete.json').exists():
            runs.append((mode,directory))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda pair:native(*pair,cases),runs))
    for mode, rows in results:
        for k, record in rows.items():
            cases[k].update({mode+'_'+field:v for field,v in record.items()})
    for mode in ('spp','multi'):
        if mode not in dict(runs):
            for record in cases.values():
                record[mode+'_llc'] = 'no_per_load_log'
    for r in cases.values():
        r['offset_unique_vs_next'] = int(r['offset_llc']=='hit' and r['next_llc']=='miss')
        r['stride_exposed'] = int(r['stride_llc']=='miss' and r['stride_head_wait']>0)
    writer(HERE/'page_exclusive_loads.csv',list(cases.values()))


if __name__=='__main__':
    main()
