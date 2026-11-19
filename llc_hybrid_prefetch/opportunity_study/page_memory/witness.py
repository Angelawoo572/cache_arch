"""Recover the fixed nonadjacent cue and its ensuing first use in every arm."""
import csv
from summarize import records
from run import DATA,HERE,MODES

ANCHOR='51608215'
DEMAND='51608294'

def main():
    out=[]
    for mode in MODES:
        folder=DATA/'runs'/mode
        cue=next((a for a in records(folder,'decisions') if a['instr']==ANCHOR and a['operand']=='0'),{})
        stages={r['stage']:r for r in records(folder,'candidates') if r['instr_id']==ANCHOR and r['operand']=='0'}
        lookup=next((r for r in records(folder,'loads') if r['instr_id']==DEMAND and r['operand']=='0'
                     and r['level']=='LLC' and r['stage']=='lookup'),{})
        head=sum(int(r['cycles']) for r in records(folder,'head') if r['instr_id']==DEMAND and r['operand']=='0')
        useful=stages.get('timely',{})
        out.append(dict(mode=mode,anchor_instr=ANCHOR,prior_age=cue.get('prior_age',''),cue_cycle=cue.get('cycle',''),
                        chosen=cue.get('chosen',''),target_instr=DEMAND,target_line=lookup.get('physical_line',''),
                        target_lookup=lookup.get('cycle',''),target_hit=lookup.get('hit',''),head_wait=head,
                        native_reject=stages.get('rejected',{}).get('reason',''),
                        dispatch=stages.get('dispatch',{}).get('cycle',''),fill=stages.get('fill',{}).get('cycle',''),
                        first_use=useful.get('demand_instr_id',''),first_use_cycle=useful.get('cycle','')))
    with (HERE/'witness.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0],lineterminator='\n');w.writeheader();w.writerows(out)
if __name__=='__main__':main()
