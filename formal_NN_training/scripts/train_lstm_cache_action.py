#!/usr/bin/env python3
"""Outcome-aware SPP-assisted LSTM cache-action trainer.

SPP is candidate + context + supervision. The LSTM learns whether a candidate
is useful/non-duplicate, whether it should be bypassed/low-priority, and a timing
bucket. The auxiliary delta head is kept for address/debug metrics, but the
primary objective is NOT next-demand-line prediction.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD_ID, UNK_ID, DELTA_OFFSET = 0, 1, 2
TIMING_EDGES = [4, 16, 64, 256, 1024, 4096, 16384]

@dataclass
class Cfg:
    trace: str = "602.gcc_s-734B"
    cache_line_bytes: int = 64
    page_bytes: int = 4096
    seq_len: int = 64
    train_fraction: float = 0.80
    max_rows: int = 2_000_000
    pc_hash_buckets: int = 8192
    semantic_hash_buckets: int = 128
    top_delta_vocab: int = 256
    emb_dim: int = 32
    cont_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    good_threshold: float = 0.50
    bypass_threshold: float = 0.60
    loss_good: float = 1.0
    loss_bypass: float = 0.5
    loss_timing: float = 0.25
    loss_delta: float = 0.15
    future_distance_cap: int = 1_000_000
    num_workers: int = 0
    seed: int = 7
    use_pred_delta_address: bool = False

def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def h(x, mod:int)->int:
    if pd.isna(x): return 0
    return int.from_bytes(hashlib.blake2b(str(x).encode(), digest_size=8).digest(), "little") % mod

def parse_int(x, default=0):
    if pd.isna(x): return default
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, float): return default if math.isnan(x) else int(x)
    s = str(x).strip()
    if not s: return default
    return int(s,16) if s.startswith(("0x","0X")) else int(float(s))

def num(df, name, default=0, dtype=np.int64):
    if name not in df.columns: return pd.Series(np.full(len(df), default), dtype=dtype)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(dtype)

def first_col(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def bucketize(v, edges):
    return np.searchsorted(np.asarray(edges, dtype=np.float64), v, side="right").astype(np.int64)

def next_use_distance(cur_lines, cand_lines, cap):
    nxt = {}; out = np.full(len(cur_lines), cap, dtype=np.int64)
    for i in range(len(cur_lines)-1, -1, -1):
        c = int(cand_lines[i])
        if c in nxt and nxt[c] > i: out[i] = min(nxt[c]-i, cap)
        nxt[int(cur_lines[i])] = i
    return out

def class_weights(labels, ncls):
    cnt = np.bincount(labels.astype(np.int64), minlength=ncls).astype(np.float64)
    w = cnt.sum() / np.maximum(cnt * ncls, 1.0)
    return torch.tensor(np.clip(w, 0.05, 20.0), dtype=torch.float32)

def prf(tp, fp, fn):
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f = 2*p*r/max(p+r,1e-12)
    return p,r,f

def build_df(path:Path, cfg:Cfg):
    nrows = None if cfg.max_rows <= 0 else cfg.max_rows
    print(f"[load] {path} nrows={nrows}")
    df = pd.read_csv(path, nrows=nrows, engine="python")
    if "pc" not in df.columns or "addr" not in df.columns:
        raise ValueError("need pc and addr columns; run 01_run_spp_trace_dump.sh first")
    if "outcome_useful" not in df.columns or "outcome_duplicate" not in df.columns:
        raise ValueError("outcome-aware training requires outcome_useful and outcome_duplicate")
    if "trace" not in df.columns: df["trace"] = cfg.trace
    if "event_id" not in df.columns: df["event_id"] = np.arange(len(df), dtype=np.int64)
    df["pc_int"] = df["pc"].map(parse_int).astype(np.int64) if not np.issubdtype(df["pc"].dtype, np.integer) else df["pc"].astype(np.int64)
    df["addr_int"] = df["addr"].map(parse_int).astype(np.int64) if not np.issubdtype(df["addr"].dtype, np.integer) else df["addr"].astype(np.int64)
    df["cycle_num"] = num(df, "cycle", 0, np.int64) if "cycle" in df.columns else num(df, "cycle_num", 0, np.int64)
    df = df.sort_values(["trace", "cycle_num", "event_id"]).reset_index(drop=True)

    cl = cfg.cache_line_bytes; page_lines = cfg.page_bytes // cl
    df["line_addr"] = (df["addr_int"] // cl).astype(np.int64)
    df["line_offset_in_page"] = (df["line_addr"] % page_lines).astype(np.int64)
    df["prev_line_addr"] = df.groupby("trace")["line_addr"].shift(1)
    df["demand_delta"] = (df["line_addr"] - df["prev_line_addr"]).fillna(0).astype(np.int64)

    pf = first_col(df, ["pf_addr", "prefetch_addr", "candidate_addr"])
    if pf:
        df["candidate_addr"] = df[pf].map(parse_int).astype(np.int64) if not np.issubdtype(df[pf].dtype, np.integer) else df[pf].astype(np.int64)
    else:
        dcol = first_col(df, ["delta", "spp_delta"])
        if not dcol: raise ValueError("need pf_addr/prefetch_addr or delta/spp_delta")
        df["candidate_addr"] = (df["line_addr"] + num(df, dcol, 0, np.int64)) * cl
    df["candidate_line_addr"] = (df["candidate_addr"] // cl).astype(np.int64)
    df["candidate_delta"] = (df["candidate_line_addr"] - df["line_addr"]).astype(np.int64)
    df["candidate_is_self"] = (df["candidate_line_addr"] == df["line_addr"]).astype(np.int64)

    df["outcome_useful_int"] = num(df, "outcome_useful", 0, np.int64).clip(0,1)
    df["outcome_duplicate_int"] = num(df, "outcome_duplicate", 0, np.int64).clip(0,1)
    df["spp_issued_int"] = num(df, "spp_issued", 1, np.int64).clip(0,1)
    df["good_prefetch_label"] = ((df.outcome_useful_int==1)&(df.outcome_duplicate_int==0)&(df.spp_issued_int==1)&(df.candidate_is_self==0)).astype(np.int64)
    df["bypass_label"] = ((df.candidate_is_self==1)|(df.outcome_duplicate_int==1)|(df.outcome_useful_int==0)).astype(np.int64)

    if "hit" in df.columns: df["hit_int"] = num(df,"hit",0,np.int64).clip(0,1)
    elif "cache_hit" in df.columns: df["hit_int"] = num(df,"cache_hit",0,np.int64).clip(0,1)
    else: df["hit_int"] = 2
    df["access_type"] = num(df,"is_store",0,np.int64).clip(0,1) if "is_store" in df.columns else 0
    df["pc_id"] = df["pc_int"].map(lambda x: h(x, cfg.pc_hash_buckets)).astype(np.int64)
    df["semantic_id"] = df["semantic_class"].map(lambda x: h(x, cfg.semantic_hash_buckets)).astype(np.int64) if "semantic_class" in df.columns else 0
    dist = next_use_distance(df.line_addr.to_numpy(np.int64), df.candidate_line_addr.to_numpy(np.int64), cfg.future_distance_cap)
    df["candidate_next_use_distance"] = dist
    df["timing_label"] = bucketize(np.minimum(dist, cfg.future_distance_cap), TIMING_EDGES)

    cont_cols=[]
    for name in ["spp_conf", "spp_confidence", "mshr_occupancy", "l2_occupancy", "bandwidth_pressure"]:
        if name in df.columns:
            c = f"{name}_float"; df[c] = pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(np.float32); cont_cols.append(c)
    df["candidate_delta_float"] = df["candidate_delta"].astype(np.float32); cont_cols.append("candidate_delta_float")
    df["recent_hit_rate"] = df.groupby("trace")["hit_int"].transform(lambda s: s.shift(1).rolling(64, min_periods=1).mean()).fillna(0.5).astype(np.float32)
    cont_cols.append("recent_hit_rate")
    meta = {"rows": int(len(df)), "good_prefetch_rate": float(df.good_prefetch_label.mean()), "duplicate_rate": float(df.outcome_duplicate_int.mean()), "candidate_self_rate": float(df.candidate_is_self.mean()), "cont_cols": cont_cols}
    print("[labels]", json.dumps(meta, indent=2))
    return df, cont_cols, meta

def delta_vocab(a,b,topk):
    vals=np.concatenate([a.astype(np.int64),b.astype(np.int64)])
    top=pd.Series(vals).value_counts().head(topk).index.astype(np.int64).tolist()
    d2i={int(d):i+DELTA_OFFSET for i,d in enumerate(top)}; i2d={i+DELTA_OFFSET:int(d) for i,d in enumerate(top)}
    return top,d2i,i2d

def map_delta(arr,d2i): return np.asarray([d2i.get(int(x),UNK_ID) for x in arr], dtype=np.int64)

class SeqDS(Dataset):
    def __init__(self, ends, feats, labs, seq_len): self.ends=ends.astype(np.int64); self.f=feats; self.l=labs; self.seq_len=seq_len
    def __len__(self): return len(self.ends)
    def __getitem__(self, idx):
        e=int(self.ends[idx]); s=e-self.seq_len+1; sl=slice(s,e+1)
        x={k:torch.from_numpy(v[sl]).long() for k,v in self.f.items() if k!="cont"}; x["cont"]=torch.from_numpy(self.f["cont"][sl]).float()
        y={k:torch.tensor(v[e]).long() for k,v in self.l.items()}; y["end_pos"]=torch.tensor(e).long(); return x,y

class Model(nn.Module):
    def __init__(self,cfg,n_delta,n_time,n_cont):
        super().__init__(); e=cfg.emb_dim
        self.pc=nn.Embedding(cfg.pc_hash_buckets,e); self.dd=nn.Embedding(n_delta,e); self.cd=nn.Embedding(n_delta,e)
        self.off=nn.Embedding(cfg.page_bytes//cfg.cache_line_bytes,e//2); self.hit=nn.Embedding(3,e//4); self.acc=nn.Embedding(2,e//4)
        self.sem=nn.Embedding(cfg.semantic_hash_buckets,e//2); self.th=nn.Embedding(n_time,e//2); self.cont=nn.Sequential(nn.Linear(n_cont,cfg.cont_dim),nn.ReLU())
        ind=e+e+e+e//2+e//4+e//4+e//2+e//2+cfg.cont_dim
        self.lstm=nn.LSTM(ind,cfg.hidden_dim,cfg.num_layers,dropout=cfg.dropout if cfg.num_layers>1 else 0,batch_first=True)
        self.trunk=nn.Sequential(nn.LayerNorm(cfg.hidden_dim),nn.Linear(cfg.hidden_dim,cfg.hidden_dim),nn.ReLU(),nn.Dropout(cfg.dropout))
        self.good=nn.Linear(cfg.hidden_dim,2); self.bypass=nn.Linear(cfg.hidden_dim,2); self.timing=nn.Linear(cfg.hidden_dim,n_time); self.delta=nn.Linear(cfg.hidden_dim,n_delta)
    def forward(self,x):
        z=torch.cat([self.pc(x["pc_id"]),self.dd(x["demand_delta_id"]),self.cd(x["candidate_delta_id"]),self.off(x["offset_id"]),self.hit(x["hit_id"].clamp(0,2)),self.acc(x["access_id"].clamp(0,1)),self.sem(x["semantic_id"]),self.th(x["timing_hist_id"]),self.cont(x["cont"])],dim=-1)
        o,_=self.lstm(z); h=self.trunk(o[:,-1,:]); return {"good":self.good(h),"bypass":self.bypass(h),"timing":self.timing(h),"delta":self.delta(h)}

def to_dev(x,y,dev): return ({k:v.to(dev,non_blocking=True) for k,v in x.items()},{k:v.to(dev,non_blocking=True) for k,v in y.items()})

def loss_fn(logits,y,losses,cfg):
    return cfg.loss_good*losses["good"](logits["good"],y["good"])+cfg.loss_bypass*losses["bypass"](logits["bypass"],y["bypass"])+cfg.loss_timing*losses["timing"](logits["timing"],y["timing"])+cfg.loss_delta*losses["delta"](logits["delta"],y["candidate_delta_id"])

def evaluate(model,loader,dev,cfg,losses=None):
    model.eval(); total=loss_sum=0; gtp=gfp=gfn=btp=bfp=bfn=emit=egood=edup=tok=dok=0
    with torch.no_grad():
        for x,y in loader:
            x,y=to_dev(x,y,dev); lg=model(x); bs=y["good"].numel(); total+=bs
            if losses: loss_sum+=float(loss_fn(lg,y,losses,cfg).cpu())*bs
            gp=torch.softmax(lg["good"],-1)[:,1]>=cfg.good_threshold; bp=torch.softmax(lg["bypass"],-1)[:,1]>=cfg.bypass_threshold
            tg=y["good"]==1; tb=y["bypass"]==1; dup=y["duplicate"]==1; selfpf=y["candidate_self"]==1; em=gp&(~bp)&(~selfpf)
            gtp+=(gp&tg).sum().item(); gfp+=(gp&~tg).sum().item(); gfn+=(~gp&tg).sum().item()
            btp+=(bp&tb).sum().item(); bfp+=(bp&~tb).sum().item(); bfn+=(~bp&tb).sum().item()
            emit+=em.sum().item(); egood+=(em&tg).sum().item(); edup+=(em&dup).sum().item(); tok+=(lg["timing"].argmax(-1)==y["timing"]).sum().item(); dok+=(lg["delta"].argmax(-1)==y["candidate_delta_id"]).sum().item()
    gp,gr,gf=prf(gtp,gfp,gfn); bp,br,bf=prf(btp,bfp,bfn)
    return {"loss":loss_sum/max(total,1) if losses else 0,"good_precision":gp,"good_recall":gr,"good_f1":gf,"bypass_precision":bp,"bypass_recall":br,"bypass_f1":bf,"emit_count":emit,"emit_good":egood,"emit_precision_good":egood/max(emit,1),"emit_duplicate_rate":edup/max(emit,1),"timing_acc":tok/max(total,1),"candidate_delta_top1":dok/max(total,1),"total":total}

def export_actions(model,loader,df,i2d,dev,cfg,out):
    rows=[]; model.eval()
    with torch.no_grad():
        for x,y in loader:
            x,y=to_dev(x,y,dev); lg=model(x); goodp=torch.softmax(lg["good"],-1)[:,1]; bypassp=torch.softmax(lg["bypass"],-1)[:,1]
            dprob=torch.softmax(lg["delta"],-1); dconf,did=dprob.max(-1); timing=lg["timing"].argmax(-1)
            for i in range(y["end_pos"].numel()):
                pos=int(y["end_pos"][i].cpu()); r=df.iloc[pos]; pdid=int(did[i].cpu()); pdelta=int(i2d.get(pdid,0)); gp=float(goodp[i].cpu()); bp=float(bypassp[i].cpu())
                line=int(r.line_addr); cand=int(r.candidate_line_addr); pfline=line+pdelta if cfg.use_pred_delta_address else cand; pfaddr=pfline*cfg.cache_line_bytes if cfg.use_pred_delta_address else int(r.candidate_addr)
                action="PREFETCH_DELTA" if gp>=cfg.good_threshold and bp<cfg.bypass_threshold and cand!=line else ("BYPASS_OR_LOW_PRIORITY_INSERT" if bp>=cfg.bypass_threshold else "INSERT_NORMAL_NO_PREFETCH")
                rows.append({"trace":r.trace,"event_id":int(r.event_id),"cycle_num":int(r.cycle_num),"pc_int":int(r.pc_int),"addr_int":int(r.addr_int),"line_addr":line,"candidate_addr":int(r.candidate_addr),"candidate_line_addr":cand,"candidate_delta":int(r.candidate_delta),"pred_delta_id":pdid,"pred_delta":pdelta,"pred_delta_conf":float(dconf[i].cpu()),"pred_good_prefetch_prob":gp,"pred_future_hit_prob":gp,"pred_bypass_prob":bp,"pred_timing_bin":int(timing[i].cpu()),"label_good_prefetch":int(r.good_prefetch_label),"outcome_useful":int(r.outcome_useful_int),"outcome_duplicate":int(r.outcome_duplicate_int),"nn_action":action,"prefetch_line_addr":int(pfline),"prefetch_addr":int(pfaddr)})
    out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); print(f"[export] {out} rows={len(rows):,}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trace",default="602.gcc_s-734B"); ap.add_argument("--events",type=Path); ap.add_argument("--artifact-dir",type=Path,default=Path("formal_NN_training/artifacts")); ap.add_argument("--max-rows",type=int,default=2_000_000); ap.add_argument("--seq-len",type=int,default=64); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--hidden-dim",type=int,default=128); ap.add_argument("--emb-dim",type=int,default=32); ap.add_argument("--top-delta-vocab",type=int,default=256); ap.add_argument("--good-threshold",type=float,default=0.50); ap.add_argument("--bypass-threshold",type=float,default=0.60); ap.add_argument("--num-workers",type=int,default=0); ap.add_argument("--use-pred-delta-address",action="store_true"); args=ap.parse_args()
    cfg=Cfg(trace=args.trace,max_rows=args.max_rows,seq_len=args.seq_len,epochs=args.epochs,batch_size=args.batch_size,hidden_dim=args.hidden_dim,emb_dim=args.emb_dim,top_delta_vocab=args.top_delta_vocab,good_threshold=args.good_threshold,bypass_threshold=args.bypass_threshold,num_workers=args.num_workers,use_pred_delta_address=args.use_pred_delta_address)
    set_seed(cfg.seed); dev="cuda" if torch.cuda.is_available() else "cpu"; print("[device]",dev)
    events=args.events or Path(f"formal_NN_training/data/generated/lstm_events_{cfg.trace}.csv"); outdir=args.artifact_dir; outdir.mkdir(parents=True,exist_ok=True)
    df,cont_cols,label_meta=build_df(events,cfg); top,d2i,i2d=delta_vocab(df.demand_delta.to_numpy(np.int64),df.candidate_delta.to_numpy(np.int64),cfg.top_delta_vocab); nd=cfg.top_delta_vocab+DELTA_OFFSET; nt=len(TIMING_EDGES)+1
    df["demand_delta_id"]=map_delta(df.demand_delta.to_numpy(np.int64),d2i); df["candidate_delta_id"]=map_delta(df.candidate_delta.to_numpy(np.int64),d2i); df["timing_hist_id"]=df.timing_label.astype(np.int64).clip(0,nt-1)
    pos=df.groupby("trace").cumcount().to_numpy(np.int64); valid=pos>=cfg.seq_len-1; train=np.zeros(len(df),bool); val=np.zeros(len(df),bool)
    for _,g in df.groupby("trace",sort=False):
        idx=g.index.to_numpy(); cut=int(len(idx)*cfg.train_fraction); train[idx[:cut]]=True; val[idx[cut:]]=True
    train_pos=np.where(valid&train)[0]; val_pos=np.where(valid&val)[0]; print(f"[split] train={len(train_pos):,} val={len(val_pos):,}")
    cont=df[cont_cols].to_numpy(np.float32); mean=cont[train].mean(0); std=cont[train].std(0)+1e-6; cont=((cont-mean)/std).astype(np.float32)
    feats={"pc_id":df.pc_id.to_numpy(np.int64),"demand_delta_id":df.demand_delta_id.to_numpy(np.int64),"candidate_delta_id":df.candidate_delta_id.to_numpy(np.int64),"offset_id":df.line_offset_in_page.to_numpy(np.int64),"hit_id":df.hit_int.to_numpy(np.int64).clip(0,2),"access_id":np.asarray(df.access_type,dtype=np.int64).clip(0,1),"semantic_id":np.asarray(df.semantic_id,dtype=np.int64),"timing_hist_id":df.timing_hist_id.to_numpy(np.int64),"cont":cont}
    labs={"good":df.good_prefetch_label.to_numpy(np.int64),"bypass":df.bypass_label.to_numpy(np.int64),"timing":df.timing_label.to_numpy(np.int64).clip(0,nt-1),"candidate_delta_id":df.candidate_delta_id.to_numpy(np.int64),"duplicate":df.outcome_duplicate_int.to_numpy(np.int64),"candidate_self":df.candidate_is_self.to_numpy(np.int64)}
    trds=SeqDS(train_pos,feats,labs,cfg.seq_len); vds=SeqDS(val_pos,feats,labs,cfg.seq_len); tr=DataLoader(trds,cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=(dev=="cuda")); va=DataLoader(vds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=(dev=="cuda"))
    model=Model(cfg,nd,nt,len(cont_cols)).to(dev); print("[params]",sum(p.numel() for p in model.parameters()))
    losses={"good":nn.CrossEntropyLoss(weight=class_weights(labs["good"][train_pos],2).to(dev)),"bypass":nn.CrossEntropyLoss(weight=class_weights(labs["bypass"][train_pos],2).to(dev)),"timing":nn.CrossEntropyLoss(weight=class_weights(labs["timing"][train_pos],nt).to(dev)),"delta":nn.CrossEntropyLoss()}
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(cfg.epochs,1)); best=-1; hist=[]; ckpt=outdir/"outcome_lstm_cache_action_predictor.pt"
    for ep in range(1,cfg.epochs+1):
        model.train(); t=time.time(); tot=seen=0
        for x,y in tr:
            x,y=to_dev(x,y,dev); lg=model(x); loss=loss_fn(lg,y,losses,cfg); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); bs=y["good"].numel(); tot+=float(loss.detach().cpu())*bs; seen+=bs
        sched.step(); m=evaluate(model,va,dev,cfg,losses); row={"epoch":ep,"train_loss":tot/max(seen,1),"sec":time.time()-t,**m}; hist.append(row); print("[epoch]",json.dumps(row))
        if m["good_f1"]>best: best=m["good_f1"]; torch.save({"model_state":model.state_dict(),"cfg":asdict(cfg),"cont_cols":cont_cols,"mean":mean.tolist(),"std":std.tolist(),"top_deltas":top},ckpt); print("[save]",ckpt)
    pd.DataFrame(hist).to_csv(outdir/"outcome_lstm_training_history.csv",index=False); model.load_state_dict(torch.load(ckpt,map_location=dev)["model_state"]); final=evaluate(model,va,dev,cfg,losses)
    full_pos=np.where(valid)[0]; full=DataLoader(SeqDS(full_pos,feats,labs,cfg.seq_len),cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=(dev=="cuda")); actions=outdir/"outcome_lstm_cache_actions.csv"; export_actions(model,full,df,i2d,dev,cfg,actions); compat=outdir/"full_lstm_cache_actions.csv"; pd.read_csv(actions).to_csv(compat,index=False)
    summary={"config":asdict(cfg),"label_meta":label_meta,"final_val_metrics":final,"events":str(events),"artifacts":{"checkpoint":str(ckpt),"actions":str(actions),"compat_actions":str(compat)}}; json.dump(summary,open(outdir/"outcome_lstm_summary.json","w"),indent=2); json.dump({"PAD_ID":PAD_ID,"UNK_ID":UNK_ID,"DELTA_OFFSET":DELTA_OFFSET,"top_deltas":top,"delta_to_id":{str(k):int(v) for k,v in d2i.items()},"id_to_delta":{str(k):int(v) for k,v in i2d.items()}},open(outdir/"outcome_delta_vocab.json","w"),indent=2); print("[done]",outdir/"outcome_lstm_summary.json")
if __name__=="__main__": main()
