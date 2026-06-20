#!/usr/bin/env python3
"""Patch the oracle-LSTM notebook to reject invalid reconstructed addresses.

Existing rich exports may contain a rare negative ``prefetch_addr`` when the
page-delta head predicts a legal vocabulary class that moves below page zero.
Those candidates cannot be sent to a hardware prefetcher. Script 10 filters
legacy files at replay conversion; this patch installs the same check upstream
in the notebook so future fair/dedup exports never contain them.

Usage:
  python3 formal_NN_training/scripts/12_patch_oracle_notebook_address_guard.py \
    --notebook formal_NN_training/LSTM/notebooks/LSTM_base_independent_oracle_prefetcher.ipynb

It is idempotent and writes a .bak backup once.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

OLD = '''    cur_page=df["page"].to_numpy(np.int64); PL=RCFG["page_lines"]\n    pred_line=(cur_page+pred_pd)*PL + ofp.astype(np.int64)\n    exp=pd.DataFrame(dict(order=df["_order"].to_numpy(np.int64), pc=df["_pc"].to_numpy(np.int64),\n        line=df["_line"].to_numpy(np.int64), issue_prob=ip, addr_conf=addr_conf,\n        pred_page_delta=pred_pd, pred_offset=ofp,\n        pred_line=pred_line, prefetch_addr=pred_line*RCFG["cache_line_bytes"]))\n'''

NEW = '''    # Hardware/replay safety: a learned page_delta can mathematically point below\n    # page zero (or beyond signed-int64 address space). It is not a valid prefetch\n    # candidate, so mark it invalid and remove it before both undedup and LRU export.\n    cur_page=df["page"].to_numpy(np.int64); PL=RCFG["page_lines"]\n    line_bytes=int(RCFG["cache_line_bytes"])\n    max_line=np.iinfo(np.int64).max // line_bytes\n    max_page=(max_line-(PL-1))//PL\n    valid_page=(pred_pd >= -cur_page) & (pred_pd <= (max_page-cur_page))\n    pred_line=np.full(N,-1,dtype=np.int64)\n    pred_line[valid_page]=(cur_page[valid_page]+pred_pd[valid_page])*PL + ofp[valid_page].astype(np.int64)\n    prefetch_addr=np.full(N,-1,dtype=np.int64)\n    prefetch_addr[valid_page]=pred_line[valid_page]*line_bytes\n    exp=pd.DataFrame(dict(order=df["_order"].to_numpy(np.int64), pc=df["_pc"].to_numpy(np.int64),\n        line=df["_line"].to_numpy(np.int64), issue_prob=ip, addr_conf=addr_conf,\n        pred_page_delta=pred_pd, pred_offset=ofp, valid_prefetch_addr=valid_page,\n        pred_line=pred_line, prefetch_addr=prefetch_addr))\n'''

OLD_KEEP = '''    keep=conf_keep&known_pd&(~self_line)                 # drop unknown page_delta class (no silent self-page)\n'''
NEW_KEEP = '''    keep=conf_keep&known_pd&(~self_line)&valid_page     # also drop invalid/underflow reconstructed addresses\n    res["invalid_addr_candidates"]=int((~valid_page).sum())\n    res["invalid_addr_dropped"]=int((conf_keep&known_pd&(~self_line)&(~valid_page)).sum())\n'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True, type=Path)
    args = ap.parse_args()
    p = args.notebook
    nb = json.loads(p.read_text())

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "valid_prefetch_addr" in source and "invalid_addr_dropped" in source:
            continue
        if OLD in source:
            source = source.replace(OLD, NEW, 1)
            changed = True
        if OLD_KEEP in source:
            source = source.replace(OLD_KEEP, NEW_KEEP, 1)
            changed = True
        cell["source"] = source.splitlines(True)

    if not changed:
        text = p.read_text()
        if "valid_prefetch_addr" in text and "invalid_addr_dropped" in text:
            print("[skip] address guard already present:", p)
            return 0
        raise SystemExit("[error] expected export block not found; notebook version differs")

    backup = p.with_suffix(p.suffix + ".bak")
    if not backup.exists():
        backup.write_text(p.read_text())
    p.write_text(json.dumps(nb, indent=1) + "\n")
    print("[patched]", p)
    print("[backup]", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
