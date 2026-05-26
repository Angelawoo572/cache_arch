"""Export GRU V9 threshold/degree sweep prefetch lists.

Run this INSIDE the already-executed `notebook/gru_sweep_v9.ipynb` runtime,
after the model has been trained and the V9 arrays exist.

In Colab / IPython, use:

    %run -i scripts/gru_v9_export_decode_sweep.py

or copy this file into the Colab working directory and run:

    %run -i gru_v9_export_decode_sweep.py

Why `%run -i`?
    This script intentionally uses variables already created by the notebook:
    `model`, `DEVICE`, `X_d`, `X_pc`, `X_pcd`, `Idx`, `Cur`, `i_va`,
    `TRACE_CSV`, `DELTA_RANGE`, and `LINE_BITS`.

It scores the V9 test slice once, writes a candidate CSV with probabilities,
and then emits multiple ChampSim list-replayer files:

    prefetch_list_GRU_V9_<trace>_th030_deg1.txt
    prefetch_list_GRU_V9_<trace>_th030_deg2.txt
    ...

Default sweep:
    thresholds = 0.30, 0.50, 0.70, 0.90
    max_degree = 1, 2, 4

Override with environment variables before running:

    SWEEP_THRESHOLDS="0.30 0.50 0.70 0.90" \
    SWEEP_DEGREES="1 2 4" \
    SWEEP_OUTPUT_DIR="/content" \
    %run -i scripts/gru_v9_export_decode_sweep.py
"""

from __future__ import annotations

import os
import re
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def _parse_floats(value: str) -> List[float]:
    toks = re.split(r"[ ,]+", value.strip())
    return [float(x) for x in toks if x]


def _parse_ints(value: str) -> List[int]:
    toks = re.split(r"[ ,]+", value.strip())
    return [int(x) for x in toks if x]


def _threshold_tag(th: float) -> str:
    # 0.30 -> th030, 0.9 -> th090, 1.00 -> th100
    return f"th{int(round(th * 100)):03d}"


def _infer_short_trace_tag(trace_csv: str) -> str:
    """Return gcc_s-734B from /content/access_trace.602.gcc_s-734B.csv.

    Falls back to the notebook's TRACE_TAG if parsing fails.
    """
    name = Path(str(trace_csv)).name
    m = re.match(r"access_trace\.\d+\.(.+?)\.csv$", name)
    if m:
        return m.group(1)
    m = re.match(r"access_trace\.(.+?)\.csv$", name)
    if m:
        return m.group(1)
    if "TRACE_TAG" in globals():
        return str(globals()["TRACE_TAG"])
    return Path(str(trace_csv)).stem


def _require_notebook_globals() -> None:
    required = [
        "model", "DEVICE", "X_d", "X_pc", "X_pcd", "Idx", "Cur", "i_va",
        "TRACE_CSV", "DELTA_RANGE", "LINE_BITS",
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Missing notebook variables: " + ", ".join(missing) + "\n"
            "Run this script with `%run -i` after executing the V9 training cells."
        )


def _score_candidates(max_degree: int, batch_size: int) -> Tuple[List[Dict[str, object]], int, np.ndarray]:
    """Score the V9 test split once and keep top `max_degree` candidate bits."""
    CENTER = int(globals()["DELTA_RANGE"])
    line_bits = int(globals()["LINE_BITS"])
    device = globals()["DEVICE"]
    model = globals()["model"]

    X_d = globals()["X_d"]
    X_pc = globals()["X_pc"]
    X_pcd = globals()["X_pcd"]
    Idx = globals()["Idx"]
    Cur = globals()["Cur"]
    i_va = int(globals()["i_va"])

    Xd_te = X_d[i_va:]
    Xpc_te = X_pc[i_va:]
    Xpcd_te = X_pcd[i_va:]
    Idx_te = Idx[i_va:]
    Cur_te = Cur[i_va:]

    rows: List[Dict[str, object]] = []
    all_top1_probs: List[np.ndarray] = []

    model.to(device).eval()
    with torch.no_grad():
        for i in range(0, len(Idx_te), batch_size):
            d = torch.from_numpy(Xd_te[i:i + batch_size]).to(device)
            p = torch.from_numpy(Xpc_te[i:i + batch_size]).to(device)
            pd_ = torch.from_numpy(Xpcd_te[i:i + batch_size]).to(device)

            logits = model(d, p, pd_)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            top_idx = np.argsort(-probs, axis=1)[:, :max_degree]
            top_prob = np.take_along_axis(probs, top_idx, axis=1)
            all_top1_probs.append(top_prob[:, 0])

            idxs = Idx_te[i:i + batch_size]
            curs = Cur_te[i:i + batch_size]

            for j in range(probs.shape[0]):
                idx = int(idxs[j])
                cur = int(curs[j])
                for rank, (ti, prob) in enumerate(zip(top_idx[j], top_prob[j]), start=1):
                    d_lines = int(ti) - CENTER
                    if d_lines == 0:
                        continue
                    pf_addr = (cur + (d_lines << line_bits)) & 0xFFFFFFFFFFFFFFFF
                    if pf_addr <= 0:
                        continue
                    rows.append({
                        "idx": idx,
                        "pf_addr": pf_addr,
                        "prob": float(prob),
                        "d_lines": d_lines,
                        "rank": rank,
                        "cur_addr": cur,
                    })

    top1 = np.concatenate(all_top1_probs) if all_top1_probs else np.array([])
    return rows, len(Idx_te), top1


def _write_candidates_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["idx", "pf_addr_hex", "prob", "d_lines", "rank", "cur_addr_hex"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "idx": r["idx"],
                "pf_addr_hex": f"0x{int(r['pf_addr']):x}",
                "prob": f"{float(r['prob']):.8f}",
                "d_lines": r["d_lines"],
                "rank": r["rank"],
                "cur_addr_hex": f"0x{int(r['cur_addr']):x}",
            })


def _write_prefetch_list(path: Path, rows: List[Dict[str, object]], threshold: float, degree: int) -> Tuple[int, int]:
    n_emit = 0
    idx_with_emit = set()
    with path.open("w") as fh:
        for r in rows:
            if int(r["rank"]) > degree:
                continue
            if float(r["prob"]) < threshold:
                continue
            fh.write(f"{int(r['idx'])} 0x{int(r['pf_addr']):x}\n")
            n_emit += 1
            idx_with_emit.add(int(r["idx"]))
    return n_emit, len(idx_with_emit)


def main() -> None:
    _require_notebook_globals()

    thresholds = _parse_floats(os.environ.get("SWEEP_THRESHOLDS", "0.30 0.50 0.70 0.90"))
    degrees = _parse_ints(os.environ.get("SWEEP_DEGREES", "1 2 4"))
    if not thresholds or not degrees:
        raise ValueError("SWEEP_THRESHOLDS and SWEEP_DEGREES must be non-empty")

    out_dir = Path(os.environ.get("SWEEP_OUTPUT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    short_tag = _infer_short_trace_tag(str(globals()["TRACE_CSV"]))
    max_degree = max(degrees)
    batch_size = int(os.environ.get("SWEEP_BATCH", "4096"))

    print("============================================================")
    print("GRU V9 decode sweep export")
    print("============================================================")
    print(f"trace_csv     : {globals()['TRACE_CSV']}")
    print(f"trace tag     : {short_tag}")
    print(f"thresholds    : {thresholds}")
    print(f"degrees       : {degrees}")
    print(f"output dir    : {out_dir}")
    print(f"batch size    : {batch_size}")
    print("============================================================")

    rows, n_scored, top1_probs = _score_candidates(max_degree=max_degree, batch_size=batch_size)

    cand_path = out_dir / f"gru_v9_decode_candidates_{short_tag}_top{max_degree}.csv"
    _write_candidates_csv(cand_path, rows)
    print(f"[candidates] wrote {cand_path}")
    print(f"             scored accesses = {n_scored:,}")
    print(f"             candidate rows  = {len(rows):,}")

    if len(top1_probs):
        print("[top1 prob percentiles]")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{p:>2d} = {np.percentile(top1_probs, p):.4f}")

    summary_path = out_dir / f"gru_v9_decode_sweep_{short_tag}.csv"
    with summary_path.open("w", newline="") as sfh:
        writer = csv.DictWriter(
            sfh,
            fieldnames=[
                "trace_tag", "threshold", "degree", "prefetch_list", "n_scored",
                "n_emitted", "n_trigger_accesses", "trigger_access_frac", "avg_degree",
            ],
        )
        writer.writeheader()

        for th in thresholds:
            th_tag = _threshold_tag(th)
            for deg in degrees:
                out_path = out_dir / f"prefetch_list_GRU_V9_{short_tag}_{th_tag}_deg{deg}.txt"
                n_emit, n_trigger = _write_prefetch_list(out_path, rows, threshold=th, degree=deg)
                writer.writerow({
                    "trace_tag": short_tag,
                    "threshold": th,
                    "degree": deg,
                    "prefetch_list": out_path.name,
                    "n_scored": n_scored,
                    "n_emitted": n_emit,
                    "n_trigger_accesses": n_trigger,
                    "trigger_access_frac": f"{n_trigger / max(1, n_scored):.6f}",
                    "avg_degree": f"{n_emit / max(1, n_scored):.6f}",
                })
                print(
                    f"[list] {out_path.name:48s} "
                    f"emitted={n_emit:9,d}  trigger_accesses={n_trigger:9,d}  "
                    f"avg_degree={n_emit / max(1, n_scored):.3f}"
                )

    print(f"[summary] wrote {summary_path}")
    print()
    print("Copy the generated prefetch_list_GRU_V9_*_th*_deg*.txt files to:")
    print("  results/generated/prefetch_lists/")
    print("Then run on the lab machine, for example:")
    print(f"  TRACE=<full trace name> bash scripts/run_gru_v9_decode_sweep.sh")


main()
