#!/usr/bin/env python3
"""Append outcome-aware runner cells to LSTM_cache_action_predictor.ipynb.

Append-only: existing cells are preserved. The added section uses
formal_NN_training/scripts/train_lstm_cache_action.py as the training entry.
"""
from __future__ import annotations

import json
from pathlib import Path

MARKER = "OUTCOME_AWARE_LSTM_APPEND_V1"
NB = Path("formal_NN_training/LSTM_cache_action_predictor.ipynb")


def md(text: str, cid: str):
    return {"cell_type": "markdown", "metadata": {"id": cid}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str, cid: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {"id": cid}, "outputs": [], "source": [line + "\n" for line in text.splitlines()]}


def main():
    nb = json.loads(NB.read_text())
    if MARKER in json.dumps(nb):
        print("[skip] outcome-aware cells already appended")
        return
    cells = [
        md("""
## Z. Outcome-aware SPP-assisted LSTM cache-action runner  <!-- OUTCOME_AWARE_LSTM_APPEND_V1 -->

This appended section keeps the older notebook cells above for comparison. The real research path now follows the project diagram:

```text
SPP = candidate + context + supervision
LSTM = stateful cache-action learner
main label = outcome_useful == 1 and outcome_duplicate == 0
```

The notebook calls `formal_NN_training/scripts/train_lstm_cache_action.py` for training. This makes Colab still useful for training, but keeps the model code in a runnable script so cluster and Colab use the same logic.
""".strip(), "outcome_md_intro"),
        code(r'''
from pathlib import Path
import os, subprocess, json, csv, collections

REPO_ROOT = Path("/content/cache_arch") if Path("/content/cache_arch").exists() else Path.cwd()
if not (REPO_ROOT / "formal_NN_training").exists() and REPO_ROOT.name == "formal_NN_training":
    REPO_ROOT = REPO_ROOT.parent
os.chdir(REPO_ROOT)

TRACE = os.environ.get("TRACE", "602.gcc_s-734B")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "2000000"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
GOOD_TH = float(os.environ.get("GOOD_TH", "0.50"))
BYPASS_TH = float(os.environ.get("BYPASS_TH", "0.60"))

EVENTS = Path(f"formal_NN_training/data/generated/lstm_events_{TRACE}.csv")
ART = Path("formal_NN_training/artifacts")
ART.mkdir(parents=True, exist_ok=True)

print("cwd=", Path.cwd())
print("TRACE=", TRACE)
print("EVENTS=", EVENTS)
print("MAX_ROWS=", MAX_ROWS, "SEQ_LEN=", SEQ_LEN, "EPOCHS=", EPOCHS)
'''.strip(), "outcome_cfg"),
        code(r'''
restore = Path("formal_NN_training/scripts/00_restore_colab_uploaded_data.sh")
if not EVENTS.exists() and restore.exists():
    subprocess.run(["bash", str(restore)], check=True, env={**os.environ, "TRACE": TRACE})
if not EVENTS.exists():
    raise FileNotFoundError(f"Missing {EVENTS}. Run 01_run_spp_trace_dump.sh on cluster or restore split upload in Colab.")
subprocess.run(["ls", "-lh", str(EVENTS)], check=True)
'''.strip(), "outcome_restore"),
        code(r'''
cmd = [
    "python3", "formal_NN_training/scripts/train_lstm_cache_action.py",
    "--trace", TRACE,
    "--events", str(EVENTS),
    "--max-rows", str(MAX_ROWS),
    "--seq-len", str(SEQ_LEN),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--hidden-dim", os.environ.get("HIDDEN_DIM", "128"),
    "--emb-dim", os.environ.get("EMB_DIM", "32"),
    "--good-threshold", str(GOOD_TH),
    "--bypass-threshold", str(BYPASS_TH),
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
'''.strip(), "outcome_train"),
        code(r'''
summary_path = ART / "outcome_lstm_summary.json"
action_path = ART / "full_lstm_cache_actions.csv"
with open(summary_path) as f:
    summary = json.load(f)
print("label_meta")
print(json.dumps(summary.get("label_meta", {}), indent=2))
print("final_val_metrics")
print(json.dumps(summary.get("final_val_metrics", {}), indent=2))

ctr = collections.Counter(); rows = 0
with open(action_path, newline="") as f:
    for row in csv.DictReader(f):
        rows += 1
        ctr[row.get("nn_action", "")] += 1
print("action rows=", rows)
print("nn_action distribution=", dict(ctr))
'''.strip(), "outcome_inspect"),
        code(r'''
pfetch = Path("formal_NN_training/results/replay_compare/prefetch_lists") / f"prefetch_list_{TRACE}_outcome_lstm.txt"
pfetch.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    "python3", "formal_NN_training/scripts/02_actions_to_prefetch_list.py",
    "--actions", "formal_NN_training/artifacts/full_lstm_cache_actions.csv",
    "--out", str(pfetch),
    "--policy", "action",
    "--prefetch-threshold", str(GOOD_TH),
    "--bypass-threshold", str(BYPASS_TH),
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)
subprocess.run(["wc", "-l", str(pfetch)], check=True)
subprocess.run(["head", str(pfetch)], check=False)
'''.strip(), "outcome_to_prefetch"),
        code(r'''
replayer = Path("external/ChampSim/bin/champsim.replayer")
trace_file = Path(f"traces/{TRACE}.champsimtrace.xz")
if replayer.exists() and trace_file.exists():
    env = {**os.environ, "TRACE": TRACE, "POLICY": "action", "PREFETCH_THRESHOLD": str(GOOD_TH), "BYPASS_THRESHOLD": str(BYPASS_TH)}
    subprocess.run(["bash", "formal_NN_training/scripts/03_run_lstm_replay.sh"], check=True, env=env)
else:
    print("[skip] no ChampSim replayer/trace here. Train/export in Colab; replay on cluster.")
'''.strip(), "outcome_replay"),
        code(r'''
# Pack/split large results. Keep huge CSV/PT out of normal git commits.
import gzip, shutil
PACK_DIR = Path(f"formal_NN_training/artifacts/packed/{TRACE.split('.')[0]}")
PACK_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_SIZE = os.environ.get("SPLIT_SIZE", "90m")
for src in [ART / "full_lstm_cache_actions.csv", ART / "outcome_lstm_cache_actions.csv"]:
    if src.exists():
        gz = PACK_DIR / f"{src.name}.gz"
        with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        subprocess.run(["split", "-b", SPLIT_SIZE, str(gz), str(gz) + ".part_"], check=True)
        subprocess.run(["ls", "-lh", str(gz)], check=True)
print("packed dir=", PACK_DIR)
'''.strip(), "outcome_pack_split"),
        code(r'''
# Git note: commit source/notebook/docs only. Do not commit huge unsplit artifacts.
subprocess.run(["git", "status", "--short"], check=True)
print("Suggested commit:")
print("git add formal_NN_training/LSTM_cache_action_predictor.ipynb formal_NN_training/scripts/train_lstm_cache_action.py formal_NN_training/scripts/02_actions_to_prefetch_list.py formal_NN_training/README_LSTM_cache_action_predictor.md")
print("git commit -m 'Add outcome-aware LSTM cache-action notebook flow'")
print("git push")
'''.strip(), "outcome_git_note"),
    ]
    nb["cells"].extend(cells)
    NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] appended {len(cells)} cells to {NB}")

if __name__ == "__main__":
    main()
