#!/usr/bin/env python3
"""One-shot migration from numbered formal-NN scripts to a role-based layout.

Run only from a clean tracked checkout on main:
  python3 formal_NN_training/tools/maintenance/apply_role_layout.py --apply

Untracked experiment outputs are intentionally preserved.  The script performs
all git moves, rewrites active callers, validates syntax, commits, pushes main,
and then removes itself from the repository.
"""
from __future__ import print_function

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SELF = Path(__file__).resolve().relative_to(ROOT)

MOVES = [
    ("formal_NN_training/scripts/01_parse_prefetch_behavior_audit.py", "formal_NN_training/tools/analysis/parse_champsim_stats.py"),
    ("formal_NN_training/scripts/02_patch_pythia_demand_logger.sh", "formal_NN_training/scripts/build/patch_demand_logger.sh"),
    ("formal_NN_training/scripts/03_collect_no_pref_demand_events.sh", "formal_NN_training/scripts/run/collect_oracle_data.sh"),
    ("formal_NN_training/scripts/05_build_standalone_oracle_dataset.py", "formal_NN_training/tools/data/build_oracle_dataset.py"),
    ("formal_NN_training/scripts/06_install_keyed_listreplayer.sh", "formal_NN_training/scripts/build/build_keyed_listreplayer.sh"),
    ("formal_NN_training/scripts/07_prepare_keyed_replay_input.py", "formal_NN_training/tools/data/prepare_keyed_replay_input.py"),
    ("formal_NN_training/scripts/08_run_standalone_lstm_replay.sh", "formal_NN_training/scripts/run/replay_exports.sh"),
    ("formal_NN_training/scripts/09_parse_standalone_lstm_replay.py", "formal_NN_training/tools/analysis/summarize_replay.py"),
    ("formal_NN_training/scripts/10_profile_champsim_trace.py", "formal_NN_training/tools/analysis/profile_trace.py"),
    ("formal_NN_training/scripts/11_run_prefetch_event_attribution.sh", "formal_NN_training/scripts/run/prefetch_campaign.sh"),
    ("formal_NN_training/scripts/12_analyze_prefetch_event_attribution.py", "formal_NN_training/tools/analysis/analyze_event_attribution.py"),
    ("formal_NN_training/scripts/13_build_cache_capacity_variant.sh", "formal_NN_training/scripts/build/build_cache_capacity_variant.sh"),
    ("formal_NN_training/scripts/14_build_base_candidate_table.py", "formal_NN_training/archive/base_aware/build_base_candidate_table.py"),
    ("formal_NN_training/scripts/14_run_cache_capacity_sweep.sh", "formal_NN_training/scripts/run/capacity_sweep.sh"),
    ("formal_NN_training/scripts/15_summarize_prefetch_evidence.py", "formal_NN_training/tools/analysis/summarize_evidence.py"),
    ("formal_NN_training/scripts/16_build_trace_dependency_features.py", "formal_NN_training/tools/dependency/build_dependency_features.py"),
    ("formal_NN_training/scripts/17_prepare_v3_9_605_dependency_sidecar.sh", "formal_NN_training/scripts/dependency/prepare_605_sidecar.sh"),
    ("formal_NN_training/scripts/19_build_oracle_ceiling_lists.py", "formal_NN_training/tools/analysis/build_ceiling_lists.py"),
    ("formal_NN_training/scripts/20_run_oracle_ceiling_replay.sh", "formal_NN_training/scripts/run/oracle_ceiling_replay.sh"),
    ("formal_NN_training/scripts/21_join_decision_ledger_attribution.py", "formal_NN_training/tools/analysis/join_ledger_attribution.py"),
    ("formal_NN_training/scripts/22_resource_summary.py", "formal_NN_training/tools/analysis/summarize_resource_pressure.py"),
    ("formal_NN_training/scripts/25_build_v4_1_notebook.py", "formal_NN_training/tools/notebook/build_v4_1_notebook.py"),
]

PATH_MAP = dict(MOVES)
LEGACY_MOVES = [
    ("formal_NN_training/LSTM/draft", "formal_NN_training/archive/legacy_action_predictor/draft"),
    ("formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor.ipynb", "formal_NN_training/archive/legacy_action_predictor/notebooks/LSTM_cache_action_predictor.ipynb"),
    ("formal_NN_training/LSTM/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb", "formal_NN_training/archive/legacy_action_predictor/notebooks/LSTM_cache_action_predictor_SPP_LSTM_direct_hybrid.ipynb"),
]


def run(*args):
    subprocess.run(list(args), cwd=str(ROOT), check=True)


def output(*args):
    return subprocess.check_output(list(args), cwd=str(ROOT)).decode().strip()


def write(path, text):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def replace_once(text, old, new, context):
    if old not in text:
        raise RuntimeError("expected text missing in {}".format(context))
    return text.replace(old, new, 1)


def rewrite_tracked_paths():
    files = output("git", "ls-files", "-z").split("\0")
    for rel in files:
        if not rel:
            continue
        path = ROOT / rel
        if path.suffix not in (".py", ".sh", ".md", ".ipynb"):
            continue
        if rel.startswith("formal_NN_training/archive/legacy_action_predictor/"):
            continue
        text = path.read_text()
        changed = text
        for old, new in PATH_MAP.items():
            changed = changed.replace(old, new)
        if changed != text:
            path.write_text(changed)


def fix_moved_shell_roots():
    rels = [
        "formal_NN_training/scripts/run/collect_oracle_data.sh",
        "formal_NN_training/scripts/run/replay_exports.sh",
        "formal_NN_training/scripts/run/prefetch_campaign.sh",
        "formal_NN_training/scripts/run/capacity_sweep.sh",
        "formal_NN_training/scripts/run/oracle_ceiling_replay.sh",
        "formal_NN_training/scripts/build/build_keyed_listreplayer.sh",
        "formal_NN_training/scripts/build/build_cache_capacity_variant.sh",
    ]
    old = '$(dirname "${BASH_SOURCE[0]}")/../..'
    new = '$(dirname "${BASH_SOURCE[0]}")/../../..'
    for rel in rels:
        path = ROOT / rel
        text = path.read_text()
        if old not in text:
            raise RuntimeError("missing ROOT expression in {}".format(rel))
        path.write_text(text.replace(old, new, 1))


def fix_collection_skip():
    path = ROOT / "formal_NN_training/scripts/run/collect_oracle_data.sh"
    text = path.read_text()
    old = '''  if [[ "$FORCE" != "1" && -s "$out" && -s "$log" ]]; then
    echo "[skip] $trace"
    return 0
  fi
  echo "[run] $trace"
'''
    new = '''  if [[ "$FORCE" != "1" && -s "$out" && gzip -t "$out" >/dev/null 2>&1 && grep -q '^Core_0_IPC ' "$log" ]]; then
    echo "[skip] $trace"
    return 0
  fi
  rm -f "$raw" "$out"
  echo "[run] $trace"
'''
    path.write_text(replace_once(text, old, new, str(path)))


def fix_campaign_resolver():
    path = ROOT / "formal_NN_training/scripts/run/prefetch_campaign.sh"
    text = path.read_text()
    marker = 'NORMAL_PARSER="$ROOT/formal_NN_training/tools/analysis/parse_champsim_stats.py"'
    text = replace_once(text, marker, marker + '\nPLAN_RESOLVER="$ROOT/formal_NN_training/scripts/replay/resolve_replay_plan.py"', str(path))
    old = '[[ -f "$NORMAL_PARSER" ]] || { echo "[error] missing normal parser: $NORMAL_PARSER" >&2; exit 2; }'
    new = '[[ -f "$NORMAL_PARSER" && -f "$PLAN_RESOLVER" ]] || { echo "[error] missing normal parser or plan resolver" >&2; exit 2; }'
    text = replace_once(text, old, new, str(path))
    pattern = r'plan_entries\(\) \{.*?\n\}\n\nbuild_all\(\) \{'
    replacement = '''plan_entries() {
  local plan="$1" root="$2" out="$3"
  python3 "$PLAN_RESOLVER" --plan "$plan" --root "$root" --out "$out"
}

build_all() {'''
    text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError("could not replace replay-plan parser in {}".format(path))
    path.write_text(text)


def fix_replay_summary_resolver():
    path = ROOT / "formal_NN_training/tools/analysis/summarize_replay.py"
    text = path.read_text()
    text = replace_once(text, 'import re\nfrom pathlib import Path', 'import re\nimport sys\nfrom pathlib import Path', str(path))
    old = '''SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pythia_stats", str(SCRIPT_DIR / "01_parse_prefetch_behavior_audit.py"))'''
    new = '''SCRIPT_DIR = Path(__file__).resolve().parent
FORMAL_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(FORMAL_DIR / "scripts/replay"))
from resolve_replay_plan import read_plan
spec = importlib.util.spec_from_file_location("pythia_stats", str(SCRIPT_DIR / "parse_champsim_stats.py"))'''
    text = replace_once(text, old, new, str(path))
    pattern = r'def load_plan\(path, plan_root\):.*?\n\n\ndef enrich_row'
    text, count = re.subn(pattern, 'def load_plan(path, plan_root):\n    return read_plan(path, plan_root)\n\n\ndef enrich_row', text, flags=re.S)
    if count != 1:
        raise RuntimeError("could not replace replay-summary plan loader")
    path.write_text(text)


def fix_event_analysis_resolver():
    path = ROOT / "formal_NN_training/tools/analysis/analyze_event_attribution.py"
    text = path.read_text()
    text = replace_once(text, 'import json\nfrom collections', 'import json\nimport sys\nfrom collections', str(path))
    anchor = 'from pathlib import Path\n\n\n'
    insert = '''from pathlib import Path

FORMAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORMAL_DIR / "scripts/replay"))
from resolve_replay_plan import read_plan


'''
    text = replace_once(text, anchor, insert, str(path))
    pattern = r'def parse_plan\(path, plan_root\):.*?\n\n\ndef main'
    replacement = '''def parse_plan(path, plan_root):
    out = []
    for row in read_plan(path, plan_root):
        out.append({"label": row["tag"], "trace": row["trace"], "artifact_dir": None, "rich_list": Path(row["rich_list"])})
    return out


def main'''
    text, count = re.subn(pattern, replacement, text, flags=re.S)
    if count != 1:
        raise RuntimeError("could not replace event-analysis plan loader")
    path.write_text(text)


def fix_notebook_materializer():
    path = ROOT / "formal_NN_training/tools/notebook/build_v4_1_notebook.py"
    text = path.read_text()
    old = 'Path(__file__).resolve().parents[1] / "LSTM/notebooks/v4_1_extension.py.gz.b64"'
    new = 'Path(__file__).resolve().parents[2] / "LSTM/notebooks/v4_1_extension.py.gz.b64"'
    path.write_text(replace_once(text, old, new, str(path)))


def fix_ceiling_contracts():
    path = ROOT / "formal_NN_training/scripts/run/oracle_ceiling_replay.sh"
    text = path.read_text()
    old = '''    ledger=$(ls "$LEDGER_DIR"/decision_ledger_${trace}_*_full_candidates.csv.gz 2>/dev/null | head -n 1 || true)
    [[ -n "$ledger" ]] || ledger=$(ls "$LEDGER_DIR"/decision_ledger_${trace}_*_val_candidates.csv.gz 2>/dev/null | head -n 1 || true)
    [[ -n "$ledger" ]] || { echo "missing candidate ledger for $trace" >&2; exit 2; }
    args+=(--ledger-candidates "$ledger" --min-lead-bin "$MIN_LEAD_BIN")
    [[ "$REQUIRE_FULL_LEDGER" == "1" ]] && args+=(--require-full-coverage)'''
    new = '''    ledger=$(ls "$LEDGER_DIR"/decision_ledger_${trace}_*_full_candidates.csv.gz 2>/dev/null | head -n 1 || true)
    [[ -n "$ledger" ]] || { echo "missing full candidate ledger for $trace" >&2; exit 2; }
    args+=(--ledger-candidates "$ledger" --min-lead-bin "$MIN_LEAD_BIN" --require-full-coverage)'''
    path.write_text(replace_once(text, old, new, str(path)))

    path = ROOT / "formal_NN_training/tools/analysis/build_ceiling_lists.py"
    text = path.read_text()
    old = '''        for raw in reader:
            if str(raw.get("trace") or "") != trace:
                raise ValueError("candidate ledger trace mismatch: {}".format(raw.get("trace")))
            if not as_int(raw.get("candidate_valid"), "candidate_valid"):
                continue
            future_label = as_int(raw.get("future_label"), "future_label")
            if future_label < min_lead_bin:
                continue
            demand_idx = as_int(raw.get("demand_idx"), "demand_idx")
            pc = as_int(raw.get("pc"), "pc")
            line = as_int(raw.get("line"), "line")
            trigger = by_identity.get((demand_idx, pc, line))
            if trigger is None:
                unmatched += 1
                continue
            ledger_events.add(demand_idx)
            target_line = as_int(raw.get("candidate_line"), "candidate_line")
'''
    new = '''        for raw in reader:
            if str(raw.get("trace") or "") != trace:
                raise ValueError("candidate ledger trace mismatch: {}".format(raw.get("trace")))
            demand_idx = as_int(raw.get("demand_idx"), "demand_idx")
            pc = as_int(raw.get("pc"), "pc")
            line = as_int(raw.get("line"), "line")
            trigger = by_identity.get((demand_idx, pc, line))
            if trigger is None:
                unmatched += 1
                continue
            ledger_events.add(demand_idx)
            if not as_int(raw.get("candidate_valid"), "candidate_valid"):
                continue
            future_label = as_int(raw.get("future_label"), "future_label")
            if future_label < min_lead_bin:
                continue
            target_line = as_int(raw.get("candidate_line"), "candidate_line")
'''
    path.write_text(replace_once(text, old, new, str(path)))


def write_docs():
    write("formal_NN_training/scripts/README.md", """# Executable experiment drivers

The active entrypoints are organized by role; historical numeric filenames are
not part of the active pipeline.

## build

- `build/patch_demand_logger.sh`
- `build/build_keyed_listreplayer.sh`
- `build/build_cache_capacity_variant.sh`

## run

- `run/collect_oracle_data.sh`
- `run/replay_exports.sh`
- `run/prefetch_campaign.sh`
- `run/capacity_sweep.sh`
- `run/oracle_ceiling_replay.sh`

## shared helpers

- `replay/resolve_replay_plan.py`
- `replay/verify_same_binary_no_pref.py`
- `dependency/prepare_605_sidecar.sh`

`replay_exports.sh` performs keyed replay and same-binary validation.
`prefetch_campaign.sh` runs normal baselines and optional event evidence.
They have distinct simulator contracts and are deliberately separate.
""")
    write("formal_NN_training/tools/README.md", """# Deterministic tools

`data/` contains dataset and keyed-replay transforms. `analysis/` contains
parsers and evidence reports. `dependency/` contains raw-trace dependency
feature builders. `notebook/` contains the v4.1 materializer.

The analysis tools are not duplicate runners: each consumes a different
artifact contract. Replay-plan parsing is centralized in
`scripts/replay/resolve_replay_plan.py`.
""")
    write("formal_NN_training/archive/legacy_action_predictor/README.md", """# Legacy action-predictor material

These notebooks and notes belong to the retired SPP/action-predictor pipeline.
They intentionally reference scripts that are no longer part of the active
standalone no-prefetch workflow. They are archived for historical reference;
do not run them as current experiments.
""")
    write("formal_NN_training/README.md", """# formal_NN_training

## Current standalone pipeline

1. `scripts/run/collect_oracle_data.sh`
2. `tools/data/build_oracle_dataset.py`
3. notebook training/export of a frozen rich list
4. `scripts/build/build_keyed_listreplayer.sh`
5. `scripts/run/replay_exports.sh`
6. `scripts/run/prefetch_campaign.sh` only when event evidence is needed
7. `tools/analysis/` for replay summaries, attribution, resource pressure, and reports

Normal prefetchers are comparison baselines. The standalone model is trained
from no-prefetch demand events and replayed by stable `(pc,line,occ)` keys.
""")
    readme = ROOT / "README.md"
    if readme.is_file():
        text = readme.read_text().replace("formal_NN_training/scripts/README.md", "formal_NN_training/scripts/README.md")
        readme.write_text(text)
    ignore = ROOT / ".gitignore"
    if ignore.is_file():
        text = ignore.read_text()
        text = text.replace("formal_NN_training/scripts/local_*.py", "formal_NN_training/tools/local_*.py")
        text = text.replace("formal_NN_training/scripts/local_*.sh", "formal_NN_training/tools/local_*.sh")
        ignore.write_text(text)


def validate():
    py = [str(ROOT / p) for p in output("git", "ls-files").splitlines()
          if p.endswith(".py") and p.startswith("formal_NN_training/")]
    for path in py:
        run("python3", "-c", "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')", path)
    sh = [str(ROOT / p) for p in output("git", "ls-files").splitlines()
          if p.endswith(".sh") and p.startswith("formal_NN_training/")]
    for path in sh:
        run("bash", "-n", path)
    old = output("git", "grep", "-n", "-E", "formal_NN_training/scripts/[0-9][0-9]_.*|scripts/[0-9][0-9]_.*", "--", ":(exclude)formal_NN_training/archive/legacy_action_predictor") if False else ""
    probe = subprocess.run([
        "git", "grep", "-n", "-E", "formal_NN_training/scripts/[0-9][0-9]_.*|scripts/[0-9][0-9]_.*",
        "--", ":(exclude)formal_NN_training/archive/legacy_action_predictor"
    ], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if probe.returncode == 0:
        raise RuntimeError("active old numbered-script reference remains:\n{}".format(probe.stdout.decode()))
    if probe.returncode not in (0, 1):
        raise RuntimeError(probe.stderr.decode())
    run("git", "diff", "--check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not args.apply:
        raise SystemExit("rerun with --apply")
    if output("git", "branch", "--show-current") != "main":
        raise SystemExit("run only on main")
    if output("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked working tree is not clean; commit/stash tracked edits first")

    for old, new in MOVES + LEGACY_MOVES:
        src, dst = ROOT / old, ROOT / new
        if not src.exists():
            raise SystemExit("expected source is missing: {}".format(old))
        dst.parent.mkdir(parents=True, exist_ok=True)
        run("git", "mv", old, new)
    rewrite_tracked_paths()
    fix_moved_shell_roots()
    fix_collection_skip()
    fix_campaign_resolver()
    fix_replay_summary_resolver()
    fix_event_analysis_resolver()
    fix_notebook_materializer()
    fix_ceiling_contracts()
    write_docs()
    run("git", "rm", "--", str(SELF))
    validate()
    run("git", "add", "-A", "formal_NN_training", "README.md", ".gitignore")
    run("git", "commit", "-m", "Reorganize formal NN pipeline by role")
    run("git", "push", "origin", "main")
    print("[done] role layout committed and pushed")


if __name__ == "__main__":
    main()
