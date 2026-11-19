"""Register families using only the official DPC4 object manifest."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import re
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
MANIFEST = STUDY / "raw/intake/official_manifest.txt"
BASE = "https://traces.rbera.com/dpc4/"
COUNTS = {"SPEC17": 92, "Graph": 79, "gtrace_v2": 359, "ai-ml": 80}
GOOGLE_DISCOVERY = ("arizona", "charlie", "delta", "merced", "tahoe", "tango", "yankee")
GOOGLE_RESERVED = ("bravo", "sierra.a.4", "whiskey")
# Reserve the whole family in the manifest index, including other Sierra sizes.
FAMILY_RULES = (
    ("graph_bfs3", "discovery", "Graph/GAP/bfs-", "Graph/GAP/bfs-3.trace.gz", "../opportunity_study/raw/traces/bfs-3.champsimtrace"),
    ("graph_sssp", "discovery", "Graph/GAP/sssp-", "", ""),
    ("graph_clique", "discovery", "Graph/GMS/gms.k_clique_count_set_based.", "", ""),
    ("graph_pagerank", "discovery", "Graph/Ligra/ligra_PageRank.", "", ""),
    ("spec_mcf", "discovery", "SPEC17/605.mcf_s-", "SPEC17/605.mcf_s-665B.champsimtrace.xz", "../raw/traces/605.mcf_s-665B.champsimtrace"),
    ("spec_xalan", "discovery", "SPEC17/623.xalancbmk_s-", "SPEC17/623.xalancbmk_s-700B.champsimtrace.xz", "../raw/traces/623.xalancbmk_s-700B.champsimtrace"),
    ("spec_cactus", "discovery", "SPEC17/607.cactuBSSN_s-", "SPEC17/607.cactuBSSN_s-2421B.champsimtrace.xz", "../raw/traces/607.cactuBSSN_s-2421B.champsimtrace"),
    ("ai_clip", "discovery", "ai-ml/clip_trace_", "", ""),
    ("ai_llama", "discovery", "ai-ml/llama2.c-stories15M.", "", ""),
    ("reserve_graph_bc", "heldout", "Graph/GAP/bc-", "", ""),
    ("reserve_ai_whisper", "heldout", "ai-ml/whisper_trace_", "", ""),
    ("reserve_spec_gcc", "heldout", "SPEC17/602.gcc_s-", "", ""),
)


def numeric_name(name):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def register():
    with MANIFEST.open() as f:
        objects = {r["path"]: r for r in csv.DictReader((s for s in f if not s.startswith("#")), delimiter="\t")}
    files = {k: v for k, v in objects.items() if k.endswith((".gz", ".xz"))}
    assert Counter(k.partition("/")[0] for k in files) == COUNTS
    assert set(objects) - set(files) == {"4c-mix.csv"}
    slots = []
    for label in GOOGLE_DISCOVERY + GOOGLE_RESERVED:
        split = "discovery" if label in GOOGLE_DISCOVERY else "heldout"
        slots.append(("gg_" + label.replace(".", "_"), split, "gtrace_v2/" + label + "/", "", ""))
    slots.extend(FAMILY_RULES)
    reservations = [g for _, s, g, _, _ in slots if s == "heldout"] + ["gtrace_v2/sierra."]
    rows = []
    for case, split, group, fixed, legacy in slots:
        candidates = sorted((k for k in files if k.startswith(group)), key=numeric_name)
        assert candidates
        key = fixed or candidates[(len(candidates) - 1) // 2]
        assert key in candidates
        assert split == "heldout" or not any(key.startswith(g) for g in reservations)
        rows.append(dict(case=case, split=split, collection=key.split("/")[0], group=group,
                         manifest_path=key, compressed_bytes=files[key]["size_bytes"],
                         source_url=BASE + key, prefix_path=f"raw/intake/traces/{case}.champsimtrace" if split=="discovery" else "",
                         legacy_path=legacy))
    assert len({r["case"] for r in rows}) == len(rows) == 22
    assert sum(r["split"] == "discovery" for r in rows) == 16
    write_csv("registry.csv", rows)
    indexes = {r["manifest_path"]: r for r in rows}
    complete = []
    for key, raw in sorted(files.items()):
        select = indexes.get(key, {})
        reserved = any(key.startswith(g) for g in reservations)
        complete.append(dict(manifest_path=key, collection=key.split("/")[0],
                             compressed_bytes=raw["size_bytes"], last_modified=raw["last_modified"],
                             reserved_family=int(reserved), selected_case=select.get("case", ""),
                             selected_split=select.get("split", "")))
    write_csv("manifest_index.csv", complete)
    print(f"{len(complete)} catalog entries, {len(rows)} registered windows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh-manifest", action="store_true")
    args = p.parse_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh_manifest or not MANIFEST.exists():
        with urlopen(BASE + "manifest.txt", timeout=30) as response:
            MANIFEST.write_bytes(response.read())
    register()
