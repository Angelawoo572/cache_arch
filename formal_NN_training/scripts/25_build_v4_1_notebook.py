#!/usr/bin/env python3
"""Materialize v4.1 by applying the committed override cell to a v4.0 notebook.

The full v4.1 override is stored compressed in the repository so the generated
notebook is exactly reproducible from the supplied v4.0 source notebook.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import json
from pathlib import Path


def source(cell):
    return "".join(cell.get("source", []))


def lines(text):
    return text.splitlines(keepends=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--payload", type=Path, default=None)
    args = p.parse_args()
    payload = args.payload or Path(__file__).resolve().parents[1] / "LSTM/notebooks/v4_1_extension.py.gz.b64"
    extension = gzip.decompress(base64.b64decode(payload.read_text().strip())).decode()
    nb = json.loads(args.base.read_text())
    if not isinstance(nb.get("cells"), list):
        raise ValueError("base is not a notebook with cells")
    b0 = next((i for i,c in enumerate(nb["cells"]) if "B0. v4.0 configuration" in source(c)), None)
    b5 = next((i for i,c in enumerate(nb["cells"]) if "B5. v4.0 candidate banks" in source(c)), None)
    b6 = next((i for i,c in enumerate(nb["cells"]) if "B6. v4.0 Run All" in source(c)), None)
    if None in {b0, b5, b6}:
        raise RuntimeError("base notebook does not match the expected v4.0 cell markers")
    s = source(nb["cells"][b0])
    s = s.replace('VERSION = "v4_0"', 'VERSION = "v4_1"')
    s = s.replace('RUN_ID = os.environ.get("RUN_ID", "v4_0_timing_coverage_seed7")', 'RUN_ID = os.environ.get("RUN_ID", "v4_1_policy_coverage_seed7")')
    s = s.replace('ORACLE_DIR = Path("formal_NN_training/results/standalone_nn_data/oracle")', 'ORACLE_DIR = Path(os.environ.get("ORACLE_DIR", "formal_NN_training/results/standalone_nn_data/oracle"))')
    s = s.replace('LEAD_EDGES = [4, 8, 16, 32, 64, 128]', 'LEAD_EDGES = [4, 8, 10, 12, 16, 32, 64, 128]')
    s = s.replace('CYCLE_LO = np.asarray([0, 64, 256, 1024, 4096, 16384], dtype=np.int64)', 'CYCLE_LO = np.asarray([0, 64, 128, 256, 1024, 4096, 16384], dtype=np.int64)')
    s = s.replace('ledger_scope="val",\n    ledger_csv="targets",', 'ledger_scope=os.environ.get("LEDGER_SCOPE", "val"),\n    ledger_csv=os.environ.get("LEDGER_CSV", "targets"),')
    nb["cells"][b0]["source"] = lines(s)
    s = source(nb["cells"][b5]).replace('V4_PREFLIGHT = v4_preflight()', '# v4.0 preflight is superseded by the v4.1 extension cell.')
    nb["cells"][b5]["source"] = lines(s)
    nb["cells"].insert(b6, {"cell_type":"markdown", "metadata":{}, "source":["## v4.1 registered experiments\n", "\n", "Run this notebook with `LEDGER_SCOPE=full` when building bank ceilings or exact event joins.\n"]})
    nb["cells"].insert(b6 + 1, {"cell_type":"code", "execution_count":None, "metadata":{}, "outputs":[], "source":lines(extension)})
    for cell in nb["cells"]:
        text = source(cell)
        if "B6. v4.0" in text or "B7. v4.0" in text or "B8. Download v4.0" in text or "v4_0_replay" in text or "v4_0_train" in text:
            cell["source"] = lines(text.replace("v4.0", "v4.1").replace("v4_0", "v4_1"))
    if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown":
        nb["cells"][0]["source"] = lines(source(nb["cells"][0]).replace("v4.0", "v4.1").replace("v4_0", "v4_1"))
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            compile(source(cell), "cell_{}".format(i), "exec")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb, indent=1))
    print("[wrote]", args.out)

if __name__ == "__main__":
    main()
