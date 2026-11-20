"""Run the frozen data-collection protocol on two additional, cross-domain validation sources."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "external_run", HERE.parent / "expansion/run.py"
)
external = importlib.util.module_from_spec(spec)
spec.loader.exec_module(external)

if __name__ == "__main__":
    external.HERE = HERE
    external.VALID = HERE / "validation"
    external.main()
