#!/usr/bin/env python3
"""CNN-only entry point for the source-input-fair 623 Stride experiment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.experiment_623_stride import run_cli


if __name__ == "__main__":
    run_cli("cnn")
