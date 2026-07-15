#!/usr/bin/env python3
"""AMPM matched-input independent direct-action LSTM entry point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_NN_training.common.direct_action_lstm import run_cli


if __name__ == "__main__":
    run_cli("ampm")
