"""Restore a checkpoint into the out-of-order machine and measure one arm.

    gem5.opt --outdir=OUT run.py --binary B --args "..." --checkpoint CPT \
        --arm nopf --warmup 5000000 --measure 20000000 --capture DIR

The capture window is counted in committed instructions by the capture object
itself, so it holds exactly the same instructions in every arm. gem5's own
statistics are reset and dumped at the nearest cycle boundaries.
"""

import argparse
import json
import shlex
from pathlib import Path

import m5
from m5.objects import NpCapture

from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

import np_machine

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--args", default="")
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--arm", required=True, choices=sorted(np_machine.ARMS))
parser.add_argument("--warmup", type=int, default=5_000_000)
parser.add_argument("--measure", type=int, default=20_000_000)
parser.add_argument("--capture", type=Path, default=None)
parser.add_argument(
    "--no-uops",
    action="store_true",
    help="skip the architectural streams S1 to S3; they are equal in every arm",
)
parser.add_argument("--stdout", type=Path, default=None)
args = parser.parse_args()

board, hierarchy, cpu = np_machine.measured_board(args.arm)
board.set_se_binary_workload(
    BinaryResource(local_path=args.binary),
    arguments=shlex.split(args.args),
    checkpoint=args.checkpoint,
    stdout_file=args.stdout,
    env_list=np_machine.WORKLOAD_ENV,
)

if args.capture is not None:
    args.capture.mkdir(parents=True, exist_ok=True)
    cpu.np_capture = NpCapture(
        manager=cpu,
        out_dir=str(args.capture),
        l1d=hierarchy.l1d,
        l2=hierarchy.l2,
        l3=hierarchy.l3,
        content_level=2,
        record_uops=not args.no_uops,
        start_inst=args.warmup,
        measure_insts=args.measure,
    )

# The run continues a little past the window so the last instructions of the
# window have committed whatever the commit width does in that cycle.
SLACK = 2000


def on_max_insts():
    m5.stats.reset()
    simulator.schedule_max_insts(args.measure + SLACK)
    yield False
    m5.stats.dump()
    yield True


simulator = Simulator(
    board=board, on_exit_event={ExitEvent.MAX_INSTS: on_max_insts()}
)
simulator.schedule_max_insts(args.warmup)
simulator.run()

info = {
    "binary": args.binary,
    "args": args.args,
    "checkpoint": str(args.checkpoint),
    "arm": args.arm,
    "prefetchers": np_machine.ARMS[args.arm],
    "warmup_instructions": args.warmup,
    "measured_instructions": args.measure,
    "clock": np_machine.CLOCK,
    "clock_period_ticks": int(
        m5.ticks.fromSeconds(m5.util.convert.anyToLatency(np_machine.CLOCK))
    ),
    "caches": np_machine.CACHES,
    "o3_sizes": np_machine.O3_SIZES,
    "last_exit": simulator.get_last_exit_event_cause(),
    "final_tick": simulator.get_current_tick(),
}
Path(m5.options.outdir, "run_info.json").write_text(
    json.dumps(info, indent=2) + "\n"
)
