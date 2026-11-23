"""Take checkpoints of a workload at fixed instruction offsets after its marker.

The workload calls the gem5 work-begin operation where its kernel starts. This
script runs it on the atomic CPU, and saves one checkpoint at every requested
number of instructions after that call. Every arm later restores the same
checkpoint, so every arm starts from the same architectural state.

    gem5.opt --outdir=OUT checkpoint.py --binary B --args "..." \
        --checkpoints DIR --offsets 0,100000000
"""

import argparse
import json
import shlex
from pathlib import Path

from gem5.resources.resource import BinaryResource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator

import np_machine

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--args", default="")
parser.add_argument("--checkpoints", required=True, type=Path)
parser.add_argument(
    "--offsets",
    default="0",
    help="comma-separated instruction counts after the work-begin marker",
)
parser.add_argument("--stdout", type=Path, default=None)
args = parser.parse_args()

offsets = sorted({int(x) for x in args.offsets.split(",")})
args.checkpoints.mkdir(parents=True, exist_ok=True)

board = np_machine.checkpoint_board()
board.set_se_binary_workload(
    BinaryResource(local_path=args.binary),
    arguments=shlex.split(args.args),
    stdout_file=args.stdout,
    env_list=np_machine.WORKLOAD_ENV,
)

taken = []
state = {"next": 0, "marker_seen": False}


def save_current():
    offset = offsets[state["next"]]
    target = args.checkpoints / f"cpt_{offset}"
    simulator.save_checkpoint(target)
    taken.append({"offset": offset, "path": str(target)})
    print(f"checkpoint: {offset} instructions after the marker -> {target}")
    state["next"] += 1


def schedule_next():
    """Returns True when every checkpoint has been taken."""
    if state["next"] >= len(offsets):
        return True
    gap = offsets[state["next"]] - offsets[state["next"] - 1]
    simulator.schedule_max_insts(gap)
    return False


def on_work_begin():
    while True:
        if state["marker_seen"]:
            # Later markers, for example one per trial, are ignored.
            yield False
            continue
        state["marker_seen"] = True
        if offsets[0] == 0:
            save_current()
            yield schedule_next()
        else:
            simulator.schedule_max_insts(offsets[0])
            yield False


def on_max_insts():
    while True:
        save_current()
        yield schedule_next()


simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: on_work_begin(),
        ExitEvent.MAX_INSTS: on_max_insts(),
    },
)
simulator.run()

(args.checkpoints / "checkpoints.json").write_text(
    json.dumps(
        {
            "binary": args.binary,
            "args": args.args,
            "offsets_requested": offsets,
            "taken": taken,
            "marker_seen": state["marker_seen"],
            "last_exit": simulator.get_last_exit_event_cause(),
        },
        indent=2,
    )
    + "\n"
)
if len(taken) != len(offsets):
    print(
        f"WARNING: took {len(taken)} of {len(offsets)} checkpoints; "
        f"the program ended with: {simulator.get_last_exit_event_cause()}"
    )
