# Simulator side: build, checkpoint, run

gem5 lives only on Sacramento, in `/home/qianruw/cache/external/gem5`. The repository pins it as a git submodule at the official `v26.0.0.0` tag (commit `fed0e07`) and keeps no copy on the Mac. Everything in this folder is the study's own code. No upstream gem5 file is edited.

## Environment

| Item | Value on Sacramento |
|---|---|
| Host | Red Hat Enterprise Linux 8.10, 48 cores, shared with other users |
| Compiler | `/opt/rh/gcc-toolset-13` (GCC 13.3.1). gem5 26.0 accepts GCC 11 to 16; the default GCC 8.5.0 is too old |
| Python | `/usr/bin/python3.11` with headers. gem5 embeds this interpreter |
| SCons, NumPy, Matplotlib | In the virtual environment `.runtime/gem5-venv` under the cache root |
| Workload compiler | The system GCC 8.5.0, dynamically linked. No static C library is installed |

`source sim/env.sh` sets all of it. gem5's build reads `PYTHON_CONFIG` from the environment; without it the build picks the Python 3.6 tool and stops.

## Build

```sh
cd /home/qianruw/cache/gem5_neural_prefetch
bash sim/build_gem5.sh 12          # jobs; runs at the lowest CPU priority
```

The script builds `build/X86/gem5.opt` with `EXTRAS=sim/np_ext`, which adds the capture object `NpCapture`.

## Machine

One x86-64 core in syscall-emulation mode at 4 GHz, 2 GiB on one DDR4-2400 channel, classic caches in timing mode. Sizes are in [`configs/np_machine.py`](configs/np_machine.py).

| Cache | Size | Ways | Tag and data latency | MSHRs |
|---|---|---:|---:|---:|
| L1I | 32 KiB | 8 | 1 cycle | 8 |
| L1D | 48 KiB | 12 | 4 cycles | 16 |
| L2 | 512 KiB | 8 | 10 cycles | 32 |
| L3 | 2 MiB | 16 | 20 cycles | 64 |

The core keeps gem5's O3 defaults except a 256-entry reorder buffer, 96 load-queue and 64 store-queue entries, and 320 physical integer and floating-point registers.

| Arm | L1D prefetcher | L2 prefetcher |
|---|---|---|
| `nopf` | none | none |
| `stride` | stride | none |
| `fast` | stride | signature path, version 2 |
| `bop` | stride | best offset |
| `ampm` | stride | access map pattern matching |
| `imp` | indirect memory | none |

Prefetchers keep gem5's default parameters. The indirect-memory arm also sets `prefetch_on_access=True` and `prefetch_on_pf_hit=False`, because that prefetcher reads index values from hits.

## Checkpoint, then run

Each workload executes gem5's work-begin operation where its kernel starts. [`configs/checkpoint.py`](configs/checkpoint.py) runs it on the atomic CPU and saves a checkpoint at each registered instruction offset after that point. [`configs/run.py`](configs/run.py) restores one checkpoint into the out-of-order machine for one arm, warms up, and records the window.

```sh
source sim/env.sh
python sim/np_runs.py checkpoint micro_list gap_bfs_kron20
python sim/np_runs.py run --arms nopf,fast --warmup 5000000 --measure 20000000 micro_list gap_bfs_kron20
python sim/np_runs.py status
```

Workloads, arguments and offsets are registered in [`workloads.json`](workloads.json). Outputs go to `raw/checkpoints/` and `raw/runs/`, which git ignores.
