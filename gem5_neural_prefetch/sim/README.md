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

The script builds `build/X86/gem5.opt` with `EXTRAS=sim/np_ext`, which adds two objects:

- `NpCapture`, the capture. It only listens to probes.
- `NpSignaturePathPrefetcherV2`, gem5's signature-path prefetcher with a bound of 1,024 steps on one lookahead walk. In gem5 26.0 a walk does not end when its confidence stops falling: the simulator stops advancing and its memory grows without bound. A walk that ends on its own within the bound is unchanged. The statistic `lookaheadsStopped` counts the walks the bound ended.

An incremental build after a change in `np_ext` takes about four minutes. The linker removes the old binary first, so simulations that are already running are not disturbed, and a simulation started during the link fails at once.

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
| `fast` | stride | signature path, version 2, with the bound above |
| `bop` | stride | best offset |
| `ampm` | stride | access map pattern matching |
| `imp` | indirect memory | none |
| `fastcheck`, `stridecheck` | as `fast`, as `stride` | Second names, so that a rerun can sit next to an earlier capture and be compared with it |

Prefetchers keep gem5's default parameters. The indirect-memory arm also sets `prefetch_on_access=True` and `prefetch_on_pf_hit=False`, because that prefetcher reads index values from hits.

**The `imp` arm does not run in gem5 26.0.** It stops with `PrefetchInfo::get called with a request with no data` at the first index value it tries to read, because the notification for a read hit no longer carries the data. Read by inspection, its prefetch loop also requests the address of the current index several times instead of looking ahead. The arm stays defined for a later gem5 or a repaired prefetcher. No result for it exists.

## Checkpoint, then run

Each workload executes gem5's work-begin operation where its kernel starts. [`configs/checkpoint.py`](configs/checkpoint.py) runs it on the atomic CPU and saves a checkpoint at each registered instruction offset after that point. [`configs/run.py`](configs/run.py) restores one checkpoint into the out-of-order machine for one arm, warms up, and records the window.

```sh
source sim/env.sh
python sim/np_runs.py checkpoint micro_list gap_bfs_kron20
python sim/np_runs.py run --arms nopf,fast --warmup 5000000 --measure 20000000 micro_list gap_bfs_kron20
python sim/np_runs.py status
```

Workloads, arguments and offsets are registered in [`workloads.json`](workloads.json). Outputs go to `raw/checkpoints/` and `raw/runs/`, which git ignores. [`run_matrix.sh`](run_matrix.sh) reproduces every capture of the measurement study and skips what is already finished.

## Sharing the server

- Every gem5 process runs at the lowest CPU priority.
- A measured run needs about 2.3 GiB: the 2 GiB of simulated memory and gem5 itself. `np_runs.py` gives each run a 4 GiB address-space limit, so a run that grows without bound fails quickly instead of taking memory from other users.
- Fourteen runs side by side use about 30 GiB. Check `free -g` before starting more, and run the analysis scripts, which take 3 to 4 GiB each, when the simulations are done.
- Do not wait for one job with `pgrep -f` on a pattern that also appears in the command line of the job that waits: two such waiters block each other. Wait on a process id.

## Cost

Measured on this server with about 40 other simulations running: 16,000 to 42,000 simulated instructions per second per run (median 28,000), so a window of 5M plus 20M instructions takes 10 to 25 minutes. Recording costs 7% to 25% of that. A window's capture takes 0.13 to 0.62 GB per arm, and 0.20 to 0.75 GB for the arm that also holds the micro-op, load and store streams.
