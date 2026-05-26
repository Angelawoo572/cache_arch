# Configuration files

This directory stores small, human-readable experiment configuration files that are useful to keep in git.

## `bypass/`

PC lists for the bypass experiments. These are generated from trace-dumper CSVs using:

```bash
python3 scripts/profile_bypass_pcs.py results/access_trace.605.mcf_s-994B.csv --top 10
```

The default bypass list is:

```text
configs/bypass/bypass_pc_list.txt
```

`run_bypass.sh` now defaults to that path, but you can still sweep list sizes explicitly:

```bash
BYPASS_PC_LIST=configs/bypass/bypass_pc_list_25.txt TAG=top25 bash scripts/run_bypass.sh
```

Generated model prefetch lists such as `prefetch_list_GRU_V8.txt` are not tracked here. They are large run outputs and are ignored by git.
