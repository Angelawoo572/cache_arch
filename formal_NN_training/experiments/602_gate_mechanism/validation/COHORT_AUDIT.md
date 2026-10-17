# Reproduce the independent native-SPP cohort audit

`verify_spp_cohorts.py` is postprocessing only. It reads original run records,
complete summary snapshots, and the corresponding analysis CSV. It does not
modify a simulation, checkpoint, or raw result. It checks the four native-SPP
cases (two future ROIs × original / common request ceiling) and keeps every case
separate.

First run `analysis_gate.py` on the same raw snapshot into a fresh analysis
directory. Then, from any working directory, supply explicit paths:

```sh
/opt/anaconda3/bin/python3 /PATH/TO/602_gate_mechanism/validation/verify_spp_cohorts.py \
  --raw /PATH/TO/gate_output/raw \
  --prior-raw /PATH/TO/original_formal_v2/raw \
  --analysis /PATH/TO/gate_output/analysis_final \
  --out /PATH/TO/docs/602_gate_mechanism/cohort_verification.json \
  --require-complete
```

The prior raw root is the preserved main-v2 data containing `test/` summaries,
not the h/c supplement. `--require-complete` requires all four SPP cases; without
it an in-progress scope certificate lists missing cases and stays INCOMPLETE.
An analysis CSV must contain the cases being checked. Overall full-matrix
verification is true only if that separate analysis audit also reports all
71 rows PASS. Four SPP cases passing does not itself certify the whole matrix.

The audit checks L2/LLC submission and actual PQ partitions, complete L2-target
lifecycle conservation, explicit null LLC-only lifecycle/accuracy/timeliness,
all five actual retirement phases, exact CSV ratios, and functional equality
to original native SPP. The latter compares instructions, cycles, LOAD misses,
LOAD accesses, requested prefetches, allowed requests, and every common actual
DRAM counter. The old mixed-fill cohort is deliberately excluded from that
functional equality check. L2-target coverage remains a partial subset, and
the old mixed-fill accuracy/timeliness must not be described as whole-SPP
metrics.
