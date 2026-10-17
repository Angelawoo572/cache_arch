> Current overview: [RESULTS_INDEX](RESULTS_INDEX.md). Both active checkouts use `main`.

# Cache consolidation

The active healthy Git checkout is now the cache root. Current lightweight results
are in [formal_study/results/602_width_templates](formal_study/results/602_width_templates/README.md),
with the same relative path on Sacramento and GitHub. The old inner checkout and
old result location are compatibility links to the same real files.

Original root Git directories and divergent source versions are retained under
`formal_study/archive/root_checkout_before_unification_20261017/` on each machine.
The guarded merge, file inventory and protected-data verification are under
`formal_study/migration/unify_cache_root_20261017/`. Raw, models, builds, historical
v1/v2 and original offline runs were preserved. The old dated top-level entries
remain removed. See [layout/recovery](docs/602_width_templates/LAYOUT_AND_RECOVERY.md).

Earlier cleanup records remain in `formal_study/migration/`; their recorded paths
describe earlier steps. Current paths are given by [RESULTS_INDEX](RESULTS_INDEX.md).
New local work and caches stay within this cache folder.
