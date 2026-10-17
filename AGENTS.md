# cache project workspace

- Keep all project edits, generated files, temporary files, downloads, model caches and reports inside `/Users/angelawoo/Documents/CMU/ece/cache`.
- Set every command's working directory to cache or a child. Source `.runtime/project_env.sh` for local temporary/cache paths.
- The healthy active Git checkout is the cache root, on `experiment/602-gate-mechanism`. `formal_study/work/cache_gate_mechanism -> ../..` is only a compatibility pointer for retained source bindings.
- Shared current results: `formal_study/results/602_width_templates`. Complete raw: `formal_study/runs/602_width_templates_20261017`. Local, Sacramento and GitHub use the same relative layout; raw/builds/traces are intentionally not committed.
- Preserve original offline and Stride-live run containers, distinct source/raw/model/build/log versions, v1/v2, old root Git archives and the Git recovery object donor. Merge differing versions into the archive rather than overwrite them.
- Use `.runtime/sacramento-ssh` for the existing authenticated Sacramento master. Do not create a project control socket outside cache or change authentication/security verification.
- Codex application state and synced read-only project references are application-managed; do not modify them as part of project cleanup.
