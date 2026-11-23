# Prompt for the next step

Paste the block below into a new agent session opened at the cache root. It covers Stages 0 and 1 of the [design memo](README.md): gem5 bring-up, the capture in the [data-collection specification](DATA_COLLECTION.md), and the measurement study. It builds no neural network.

Five decisions in [section 10 of the memo](README.md#10-priorities-and-open-decisions) change what gets built. Answer them in the first message, or let the agent ask.

```text
You are continuing a computer-architecture research project in the `cache` repository
(GitHub Angelawoo572/cache_arch, branch experiment/gem5-neural-prefetch). Read AGENTS.md
first and follow it. Then read gem5_neural_prefetch/README.md (design memo) and
gem5_neural_prefetch/DATA_COLLECTION.md (capture specification). Those two files are the
specification for this task. Where they and this prompt disagree, ask.

BACKGROUND
- Earlier ChampSim studies at the shared LLC (llc_hybrid_prefetch/, llc_lstm_research/)
  found that conventional prefetchers miss useful lines or request them too late; that
  richer LLC history improves address probability mostly on loads SPP already covers; that
  graph workloads are limited by reaching the right page, not by lead; and that an exact
  address does not by itself shorten execution.
- ChampSim traces carry no load values, so dependent accesses (pointer chains, trees,
  index arrays, graph frontiers) could not be studied. gem5 executes the program, so
  values, register dataflow and ready times are observable.
- Proposed design: conventional prefetchers stay an independent fast path. A tiny learned
  slow path decides whether extra prefetching adds value and emits a bounded plan. A small
  traversal unit executes the plan by reading pointer or index fields from lines as they
  arrive. Both paths share one issuer. No model has been chosen; the memo requires the
  measurements below first.

GOAL OF THIS STEP (no neural network, no training)
1. Stage 0: build gem5, run an O3 core with NoPF and conventional-prefetcher arms on the
   agreed workloads, and show the runs are deterministic and joinable.
2. Stage 1a: implement and validate the capture streams in DATA_COLLECTION.md.
3. Stage 1b: produce measurement tables M1-M9 from the memo and state go or no-go
   against the memo's stop rules.

STEPS
1. Confirm the five open decisions in memo section 10 with the user (ISA, workloads and
   SPEC licence, fast-path configuration, tap and fill level, compute and disk budget).
2. Build a pinned gem5 release under external/gem5 (already git-ignored): 26.0.0.0
   unless there is a reason to differ, and not older than 25.1. On 22 November 2026
   Sacramento had gcc 8.5.0, system Python 3.6.8, no SCons and no gem5 tree: check
   gem5's build requirements first and install what is missing inside the cache root,
   without changing the system. Record the version, commit, compiler and build command. Keep project code in a separate directory
   compiled through EXTRAS; do not edit upstream gem5 files unless a hook is missing
   (DATA_COLLECTION.md lists the small patches expected), and record each patch.
3. Write one configuration script for the baseline machine: O3 CPU, classic caches in
   timing mode (the cache probes do not exist in Ruby), syscall emulation. Create
   checkpoints so that every arm measures the same instructions. Check: two identical
   runs give identical statistics; the committed-load hash (V1) is equal across arms.
4. Write the validation microbenchmarks (stride array, linked list, binary tree, CSR
   indirect, hash chain) with working sets larger than the LLC. Their known structure
   validates the extractors. It is never a training label.
5. Implement the capture streams as probe listeners. Run checks V1-V6. Do not start the
   capture matrix until all six pass.
6. Run the capture matrix. Compute M1-M9 with explicit denominators. Write
   gem5_neural_prefetch/measurement/FINDINGS.md with tables, figures and the go or
   no-go statement. Report negative results in full.
7. After each completed step: commit to experiment/gem5-neural-prefetch, push, and
   rsync gem5_neural_prefetch/ to Sacramento through .runtime/sacramento-ssh. Verify
   the three copies match. Raw captures, builds and checkpoints stay out of git.

RULES
- A predictor input must have been observable at its tap at prediction time. Future
  addresses, future values, baseline outcomes of future loads and source-level structure
  are labels only.
- Conventional prefetchers' candidates are never inputs to the learned unit.
- Never assume a pointer target is known before the line holding the pointer arrives.
- No reinforcement learning. No claim of novelty or speedup. Every number in a report
  links to the file that holds it.
- Do not move, overwrite or re-index the existing ChampSim data.
- Ask before any run expected to exceed a few hours or tens of gigabytes.

REPORT BACK
- What was built, what passed, what failed, with paths.
- M1-M9 per workload, and which stop rule (if any) fired.
- The filled hardware budget sheet (memo section 5), with measured values replacing
  every "to measure" entry that this step can resolve.
- The smallest next experiment you recommend and why.
```
