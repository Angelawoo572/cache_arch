# Prompt for the next step

Paste the block below into a new agent session opened at the cache root. It covers the first half of Stage 2 of the [design memo](README.md): the table design in the loop. It builds no neural network. The [measurement findings](measurement/FINDINGS.md) explain why this is the next step.

The earlier version of this file asked for Stages 0 and 1. Those are done, and that prompt is in the git history of this file.

```text
You are continuing a computer-architecture research project in the `cache` repository
(GitHub Angelawoo572/cache_arch, branch experiment/gem5-neural-prefetch). Read AGENTS.md
first and follow it. Then read, in this order:
  gem5_neural_prefetch/measurement/FINDINGS.md   what was measured and what it decides
  gem5_neural_prefetch/README.md                 the design memo, updated with "Measured" notes
  gem5_neural_prefetch/DATA_COLLECTION.md        the capture and the gem5 traps
  gem5_neural_prefetch/sim/README.md             build, run, and how to share the server
Where they and this prompt disagree, ask.

BACKGROUND
- Goal of the project: a tiny slow path beside conventional prefetchers that covers the
  dependent accesses they leave (index arrays, pointer chains, graph frontiers), without
  filtering their candidates, with every input causally available and every hardware
  cost accounted for.
- gem5 26.0.0.0 is built on Sacramento only (external/gem5, a submodule). The study's
  code is in gem5_neural_prefetch/sim/np_ext and is compiled through EXTRAS. The capture
  passes checks V1 to V5.
- Measured on 16 GAP windows: after L1D stride plus L2 signature path, a large share of
  run time is still stall on loads whose address is (loaded value << shift) + base.
  One to a few static relations per window carry it. Given the PC of the starting load,
  which relation to follow is nearly deterministic. Whether one instance will miss is
  not predictable from PC, relation or recent outcomes. A traversal that starts when the
  producer's value returns has no lead: the core issues the dependent load within a few
  cycles. Only an early start has lead: step a root whose address continues a delta,
  read the other fields of a line a load has opened, or start at a line a conventional
  prefetcher brought into L2.
- So the next design has no learned part: an exact relation table, the three early
  starts, a bounded traversal unit, a per-relation usefulness count and a shared issuer.
  A learned component is considered only where this design leaves a measured gap.
- gem5 26.0's own indirect-memory prefetcher does not run, and its signature-path
  prefetcher needs the bound in np_ext. There is no indirect-memory baseline yet.

GOAL OF THIS STEP (no neural network, no training)
1. Implement the table design as a gem5 prefetcher under sim/np_ext, with every table,
   register and queue sized explicitly and a modeled latency per lookup and per link.
2. Measure it closed-loop on the existing checkpoints against the fast path alone.
3. Close the measurements Stage 1 left open: M9 (issue-budget sensitivity) and the M4
   contexts built from the contents of an arriving line.
4. Say, from those results, whether a learned component has a measured gap to fill.

STEPS
1. Relation detection online: a small window of recent loads with their values; at each
   new load address, test address == (value << s) + base for s in 0..3 against the
   tagged producers; confirm with a confidence count per pair of PCs. Compare what it
   learns with the offline relations in measurement/tables/m5_top_relations.csv.
2. Early starts, one at a time, each behind a switch so its contribution can be
   measured: (a) stepped root, (b) the other fields of the line a load opened,
   (c) lines arriving at L2 from any prefetcher. The unit may observe fills. It must not
   read a conventional prefetcher's candidate list or internal state.
3. Traversal unit: at most 3 links, bounded width and lifetime, reads a field only from
   a line that is present, never before the line arrives. Requests go through one
   issuer that drops lines already present or in flight, translates through the data
   TLB, and meters both paths.
4. Add the capture streams the evaluation needs: S7 (prefetch lifecycle, with the unit
   that issued each request) and S10 (queue pressure). Check V1, V2, V4 and V5 again.
5. Arms, on every existing GAP window and on the micro programs list, tree, hash and csr:
   nopf, fast, fast + table design with each early start alone and all together.
   Report per window: original misses that become timely first uses and that stop being
   timely (gained, lost, net), late share, cycles for the fixed 20M instructions,
   requests issued, used and unused, DRAM reads. Sweep distance, depth, width and the
   issue budget (M9). Charge the modeled latency.
6. Write a faithful one-link indirect prefetcher as a baseline arm (the table design
   with only start (a) and one link is that baseline), and say so in the report.
7. M4 with arriving-line contexts, offline, from the captured fill contents (S8): for
   each field of each line filled into L2, does it name a later residual target, and
   how many bits does a table need to decide given the relation, the field position
   and the line's page.
8. After each completed step: commit to experiment/gem5-neural-prefetch, push, sync
   gem5_neural_prefetch/ to Sacramento through .runtime/sacramento-ssh, and check that
   the three copies match. Raw captures, builds and checkpoints stay out of git.

RULES
- A predictor or table input must have been observable at its tap at decision time.
  Outcomes of loads that have not resolved are labels only.
- Conventional prefetchers' candidates are never inputs. Their fills may be observed.
- Never use a field before its line is present.
- No reinforcement learning. No claim of novelty or speedup without the evidence the
  memo's section 8 asks for. Report losses next to gains and negative results in full.
- Every number in a report links to the file that holds it.
- gem5 stays on Sacramento. Run simulations at the lowest priority, at most about 14 at
  once, with the 4 GiB address-space limit in sim/np_runs.py, and check free memory
  before starting analysis jobs: the server is shared.
- Do not move, overwrite or re-index earlier captures. New arms get new names.
- Ask before any run expected to exceed a few hours or tens of gigabytes.

REPORT BACK
- What was built, what passed, what failed, with paths.
- Per window: gained, lost and net timely coverage, time for fixed work, traffic, and
  the cost of the unit as separate lines for tables, registers, queues and latency.
- Which early start carries the gain, and what distance, depth and width it needs.
- What the table design leaves, by mechanism, and whether any of it is a decision a
  learned component could make from causally available inputs. If yes, name the
  decision, its inputs, its label and the smallest experiment that would test it.
  If no, say so.
```
