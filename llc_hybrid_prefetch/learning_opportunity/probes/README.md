# Does more history help predict an address?

The probes learn address actions from the first 60% of a complete LLC arrival stream. Each can independently generate a physical line. The next 20% supplies development measurements and the final 20% is held out. A second evaluation applies the original fit to another recording, program, graph input or page layout. [Findings](FINDINGS.md) · [Protocol](PROTOCOL.md) · [Samples and trace paths](sampling.csv)

For each result the source column identifies the only training recording, except the explicitly pooled 36,000-cue sensitivity. The older project name `soplex06_heldout` refers to Soplex-92B. It is a held-out destination when trained on BFS, a donor in the reverse test and one of three donors in the pooled test. Its 247B recording is independent. The protected eleven expansion recordings supply no fitted parameters of either BFS-only or pooled models.

## Inputs and targets

Every admission updates the history. For fitting and scoring, draw one cue at a deterministic random position in each arrival block, targeting about 30,000 cues per file. The source is an ordered Stride execution. All inputs use values available by the cue. The output is the physical line at the first, eighth or thirty-second subsequent LLC admission. Purge training and development anchors whose 32nd successor crosses the partition boundary. A second diagnostic predicts the first and eighth future demand at the same PC, a different PC or a different physical region. It censors queries crossing the partition independently for each definition. [Fixed-horizon scores](scores.csv) · [Related targets](related_targets.csv)

Each encoding has up to 64 actions learned exclusively from the training targets. One encoding adds a learned signed displacement to the current physical line. The other selects an offset on one of eight recently observed physical pages. Page slot zero is the current page. The remaining slots are the most recent distinct pages within the preceding 64 arrivals. One extra class groups out-of-vocabulary targets. A legal hit requires the exact line on a page in that causal bank. This vocabulary allows a candidate on a different, already observed physical page. Stride’s implementation may have a stricter same-page issue rule. The bank check charges neither a physical lookup nor duplicate/queue rejection. The prospective one-off replay enforces the current physical page.

| Feature scope | Observations available to the probe |
|---|---|
| Current | Exact trigger PC encoded with a training-only dictionary, and offset within its 64-line page. Unseen PC IDs share one code. |
| Current plus delta | Current PC and offset plus the signed line distance to the most recent access by this PC. |
| Same PC or region | Current features plus events 1, 2, 4 and 8 positions back in the active PC or physical 256 KiB region. Each token includes relative line distance, offset, and PC/page/region equality. |
| Global 4–64 | The PC and region histories plus every event in the indicated ordered global window. Absolute producer identities use the same training dictionary. |
| Shuffled 32 | The 32 global token vectors receive a common within-anchor permutation across all fields. |
| Hide other PCs | Keep global positions and PC-equality flags. Hide the remaining global token fields when the producer PC differs. Region-group history is retained. |
| No absolute PC IDs | Use the same 32 events and remove PC ID fields. PC equality and same-PC history lookup still require comparing PC identities. |
| Time and pressure | Add causal cycle gaps to the 32-event model. A separate ablation adds measured ingress LLC MSHR/RQ/PQ/WQ and DRAM RQ/WQ occupancy. |
| Cache and conventional feedback | Add the outcome and age of the 1st, 2nd, 4th and 8th earlier LLC tag lookup. A second fit adds stage, reason and age of the same lags among earlier Stride decisions, admissions and first-use/eviction feedback. Each callback must precede the cue in the simulator’s event order. |
| Core sideband | Add source/destination register counts, operand slot, nearest recorded definition of each source register in the preceding 32 program instructions, and four earlier branch ages and directions in the 257–320 instruction band. These trace-derived summaries approximate an extra hardware feedback path. Branch outcomes come from beyond the 256-entry ROB. The trace does not provide actual register values or realized dependency latency. |

Each offline Extra Trees probe has 16 trees, up to 192 leaves per tree, a minimum of 12 observations per leaf, and at most 24 randomized features per split. Five percent of the predicted distribution comes from the smoothed training prior. The same parameters apply throughout. Extra context can help or hurt a finite probe. The score measures held-out predictive performance, with eight consecutive test blocks showing phase variation. Hardware area, throughput, candidate contention and IPC are outside this fit.

## Read the numbers

- `mean_bits` is average negative log-base-two probability of the observed action, including the grouped other class. `conditional_mean_bits` measures address loss only where an exact target is expressible, after normalizing over concrete actions. Compare scopes on the same fitted vocabulary.
- `reachable` counts targets on a currently known page. `expressible` counts targets in the training vocabulary, independently of page legality for signed displacements. `correct` requires an exact physical line. `hit` additionally requires legal construction. `deadline` additionally requires at least 130 cycles to the observed target demand. These are offline opportunities. They assume zero probe latency and do not establish a timely cache fill.
- The primary decision chooses the largest probability across concrete actions and the other class. It abstains when other wins. The `forced_*` counters ask what happens when every cue must propose its highest-probability concrete action, before applying the legality check. Neither decision policy uses future target status.
- `unseen_train_pc` compares trigger identities with the source training-anchor dictionary. The seen-line, known-page and known-region cohorts compare each target with **all earlier admissions** in its destination's measured stream. The first reference in that stream can have a warmup or higher-level antecedent. [Novelty strata](unseen_contexts.csv)
- [Paired target comparisons](paired_targets.csv) partition individual gains, losses and common lead-qualified addresses at horizon eight. [Conventional outcomes](conventional_outcomes.csv) join source instruction and operand in NoPF and conventional executions. [Dynamic examples](paired_examples.csv) give the first added qualifying target in each cohort. An example requires both exact physical-line prediction and a recorded conventional late merge.
- Each `raw/probes/fits/<experiment>/` folder retains all development/test scores, eight test-block rows per scope, detailed strata and conventional outcomes. The compact tables retain both encodings and all three global horizons. The aligned stratified extracts use horizon eight. [History value by horizon](history_value.csv)

The [paired placement test](matched_placements.csv) joins cues and future targets by source instruction and operand across independently simulated placements. The target’s physical line can change even when both dynamic operands match.

Local ambiguity is estimated only from the training split. A context uses the page offset, two prior same-PC distances, and optionally the producer PC. Require six training examples and at least two concrete actions with two examples each. Test results include unsupported contexts separately. [Ambiguity study](ambiguous_contexts.csv)

The [one-shot replay selector](prospective.py) consumes the donor training split and a cue-only view of the destination. The first predeclared window had zero eligible proposals in 1,500 cues. The second supplied one valid proposal, with the same-PC probe abstaining. [Selection protocol](PROTOCOL.md) · [Frozen offer](prospective_offers.csv)

## Reproduce

Use a C++17 compiler with zlib and Python with NumPy, pandas, scikit-learn and Matplotlib. The measured run used NumPy 2.3.5 and scikit-learn 1.7.2. Run from this folder with native trace views and the baseline operand outcomes available. The [sequence builder](../labels/README.md) supplies related-event indices.

```sh
p=../raw/probes
python3 -B check.py
python3 -B extract.py bfs_discovery soplex06_heldout snap_stanford snap_google
python3 -B sideband.py bfs_discovery soplex06_heldout
python3 -B pressure.py bfs_discovery
for source in bfs_discovery soplex06_heldout; do
    python3 -B fit.py --source "$source" --dest bfs_discovery soplex06_heldout --out "$p/fits/$source"
    python3 -B fit.py --source "$source" --dest bfs_discovery soplex06_heldout --scopes with_core --out "$p/fits/${source}_sideband"
    python3 -B paired.py --source "$source" --dest bfs_discovery soplex06_heldout --out "$p/paired/$source"
    python3 -B first_demand.py --source "$source" --dest bfs_discovery soplex06_heldout
done
python3 -B fit.py --source bfs_discovery --dest bfs_discovery --scopes with_pressure --out "$p/fits/bfs_pressure"
python3 -B fit.py --source snap_stanford --dest snap_stanford snap_google --scopes prior current same_pc local_region hide_other_pc global32 no_pc_ids with_time --out "$p/fits/snap_stanford"
python3 -B paired.py --source snap_stanford --dest snap_stanford snap_google --out "$p/paired/snap_stanford"
python3 -B related.py bfs_discovery soplex06_heldout
python3 -B paired.py --source bfs_discovery --dest bfs_transfer --out "$p/paired/bfs14"
python3 -B first_demand.py --source mcf17_discovery --dest mcf17_discovery
python3 -B ambiguity.py
python3 -B prospective.py
python3 -B sideband.py mcf17_discovery
python3 -B pressure.py mcf17_discovery
python3 -B fit.py --source mcf17_discovery --dest mcf17_discovery --scopes with_core with_pressure --out "$p/fits/mcf_aux"
for source in bfs_discovery mcf17_discovery; do
    python3 -B feedback.py "$source"
    python3 -B fit.py --source "$source" --dest "$source" --scopes with_feedback with_stride_feedback --out "$p/fits/${source}_feedback"
done
python3 -B suite.py --wait
python3 -B pooled.py
python3 -B layout.py
python3 -B external_feedback.py
python3 -B domains.py --wait
python3 -B join_late_arms.py
python3 -B tables.py
python3 -B report.py
python3 -B audit.py --require-complete
```
