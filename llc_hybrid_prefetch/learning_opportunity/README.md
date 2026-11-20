# What can LLC history tell us about a future address?

At an LLC demand arrival, can information already visible to the hardware identify a useful physical cache line that Stride, Multi-delta or SPP will miss or request too late? A predictor can generate its own address. The measured outcomes are an exact future line, enough lead for service, a timely first use and a change in fixed-work execution time. Each has a different denominator.

[Findings](FINDINGS.md) · [Eleven external recordings](WORKLOADS.md) · [Annotated figures](figures/README.md) · [Find the raw data](DATA.md) · [All trace and run paths](data_registry.csv)

The nine original 200M-instruction windows now contain continuous source and cache histories under all four policies. Together the 36 runs recorded **333,826,717 LLC demand arrivals** and **2,195,229,280 source load operands**, counting each policy execution separately. The retirement outcomes match the original simulator for every measured instruction. The seven compressed event streams occupy 43.22 GiB. [Per-run counts and raw paths](capture/captures.csv)

Eleven additional 200M Stride runs contribute **105,009,979 LLC arrivals** and **667,753,481 source operands**. The 47 full-path captures together occupy 56.38 GiB. Every external NoPF, Multi-delta and SPP run retains its own continuous LLC sequence and per-operand outcomes. [External capture inventory](capture/external_captures.csv)

The learning dataset covers **840,353,358 LLC arrivals in 104 policy executions**. Its 26 views include alternate phases, two uniform physical placements, two named BFS graph inputs and the Google and AI recordings. The seven future ranks and time-window labels can be queried at every arrival. Counts across policies and placements include repeated source instructions. [Indexed datasets](labels/datasets.csv) · [Read an example](labels/README.md)

| Read in this order | Decision the evidence supports |
|---|---|
| [Complete streams and matching](capture/README.md) | Which load, private-cache and LLC events were genuinely observable at the cue? |
| [Future targets and physical reach](labels/RESULTS.md) | Which lines were representable at that cue, and how much time remained before demand? |
| [Held-out address prediction](probes/README.md) | What does richer history add on a later phase, another recording, input or page placement? |
| [A forecast and SPP's actual request](probes/SPP_CLOCK.md) | Can a frozen probe name the same physical line before SPP proposes it, on the same clock? |
| [Prospectively chosen offer](one_offer/README.md) | What happened when a fitted probe's independently generated address went through the shared issuer? |
| [Oracle timing replays](trials/README.md) | How much did the timing of a legal, known address change first uses, traffic and completion? |
| [Resource measurements](resources/README.md) | How frequently do decisions arrive, and what pressure and interference would an online unit face? |

The [workload register](expansion/README.md) extends the [original fixed-work study](../full_workload/README.md) across SPEC CPU2006, SPEC CPU2017 and GAP. Two shorter runs hold the BFS executable fixed and change the graph input. Two [cross-domain transfers](domains/README.md) use a Google service and an AI inference recording. The original Google and AI/ML reserve assignments remain available for future tests. No hardware inference engine is assumed in this experiment.
