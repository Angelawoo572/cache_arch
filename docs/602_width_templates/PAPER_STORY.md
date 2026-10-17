# Paper story: retained H16 gains and simple mechanisms

- **Finding:** retained teacher-action H16 students beat native in this live MAC0 matrix, while simple validation templates explain almost all gain.
- This supports a reproducible empirical mechanism study; it does not establish a new prefetch algorithm or necessary recurrent-state advantage.
- Existing checkpoints passed actual-core Python/C++ state/action checks. **2 full validation + 22 full test runs** use **64 explicit compatible references**, preserving protocol identities and keeping old replay separate.
- Weights remain frozen; only h/c changes during NN inference. Start with the [project overview](../../RESULTS_INDEX.md).

| Claim | Evidence status | Support and limit |
|---|---|---|
| Real AMPM/SPP H16 weights support independent live inference | Supported | [Model/log identity](../../formal_NN_training/experiments/602_four_prefetcher_formal/width_templates/EVIDENCE_ALIGNMENT.md), actual-build parity and [9,748-check audit](../../formal_study/results/602_width_templates/audit.json); no substitute retraining |
| H16 retains an advantage over native | Supported within this matrix | [8/8 native comparisons](../../formal_study/results/602_width_templates/summary.csv): AMPM +0.53%–2.21%, SPP +1.41%–4.15%; H16 also beats its H8 NN in 8/8 cases |
| Learned payload or recurrent state is necessary for the gain | Not supported | [Actual template interventions](REPORT.md): SPP H16 matches IPC/cycles/requests/DRAM in 4 cases; AMPM NN has only 0.035%/0.125% uncapped residuals and loses capped; residuals do not isolate state |
| Capping H8 eliminates a no-NN explanation | Not supported | [6 capped H8 template cases](../../formal_study/results/602_width_templates/summary.csv): Streamer/SPP nearly match; AMPM templates gain 0.89%/0.95% with fewer requests; failures remain |
| A common cap equalizes traffic, or fewer requests guarantee less DRAM | Not supported | [Measured traffic](REPORT.md): uncapped SPP H16 submits fewer requests than native but raises DRAM 4.74%/11.57%; requests × 64 is not used |
| Finite-resource gains or seed/workload generalization | Pending | This supplement is MAC0/602/seed7. H16 ceil(MACs/4)/measured MAC0 cycles is 12.95–31.15×, not finite IPC; [storage and assumptions](REPORT.md) |

- Negative results remain: small AMPM residuals, lower future-ROI absolute IPC under cap, H8 SPP losses to native, and templates that do not always match cycle for cycle.
- Template failure and first cumulative crossing of NoPF do not establish stateful novelty or a minimum observation count.
- Old keyed-match losses, **25M** replay warmup and distinct h/c initialization remain in [slide corrections](SLIDE_ERRATA.md); current live warmup is **5M**. Old IPC is not pooled with live IPC.
- [Full report](REPORT.md) · [method](../../formal_study/results/602_width_templates/method.svg) · [figures and metric scope](../../formal_study/results/602_width_templates/README.md) · [raw paths/recovery](LAYOUT_AND_RECOVERY.md).
