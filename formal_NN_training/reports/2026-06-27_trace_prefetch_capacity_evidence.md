# Trace, event, and cache-capacity evidence — 2026-06-27

## Source and extraction audit

This report copies the measured summary-level evidence from the uploaded `analysis_review_20260627_220800.tar.gz` review bundle. The bundle is an analysis artifact, not a tracked raw-results directory.

- Archive members: **154**; extracted files: **113**; uncompressed file bytes: **754,634**.
- Every extracted file was read as UTF-8; all **66 CSVs** parsed successfully and all **11 JSON files** parsed successfully.
- C / assembly mapping remains N/A: the trace format does not contain opcode bytes, and no matching original executable address space was supplied.
- `A_roi_trace_profile`: 25 text files, 101,246 uncompressed bytes.
- `B_full_trace_profile`: 25 text files, 110,487 uncompressed bytes.
- `DEF_event_attribution_report`: 14 text files, 399,210 uncompressed bytes.
- `G_capacity_sweep`: 47 text files, 116,735 uncompressed bytes.

### Capacity-sweep completeness

| requested point | status in uploaded bundle | evidence present |
|---|---|---|
| L1D half: 16 KiB | complete | normal summary + v3.1 summary + v3.3 summary |
| L1D double: 64 KiB | complete | normal summary + v3.1 summary + v3.3 summary |
| L2C half: 128 KiB | complete | normal summary + v3.1 summary + v3.3 summary |
| L2C double: 512 KiB | complete | normal summary + v3.1 summary + v3.3 summary |
| LLC half: 1 MiB | complete | normal summary + v3.1 summary + v3.3 summary |
| LLC double: 4 MiB | **incomplete** | normal build-info and normal RUN_INFO only; no normal summary, no replayer build-info, no v3.1/v3.3 summary |

Therefore the uploaded archive supports **five completed frozen-list capacity points**, not a complete six-point L1D/L2C/LLC half/base/double matrix. The 4 MiB LLC point must not be used in a final capacity claim.

## Experiment contract

All completed capacity points use 25M warmup + 25M simulation instructions, five traces, normal policies `no_pref sandbox sms ampm spp`, and baseline-capacity frozen standalone L2C exports (`v3_1`, `v3_3`). These are system-sensitivity controls, not capacity-trained neural results.

## A/B. Dynamic trace structure

Full profiles read 2,000,000,000 records per trace. ROI profiles read records 25M–50M, matching the current replay experiments.

| trace | ROI PCs | full PCs | ROI load pages | full load pages | ROI branch | full branch | ROI memory | full memory | ROI top-10 PC | full top-10 PC | ROI top-10 delta | full top-10 delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 602.gcc_s-734B | 485 | 1933 | 4965 | 13689 | 0.2120 | 0.2128 | 0.3745 | 0.3749 | 0.2464 | 0.2433 | 0.4787 | 0.4777 |
| 605.mcf_s-994B | 715 | 715 | 102356 | 323545 | 0.1919 | 0.1917 | 0.4186 | 0.4182 | 0.3134 | 0.3113 | 0.4525 | 0.4242 |
| 619.lbm_s-4268B | 510 | 537 | 4603 | 312784 | 0.0178 | 0.0160 | 0.4439 | 0.4269 | 0.0416 | 0.0343 | 0.7873 | 0.7719 |
| 620.omnetpp_s-874B | 6591 | 7789 | 32774 | 53736 | 0.1491 | 0.1484 | 0.5104 | 0.5098 | 0.0538 | 0.0532 | 0.4225 | 0.4231 |
| 623.xalancbmk_s-700B | 4156 | 14913 | 2203 | 8042 | 0.2115 | 0.2502 | 0.4086 | 0.3809 | 0.1772 | 0.2385 | 0.5783 | 0.6083 |

`top-10 PC` is the dynamic-instruction share of the ten hottest PCs. `top-10 delta` is the share of observed load-line deltas held by the ten most frequent delta values.

## D. Default-capacity normal-prefetcher evidence

| trace | policy | IPC | issued | useful | useless | late | selected acc | timeliness | miss red |
|---|---|---|---|---|---|---|---|---|---|
| 602.gcc_s-734B | no_pref | 0.36800 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 602.gcc_s-734B | stride | 0.37006 | 659625 | 22623 | 44832 | 144 | 0.034003 | 0.993674 | 0.033669 |
| 602.gcc_s-734B | streamer | 0.43161 | 710180 | 373381 | 8717 | 227 | 0.526530 | 0.999392 | 0.911001 |
| 602.gcc_s-734B | ampm | 0.43069 | 2846149 | 373090 | 12600 | 604 | 0.131168 | 0.998384 | 0.916153 |
| 602.gcc_s-734B | spp | 0.42842 | 4490488 | 265316 | 1194 | 386 | 0.059166 | 0.998547 | 0.651850 |
| 602.gcc_s-734B | ipcp | 0.36796 | 153749 | 115 | 2367 | 195 | 0.000748 | 0.370968 | 0.000104 |
| 602.gcc_s-734B | sms | 0.42212 | 696632 | 319575 | 23394 | 200 | 0.458743 | 0.999375 | 0.797239 |
| 602.gcc_s-734B | sandbox | 0.43628 | 3839474 | 390102 | 32524 | 58 | 0.101838 | 0.999851 | 0.964845 |
| 602.gcc_s-734B | power7 | 0.43075 | 1592206 | 364759 | 12010 | 314 | 0.229092 | 0.999140 | 0.883550 |
| 605.mcf_s-994B | no_pref | 0.18318 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 605.mcf_s-994B | stride | 0.18353 | 903749 | 246 | 1016 | 1083 | 0.000272 | 0.185520 | 0.000332 |
| 605.mcf_s-994B | streamer | 0.18506 | 904522 | 33879 | 8680 | 1373 | 0.037455 | 0.961049 | 0.045771 |
| 605.mcf_s-994B | ampm | 0.18874 | 509840 | 90495 | 58586 | 1305 | 0.177874 | 0.985784 | 0.122252 |
| 605.mcf_s-994B | spp | 0.18874 | 622764 | 14701 | 4967 | 1346 | 0.023606 | 0.916121 | 0.019860 |
| 605.mcf_s-994B | ipcp | 0.18311 | 903751 | 0 | 25023 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 605.mcf_s-994B | sms | 0.18713 | 1389787 | 258164 | 661093 | 61568 | 0.185758 | 0.807439 | 0.075635 |
| 605.mcf_s-994B | sandbox | 0.16722 | 17207188 | 461994 | 7694072 | 138805 | 0.030772 | 0.768966 | 0.025754 |
| 605.mcf_s-994B | power7 | 0.18551 | 523176 | 40761 | 226935 | 6967 | 0.077912 | 0.854015 | 0.055067 |
| 619.lbm_s-4268B | no_pref | 0.32200 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 619.lbm_s-4268B | stride | 0.32465 | 294415 | 142 | 587626 | 26461 | 0.000482 | 0.005339 | 0.000476 |
| 619.lbm_s-4268B | streamer | 0.36368 | 294415 | 262433 | 0 | 25849 | 0.891371 | 0.910402 | 0.891340 |
| 619.lbm_s-4268B | ampm | 0.37483 | 2019581 | 496377 | 52 | 25216 | 0.245782 | 0.951656 | 0.801448 |
| 619.lbm_s-4268B | spp | 0.37927 | 3832728 | 190195 | 19 | 64941 | 0.049624 | 0.745465 | 0.232354 |
| 619.lbm_s-4268B | ipcp | 0.32200 | 294415 | 0 | 31410 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 619.lbm_s-4268B | sms | 0.38105 | 568017 | 535130 | 84 | 26099 | 0.942102 | 0.953497 | 0.870026 |
| 619.lbm_s-4268B | sandbox | 0.37443 | 3450777 | 488678 | 110 | 76726 | 0.201959 | 0.864299 | 0.713065 |
| 619.lbm_s-4268B | power7 | 0.35813 | 3273611 | 297329 | 8 | 93600 | 0.090826 | 0.760624 | 0.397839 |
| 620.omnetpp_s-874B | no_pref | 0.23750 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 620.omnetpp_s-874B | stride | 0.23784 | 394808 | 277 | 580 | 560 | 0.000702 | 0.330143 | 0.000878 |
| 620.omnetpp_s-874B | streamer | 0.39848 | 647061 | 171358 | 246304 | 1128 | 0.264824 | 0.993460 | 0.540662 |
| 620.omnetpp_s-874B | ampm | 0.39347 | 402867 | 45138 | 105103 | 2046 | 0.112042 | 0.956638 | 0.142830 |
| 620.omnetpp_s-874B | spp | 0.39202 | 195923 | 58 | 39 | 29 | 0.000296 | 0.666667 | 0.000184 |
| 620.omnetpp_s-874B | ipcp | 0.37917 | 394812 | 16216 | 2093 | 150 | 0.041079 | 0.990836 | 0.051333 |
| 620.omnetpp_s-874B | sms | 0.24695 | 808086 | 197737 | 394038 | 3655 | 0.244698 | 0.981851 | 0.224797 |
| 620.omnetpp_s-874B | sandbox | 0.24380 | 8064016 | 372994 | 2890951 | 10066 | 0.050890 | 0.973722 | 0.250191 |
| 620.omnetpp_s-874B | power7 | 0.39657 | 577539 | 184670 | 220120 | 1640 | 0.319754 | 0.991196 | 0.584341 |
| 623.xalancbmk_s-700B | no_pref | 0.35321 | 0 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 623.xalancbmk_s-700B | stride | 0.35335 | 1047219 | 102 | 411 | 536 | 0.000097 | 0.159375 | 0.000228 |
| 623.xalancbmk_s-700B | streamer | 0.32466 | 1458076 | 6957 | 10006 | 1296 | 0.004771 | 0.842902 | 0.015550 |
| 623.xalancbmk_s-700B | ampm | 0.34537 | 2197781 | 54302 | 94143 | 3397 | 0.024708 | 0.941125 | -0.135508 |
| 623.xalancbmk_s-700B | spp | 0.35391 | 1670201 | 7816 | 23108 | 1870 | 0.004846 | 0.806938 | 0.018647 |
| 623.xalancbmk_s-700B | ipcp | 0.31544 | 1047232 | 0 | 72485 | 0 | 0.000000 | 0.000000 | -0.141256 |
| 623.xalancbmk_s-700B | sms | 0.33861 | 2214542 | 295343 | 757315 | 12144 | 0.133365 | 0.960506 | -0.443592 |
| 623.xalancbmk_s-700B | sandbox | 0.33544 | 13366262 | 371624 | 3589232 | 15638 | 0.034077 | 0.959619 | -0.596643 |
| 623.xalancbmk_s-700B | power7 | 0.34286 | 1659130 | 140884 | 533779 | 6276 | 0.084914 | 0.957344 | -0.311841 |

## D. Default-capacity standalone replay evidence

| trace | variant | IPC | vs best normal | selected acc | timeliness | issued | useful | useless | late | unique event cov |
|---|---|---|---|---|---|---|---|---|---|---|
| 602.gcc_s-734B | v3_1 | 0.42863 | -0.01753 | 0.973754 | 0.998591 | 187803 | 182874 | 4068 | 258 | 0.899996 |
| 605.mcf_s-994B | v3_1 | 0.18862 | -0.00064 | 0.526061 | 0.801153 | 104121 | 54774 | 5304 | 13595 | 0.073720 |
| 619.lbm_s-4268B | v3_1 | 0.38492 | +0.01016 | 0.978263 | 0.984176 | 293605 | 287223 | 1732 | 4618 | 0.972339 |
| 620.omnetpp_s-874B | v3_1 | 0.24503 | -0.00778 | 0.454822 | 0.990821 | 169940 | 77286 | 63543 | 716 | 0.236165 |
| 623.xalancbmk_s-700B | v3_1 | 0.36407 | +0.02871 | 0.683155 | 0.987180 | 317571 | 216910 | 21844 | 2817 | 0.454340 |
| 602.gcc_s-734B | v3_3 | 0.42887 | -0.01698 | 0.980913 | 0.999002 | 186723 | 183158 | 3056 | 183 | 0.901364 |
| 605.mcf_s-994B | v3_3 | 0.18729 | -0.00768 | 0.431481 | 0.738338 | 118930 | 51316 | 5380 | 18186 | 0.069089 |
| 619.lbm_s-4268B | v3_3 | 0.38434 | +0.00863 | 0.985636 | 0.992914 | 293442 | 289227 | 2115 | 2064 | 0.978893 |
| 620.omnetpp_s-874B | v3_3 | 0.24559 | -0.00551 | 0.444369 | 0.992572 | 187366 | 83252 | 74440 | 623 | 0.255163 |
| 623.xalancbmk_s-700B | v3_3 | 0.37893 | +0.07070 | 0.751036 | 0.997001 | 570297 | 428219 | 25092 | 1288 | 0.899447 |

## E. Timely demand-miss overlap against the per-trace best normal baseline

| trace | variant | best normal | both timely | normal-only timely | standalone-only timely | both late | neither | selected but late | no earlier selected |
|---|---|---|---|---|---|---|---|---|---|
| 602.gcc_s-734B | v3_1 | sandbox | 180554 | 15340 | 2263 | 1 | 4933 | 248 | 14961 |
| 605.mcf_s-994B | v3_1 | ampm | 40159 | 17513 | 14410 | 80 | 653507 | 65 | 17080 |
| 619.lbm_s-4268B | v3_1 | sms | 250052 | 5218 | 36437 | 729 | 1457 | 3545 | 974 |
| 620.omnetpp_s-874B | v3_1 | sms | 38558 | 47449 | 36201 | 48 | 192359 | 209 | 46252 |
| 623.xalancbmk_s-700B | v3_1 | spp | 880 | 252 | 203732 | 39 | 242859 | 12 | 225 |
| 602.gcc_s-734B | v3_3 | sandbox | 180405 | 15489 | 2690 | 1 | 4508 | 182 | 15153 |
| 605.mcf_s-994B | v3_3 | ampm | 28390 | 29282 | 22751 | 230 | 640924 | 239 | 27960 |
| 619.lbm_s-4268B | v3_3 | sms | 252030 | 3240 | 36390 | 384 | 2023 | 1517 | 961 |
| 620.omnetpp_s-874B | v3_3 | sms | 40520 | 45487 | 40253 | 51 | 188378 | 180 | 44089 |
| 623.xalancbmk_s-700B | v3_3 | spp | 1012 | 120 | 404054 | 48 | 43997 | 7 | 101 |

`no earlier selected` means no earlier entry existed in the final frozen export. It is not proof that the candidate bank lacked the target; the notebook currently lacks the full candidate decision ledger needed to distinguish candidate absence from policy rejection.

## G. Completed frozen-list cache-capacity controls

| level | capacity | trace | no-pref IPC | best normal | v3.1 IPC | v3.1 - best | v3.3 IPC | v3.3 - best | winner |
|---|---|---|---|---|---|---|---|---|---|
| L1D | double (64 KiB) | 602.gcc_s-734B | 0.36799 | sandbox 0.43656 | 0.42855 | -0.00801 | 0.42880 | -0.00776 | v3_3 |
| L1D | double (64 KiB) | 605.mcf_s-994B | 0.18358 | spp 0.18923 | 0.18909 | -0.00014 | 0.18771 | -0.00152 | v3_1 |
| L1D | double (64 KiB) | 619.lbm_s-4268B | 0.32193 | sms 0.38112 | 0.38501 | +0.00389 | 0.38433 | +0.00321 | v3_1 |
| L1D | double (64 KiB) | 620.omnetpp_s-874B | 0.23810 | sms 0.24782 | 0.24554 | -0.00228 | 0.24604 | -0.00178 | v3_3 |
| L1D | double (64 KiB) | 623.xalancbmk_s-700B | 0.35519 | spp 0.35244 | 0.36617 | +0.01373 | 0.38029 | +0.02785 | v3_3 |
| L1D | half (16 KiB) | 602.gcc_s-734B | 0.36800 | sandbox 0.43628 | 0.42863 | -0.00765 | 0.42887 | -0.00741 | v3_3 |
| L1D | half (16 KiB) | 605.mcf_s-994B | 0.18276 | ampm 0.18817 | 0.18818 | +0.00001 | 0.18686 | -0.00131 | v3_1 |
| L1D | half (16 KiB) | 619.lbm_s-4268B | 0.32192 | sms 0.38103 | 0.38472 | +0.00369 | 0.38419 | +0.00316 | v3_1 |
| L1D | half (16 KiB) | 620.omnetpp_s-874B | 0.23617 | sms 0.24562 | 0.24363 | -0.00199 | 0.24415 | -0.00147 | v3_3 |
| L1D | half (16 KiB) | 623.xalancbmk_s-700B | 0.34960 | spp 0.35004 | 0.35953 | +0.00949 | 0.37369 | +0.02365 | v3_3 |
| L2C | double (512 KiB) | 602.gcc_s-734B | 0.36804 | sandbox 0.43629 | 0.42868 | -0.00761 | 0.42891 | -0.00738 | v3_3 |
| L2C | double (512 KiB) | 605.mcf_s-994B | 0.18376 | spp 0.19026 | 0.18933 | -0.00093 | 0.18808 | -0.00218 | v3_1 |
| L2C | double (512 KiB) | 619.lbm_s-4268B | 0.33043 | sms 0.39463 | 0.39783 | +0.00320 | 0.39741 | +0.00278 | v3_1 |
| L2C | double (512 KiB) | 620.omnetpp_s-874B | 0.23908 | sms 0.24865 | 0.24693 | -0.00172 | 0.24751 | -0.00114 | v3_3 |
| L2C | double (512 KiB) | 623.xalancbmk_s-700B | 0.38070 | spp 0.38065 | 0.38128 | +0.00063 | 0.38237 | +0.00172 | v3_3 |
| L2C | half (128 KiB) | 602.gcc_s-734B | 0.36787 | sandbox 0.43634 | 0.42851 | -0.00783 | 0.42874 | -0.00760 | v3_3 |
| L2C | half (128 KiB) | 605.mcf_s-994B | 0.18162 | ampm 0.18774 | 0.18747 | -0.00027 | 0.18599 | -0.00175 | v3_1 |
| L2C | half (128 KiB) | 619.lbm_s-4268B | 0.32184 | sms 0.37942 | 0.38340 | +0.00398 | 0.38302 | +0.00360 | v3_1 |
| L2C | half (128 KiB) | 620.omnetpp_s-874B | 0.23615 | sms 0.24545 | 0.24361 | -0.00184 | 0.24408 | -0.00137 | v3_3 |
| L2C | half (128 KiB) | 623.xalancbmk_s-700B | 0.32794 | sandbox 0.34172 | 0.34140 | -0.00032 | 0.35571 | +0.01399 | v3_3 |
| LLC | half (1024 KiB) | 602.gcc_s-734B | 0.36414 | sandbox 0.43597 | 0.42785 | -0.00812 | 0.42813 | -0.00784 | v3_3 |
| LLC | half (1024 KiB) | 605.mcf_s-994B | 0.17710 | ampm 0.18388 | 0.18320 | -0.00068 | 0.18195 | -0.00193 | v3_1 |
| LLC | half (1024 KiB) | 619.lbm_s-4268B | 0.32212 | sms 0.38156 | 0.38564 | +0.00408 | 0.38488 | +0.00332 | v3_1 |
| LLC | half (1024 KiB) | 620.omnetpp_s-874B | 0.23267 | sms 0.24225 | 0.24026 | -0.00199 | 0.24096 | -0.00129 | v3_3 |
| LLC | half (1024 KiB) | 623.xalancbmk_s-700B | 0.35303 | spp 0.35359 | 0.36387 | +0.01028 | 0.37868 | +0.02509 | v3_3 |

## Raw-source inventory retained in the review bundle

The uploaded bundle also contains the complete source tables that underlie this report: `run_unique_event_outcomes.csv` (55 rows), `normal_vs_standalone_target_attribution.csv` (90 rows), `normal_only_timely_standalone_reason.csv` (215 rows), and `top_residual_contexts.csv` (5,500 rows), plus all top-PC, top-delta, top-page, top-register, build-info, and RUN_INFO files for the included points. They were all read during review; they are not duplicated here because results/raw directories are intentionally untracked and this report is the tracked research record.

## Data-validity notes

- Do not compare frozen-list capacity control values to a hypothetical capacity-trained NN. The oracle and exports were built at baseline cache capacity.
- IPC is the decision metric. Accuracy, timeliness, useful count, and coverage explain behavior but cannot alone establish a winner.
- No result for `LLC double / 4 MiB` is included because the bundle lacks simulation summaries.