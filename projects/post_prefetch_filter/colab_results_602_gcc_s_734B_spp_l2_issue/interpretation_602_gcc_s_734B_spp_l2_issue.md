# Colab result interpretation

trace: 602_gcc_s_734B
scope: spp_l2_issue

## Meaning

This run tests whether candidate-time features can filter SPP L2-scope candidates.

Important metrics:
- issued_ratio: fraction of candidates admitted by the learned filter
- accuracy: useful admitted / admitted
- useful_kept_ratio: useful admitted / all useful candidates
- bad_suppressed_ratio: useless candidates suppressed / all useless candidates
- estimated_reward: offline utility proxy, not final IPC

For 605.mcf, the key observation is that F0_candidate is already strong:
- it admits only a small fraction of candidates
- admitted candidates have much higher useful rate
- most useful candidates are still kept
- most bad candidates are suppressed

This means 605 behaves like a LOW_TRUST_SPP / FILTER_BAD_GENERATOR case.
