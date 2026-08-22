# Report build

The tracked TeX file contains the contract and conclusion structure but no
invented measurements. compare_offline_live.py writes the actual conclusion
table to the ignored run directory.

From this directory:

~~~bash
RUN_ID=602_gcc_stride_prefix_seed7
pdflatex "\def\RunDir{../runs/$RUN_ID}\input{602_stride_training_budget_live_inference.tex}"
pdflatex "\def\RunDir{../runs/$RUN_ID}\input{602_stride_training_budget_live_inference.tex}"
~~~

If live results are absent, the PDF explicitly reports that execution is still
required rather than filling unavailable metrics with zero.
