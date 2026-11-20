# Google service and AI inference transfer

Use the smallest published archive in each of the earlier discovery groups for Google gtrace_v2 and AI/ML. Archive size is the selection criterion. It uses no cache measurements. The original study screened early, short intervals of these discovery sources. A frozen SPEC/GAP probe can test its vocabulary much later in each recording without tuning on these domains. Previously reserved Google and AI/ML recordings remain available for a future hardware evaluation. [Frozen source register](sources.csv) · [Original discovery and reserve assignments](../../workload_expansion/intake/registry.csv) · [DPC4 distribution](https://github.com/CMU-SAFARI/DPC4)

Use the same complete-archive validation and 50M warmup plus 200M measured protocol as the external SPEC/GAP panel. Each view has all four policy outcomes and a continuous LLC admission stream. This small domain check is reported separately from the eleven-recording comparison. Google service labels do not identify an application source line. Register and branch fields are those supplied by the converted trace. [Interface and dependency limitations](../../workload_expansion/screen/README.md)

On Charlie, SPP covered 33.70% of original misses on time and reduced completion from 421.49M to 399.24M cycles. On Llama2-stories15M, SPP covered 19.91% on time but increased completion from 275.84M to 286.64M cycles. The frozen address probes also behaved differently across the two domains. [Matched policy counts](results/performance.csv) · [Held-out forecasts](../probes/domain_transfer.csv)

```sh
p=llc_hybrid_prefetch/learning_opportunity
python3 -B "$p/domains/run.py" --jobs 3 --decoders 1
```

Run `python3 -B llc_hybrid_prefetch/learning_opportunity/domains/summarize.py` to regenerate the eight policy comparisons from the recorded outcomes.
