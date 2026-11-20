# One BFS executable, two named graphs

We run the same compiled queue-based breadth-first traversal on Stanford's web crawl and Google's web graph. We retain original vertex identifiers, add the reverse of every input edge and preserve neighbor order when constructing the CSR arrays. The starting vertex has the highest resulting degree (lowest ID on a tie). Each trace ends after examining 400,000 edges or exhausting the component. Initialization and CSR construction precede the recorded kernel. The emulator records x86 instructions, branch outcomes, register dependencies and load/store operands. An independent queue traversal checks all discovered vertices in order. [Stanford web graph](https://snap.stanford.edu/data/web-Stanford.html) · [Google web graph](https://snap.stanford.edu/data/web-Google.html)

The first 250,000 recorded instructions warm each replay. The following 2,000,000 form the measured interval under four conventional policies. Stanford is the designated donor and Google is the input-transfer test. The controlled kernel and shorter interval provide a named input contrast alongside the long published GAP recordings. All observations refer to this traversal, starting vertex and source window. Physical arrays use the same virtual base addresses in both executions.

Use the Python dependencies in the earlier [compiled tracer](../../opportunity_study/workloads/README.md), Clang, NumPy, and the full-workload continuous simulator. Commands run from the cache root.

```sh
p=llc_hybrid_prefetch/learning_opportunity
python3 -B "$p/graph_inputs/record.py" snap_stanford
python3 -B "$p/graph_inputs/record.py" snap_google
python3 -B "$p/graph_inputs/run.py"
```
