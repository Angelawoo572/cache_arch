# Synthetic parity fixtures

`compare_python_cpp_outputs.py --synthetic` generates deterministic float32
models and events at runtime. No checkpoint or `model.bin` is tracked.

The generated cases cover silent output, K=1, K=2, K=4, positive/negative/zero
deltas, free-running multi-slot decoding, new/repeated/interleaved PCs, a large
line address, wraparound at the 58-bit line-address boundary, and nontrivial
recurrent-state evolution.
