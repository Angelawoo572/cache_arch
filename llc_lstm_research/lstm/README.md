# 605 LLC LSTM experiment

- **Input.** Accepted L2D demand-load misses at LLC ingress. Features use the load PC and translated line. A 32-event per-core directory produces at most 16 legal candidates in previously seen physical pages. All three scorers use the same directory. The offline NoPF log keeps events that complete an LLC lookup. At runtime the model observes ingress directly.
- **Label.** A candidate is positive when its first later same-core demand is an LLC miss in NoPF with 300–2,000 cycles of lead. The first 128 events initialize the directory. The label exporter masks incomplete follow-up at the end of each core's ROI. Both cores have separate address spaces in these traces.
- **Split.** Train on 10M, 30M, and 50M. Select checkpoints and thresholds on 70M. Test on 90M and 110M. The numbers denote initial trace records skipped. Each region runs 200,000 warmup instructions and 2 million measured instructions per core.
- **Model.** One causal LSTM layer with eight hidden units per core and shared weights. A four-value query combines the current input and hidden state. Each candidate has a projected key and scalar bias, so state can change the order of the scores. The event-only control shares the scoring design and uses no recurrent input. The page-PC table is fit on the same NoPF data.
- **Output.** Use one signed integer logit per candidate, scaled by 1,024. The simulator compares scores directly. It applies a threshold and issues at most one eligible request. LSTM weights and gates use float32. This prototype does not estimate RTL timing.
- **Comparison.** Offline budgets count decisions on frozen NoPF events. ChampSim replays each policy with its own cache and traffic. Runtime issue counts can differ even with a common cap. Use [metric definitions](../PERFORMANCE.md) for direct coverage, timeliness, IPC, and completion time.

## Reproduce

- Use Python with NumPy and PyTorch. From `llc_lstm_research`, build the NPZs from the NoPF event logs and use the fixed split below. Keep raw data and checkpoints under `lstm/runs/`.

```sh
for skip in 10 30 50 70 90 110; do
  python3 lstm/dataset.py export \
    "conventional/runs/long_memory/605_nopf_${skip}m" \
    "lstm/runs/data/605_${skip}m.npz"
done
python3 lstm/dataset.py table lstm/runs/data/page_pc.txt \
  lstm/runs/data/605_10m.npz lstm/runs/data/605_30m.npz lstm/runs/data/605_50m.npz
python3 lstm/train.py \
  --train lstm/runs/data/605_10m.npz lstm/runs/data/605_30m.npz lstm/runs/data/605_50m.npz \
  --validation lstm/runs/data/605_70m.npz \
  --test lstm/runs/data/605_90m.npz lstm/runs/data/605_110m.npz \
  --page-pc lstm/runs/data/page_pc.txt --output lstm/runs/reproduction
```

- The portable `.txt` files load into ChampSim. Build using `JOBS=14 ./conventional/build.sh 2 conventional/runs/build-2core-lstm`. Use the validation cutoffs below for the original checkpoint. Use a new output directory for each sensitivity run. Set the latency and interval to 64 cycles for the latency check or the issue limit to 150 for the equal-request check.

```sh
colab=lstm/runs/experiments/mcf_605_colab
model="$colab/training"
trace=../traces/605.mcf_s-994B.champsimtrace.xz
for skip in 90 110; do
  run=conventional/runs/lstm_605/colab_repro_${skip}m
  python3 conventional/run.py --binary conventional/runs/build-2core-lstm/pythia/bin/champsim \
    --output "$run" --skip "${skip}000000" --warmup 200000 --instructions 2000000 \
    --degree 1 --events --policy nopf --policy page_pc --policy event --policy lstm \
    --page-pc-model "$colab/page_pc.txt" --page-pc-threshold=-1215 \
    --event-model "$model/event.txt" --event-threshold=-1043 \
    --lstm-model "$model/lstm.txt" --lstm-threshold=-1037 \
    --prediction-latency 16 --prediction-interval 16 --issue-limit 1000 "$trace" "$trace"
  python3 lstm/compare.py "conventional/runs/long_memory/605_nopf_${skip}m" "$run" \
    --output "$colab/replay_${skip}m_16cycles.csv"
done
```

- The [results](RESULTS.md) report the training and replay metrics. On both local and Sacramento copies, `lstm/runs/experiments/mcf_605_colab/` holds the Colab model weights, metrics, and logs. `mcf_605_training/` holds the local replication. `conventional/runs/lstm_605/` holds the simulator logs, event traces, and configurations.
- Run `python3 lstm/reach.py conventional/runs/long_memory/605_nopf_{90,110}m --output lstm/runs/reach_check` to reproduce the candidate and lead-window counts.
