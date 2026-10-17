#!/usr/bin/env python3
"""Export the actual frozen Streamer checkpoint through the shared LV602R1 writer."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
from streamer_reference import (ROOT, REVISION, TRACE, checked_output, configure_runtime,
                                literal_graph_cost, load_checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--allow-legacy-metadata", action="store_true")
    args = parser.parse_args()
    configure_runtime(args.runtime_dir)
    out = checked_output(args.out_dir)
    if (out / "model.bin").exists() or (out / "model_metadata.json").exists():
        raise RuntimeError("Preserve existing export; choose a new output directory")
    model, checkpoint, source, legacy = load_checkpoint(args.checkpoint, args.run_metadata, args.seed, args.allow_legacy_metadata)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from formal_NN_training.common.live602_runtime.model_format import canonicalize_poisson_lstm, read_model, write_model
    tensors = canonicalize_poisson_lstm(model.state_dict())
    out.mkdir(parents=True, exist_ok=True)
    config = write_model(out / "model.bin", tensors, hidden_size=model.model_size, feature_count=64,
                         mixture_components=4, fill_classes=0, count_mode=0, decoder_initial=0,
                         kind_feature=0, demand_kind=0, delta_limit_bits=57, fill_level0=2, fill_level1=4, delta_mode=0)
    actual, values = read_model(out / "model.bin")
    if config["parameter_count"] != checkpoint["parameters"]:
        raise RuntimeError("Export changed parameter count")
    for name, value in tensors.items():
        if list(value.reshape(-1).tolist()) != values[name]["data"]:
            raise RuntimeError("Export altered FP32 values: " + name)
    metadata = dict(schema_version=1, policy="streamer", experiment_revision=REVISION,
                    checkpoint_path=str(args.checkpoint.resolve()), training_metadata_path=str(args.run_metadata.resolve()),
                    seed=args.seed, seed_identity_source="original_external_training_metadata" if legacy else "checkpoint_and_metadata",
                    legacy_checkpoint_without_seed=legacy, format="LV602R1", configuration=config,
                    input_fields=["line_aligned_byte_address"], runtime_feature_count=64,
                    input_encoding="LSB-first uint64 bits of cache_line_address << 6; PC is not an input",
                    state_scope="one_global_chronological_h_c_stream", weights_frozen=True, online_weight_training=False,
                    output_semantics="floor(softplus count) then free-running GRU mixture-logit argmax mean; signed expm1 ties-even delta; wrap58; interface FILL_L2",
                    learned_fill_head=False, learned_fill_classes=0, neural_degree_cap=None, neural_same_page_filter=False,
                    source_training_trace=TRACE, source_training_instruction_interval=[0, 20000000],
                    source_training_rows=source["train_rows"], source_legacy_evaluation_instruction_interval=[25000000, 50000000],
                    source_legacy_evaluation_rows=source["eval_rows"], source_legacy_evaluation_previously_inspected=True,
                    source_legacy_evaluation_state="independent_zero_then_continuous_global_state",
                    validation_selection="not performed by export; original Stride5M..10M validation overlaps this checkpoint training",
                    source_label_role="Streamer actions are offline supervision only", python_reference="streamer_reference.StreamerReference",
                    parameter_count=config["parameter_count"], weight_bytes=config["weight_bytes"],
                    global_recurrent_state_bytes=model.model_size*8, literal_graph_cost=literal_graph_cost(model.model_size, 0),
                    backend_cost_formula=dict(base_macs="4H(64+H)+H", positive_count_initial_macs="H^2 if K>0", per_action_macs="3H(1+H)+12H", base_nonlinears="5H+1", positive_count_initial_nonlinears="H if K>0", per_action_nonlinears="3H+4+1", dead_initial_skip_when_K0=True, scale_head_computed=True, final_action_gru_advance=True),
                    finite_cost_scope="Backend actual operation counters and parent serial engine service/queue controls must be recorded separately")
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS", model=str(out / "model.bin"), parameters=config["parameter_count"], seed=args.seed)))


if __name__ == "__main__":
    main()
