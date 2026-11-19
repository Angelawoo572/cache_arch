"""Pre-registered finite workload matrix, independent of measured outcomes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
TRACES = RAW / "traces"
BINARY = ROOT.parent / "raw/build-2/pythia/bin/champsim"
# case, dependency chain length (instructions), concurrent register chains, split
CONTROLS = [
    ("row_period2_tight", 0, 1, "development"),
    ("row_period2_paced", 192, 1, "development"),
    ("row_period5_paced", 192, 1, "held_out_schedule"),
    ("row_previous_paced", 192, 1, "held_out_schedule"),
    ("row_independent_paced", 192, 1, "held_out_negative"),
    ("streams_onepc_paced", 128, 1, "development"),
    ("streams_pc_paced", 128, 1, "held_out_pc_control"),
    ("crosspc_cue_paced", 128, 1, "development"),
    ("crosspc_mask_paced", 128, 1, "held_out_mask"),
    ("motif_noise_paced", 128, 1, "held_out_pattern"),
    ("phase_flip_paced", 128, 1, "held_out_pattern"),
    ("steady_paced", 192, 1, "development"),
    ("ramp_mlp1", 8, 1, "development"),
    ("ramp_mlp4", 8, 4, "held_out_mlp"),
    ("page_same_paced", 192, 1, "development"),
    ("page_cross_paced", 192, 1, "held_out_page"),
    ("ws_small", 0, 4, "development"),
    ("ws_large", 0, 4, "held_out_capacity"),
]
FOLLOWUP_CONTROLS = [
    ("row_hierarchy_paced", 192, 1, "held_out_hierarchy"),
    ("streams_three_switch_paced", 128, 1, "held_out_switch"),
    ("streams_three_identified_paced", 128, 1, "held_out_stream_control"),
    ("xor_context_paced", 128, 1, "held_out_joint_context"),
    ("xor_maskfirst_paced", 128, 1, "held_out_mask"),
]


def specs():
    for name, gap, chains, split in CONTROLS + FOLLOWUP_CONTROLS:
        warm, roi = (98304, 196608) if name.startswith("ws_") else (2048, 8192)
        period = 4 + gap
        yield dict(
            **(
                {"registration": "followup"}
                if (name, gap, chains, split) in FOLLOWUP_CONTROLS
                else {}
            ),
            case=name,
            group="synthetic",
            split=split,
            traces=[name],
            workloads=[name],
            warmup=warm * period,
            roi=roi * period,
            skip=0,
            slots=[0],
            period=period,
            chains=chains,
            warm_loads=warm,
            roi_loads=roi,
        )
    for case, split, ts, ws, slots in [
        ("gap_bfs3", "development", ["bfs-3"], ["bfs"], [0]),
        ("gap_bfs8", "held_out_recording", ["bfs-8"], ["bfs"], [0]),
        ("lbm_peer", "paired_solo", ["619.lbm_s-3766B"], ["lbm"], [1]),
        (
            "gap_bfs8_lbm",
            "held_out_interference",
            ["bfs-8", "619.lbm_s-3766B"],
            ["bfs", "lbm"],
            [0, 1],
        ),
    ]:
        yield dict(
            case=case,
            group="real",
            split=split,
            traces=ts,
            workloads=ws,
            warmup=1000000,
            roi=2000000,
            skip=0,
            slots=slots,
        )
    yield dict(
        case="mcf994_adjacent",
        group="real",
        split="held_out_time",
        traces=["605.mcf_s-994B"],
        workloads=["mcf"],
        warmup=10000000,
        roi=5000000,
        skip=5000000,
        slots=[0],
    )
    for index in (3, 8):
        yield dict(
            case=f"gap_bfs{index}_late",
            group="real",
            registration="followup",
            split="held_out_later_time",
            traces=[f"bfs-{index}-long"],
            workloads=["bfs"],
            warmup=1000000,
            roi=2000000,
            skip=49000000,
            slots=[0],
        )
    yield dict(
        case="compiled_rows",
        group="compiled",
        split="held_out_compilation",
        traces=["compiled_rows"],
        workloads=["compiled_rows"],
        warmup=40000,
        roi=160000,
        skip=0,
        slots=[0],
    )


if __name__ == "__main__":
    import json

    print(json.dumps(list(specs()), indent=2))
