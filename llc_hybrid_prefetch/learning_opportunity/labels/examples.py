"""Export reproducible example queries from the midpoint of each indexed Stride stream."""

from measure import HERE, save
from sequence import HORIZONS, ROOT, SCOPES, Sequence


def main():
    anchors = []
    future = []
    windows = []
    for folder in sorted(ROOT.glob("*__stride")):
        if not (folder / "index.json").is_file():
            continue
        seq = Sequence(folder)
        case = seq.meta["case"]
        a = seq.size // 2
        last = 3 * seq.size // 4
        obs = seq.observations([a])[0]
        ctx = seq.context(a, 8)
        anchors.append(
            dict(
                case=case,
                anchor_rank=a,
                source_ordinal=int(obs[0]) >> 2,
                operand=int(obs[0]) & 3,
                pc=int(obs[1]),
                physical_line=int(obs[2]),
                cycle=int(obs[3]),
                exclusive_future_stop=last,
                pcs_through_trigger=" ".join(map(str, ctx["pc_sequence"])),
                lines_through_trigger=" ".join(map(str, ctx["physical_lines"])),
                cycle_gaps=" ".join(map(str, ctx["cycle_gaps"])),
            )
        )
        for scope in SCOPES:
            bundle = seq.future(a, scope=scope, stop=last)
            k = 0
            for h, rank, ok in zip(HORIZONS, bundle["ranks"], bundle["valid"]):
                row = dict(
                    case=case,
                    anchor_rank=a,
                    scope=scope,
                    horizon=h,
                    censored=int(not ok),
                    target_rank="",
                    source_ordinal="",
                    operand="",
                    target_pc="",
                    target_physical_line="",
                    target_cycle="",
                    lead_cycles="",
                    line_seen="",
                    page_seen="",
                    region_seen="",
                    nopf_mask="",
                    stride_mask="",
                    multi_mask="",
                    spp_mask="",
                )
                if ok:
                    key, pc, line, t = map(int, bundle["observations"][k])
                    row.update(
                        target_rank=int(rank),
                        source_ordinal=key >> 2,
                        operand=key & 3,
                        target_pc=pc,
                        target_physical_line=line,
                        target_cycle=t,
                        lead_cycles=int(bundle["lead_cycles"][k]),
                        line_seen=int(bundle["line_seen"][k]),
                        page_seen=int(bundle["page_seen"][k]),
                        region_seen=int(bundle["region_seen"][k]),
                    )
                    for arm, mask in zip(seq.meta["arms"], bundle["outcomes"][k]):
                        row[arm + "_mask"] = int(mask)
                    k += 1
                future.append(row)
        for width in (32, 128, 512):
            footprint = seq.time_window(a, width, stop=last)
            windows.append(
                dict(
                    case=case,
                    anchor_rank=a,
                    window_cycles=width,
                    complete=int(footprint["complete"]),
                    future_events=len(footprint["ranks"]),
                    distinct_physical_lines=" ".join(map(str, footprint["lines"])),
                )
            )
    save(HERE / "sample_anchors.csv", anchors)
    save(HERE / "sample_targets.csv", future)
    save(HERE / "sample_footprints.csv", windows)


if __name__ == "__main__":
    main()
