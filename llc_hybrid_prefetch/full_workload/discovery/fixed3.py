"""Full fixed +3 falsification of the BFS-3 witness and of the associated performance claim."""

from pattern import HERE, table, save
from streams import location, events, keys
from score_replay import compressed_join, accepted
from characterize import target_kind

CASE = "bfs_discovery"
KEY = (100101506, 0)


def main():
    paths = {
        p: location(CASE, p, "full")
        for p in ("nopf", "stride", "multi", "spp", "shape", "sparse3")
    }
    if any(not (f / "complete.json").exists() for f in paths.values()):
        raise ValueError("wait for the full fixed +3 control and matched native arms")
    results = []
    for p, f in paths.items():
        z = events(CASE, p, "full")
        idx = dict(zip(keys(z), range(len(z["t"]))))[KEY]
        assert z["pa"][idx] == 3500606
        outcome = target_kind(z, idx)
        use = accepted(f).get(KEY, {})
        core = table(f / "cores.csv")[0]
        results.append(
            dict(
                case=CASE,
                arm=p,
                fixed_work_cycles=core["cycles"],
                target_instr=KEY[0],
                target_line=int(z["pa"][idx]),
                outcome=outcome,
                producer_instr=use.get("cue"),
                producer_line=use.get("line"),
                independent_source=use.get("source"),
                generation=use.get("gen"),
                dispatch=use.get("dispatch"),
                fill=use.get("fill"),
                ingress=int(z["t"][idx]),
                lookup=int(z["tag"][idx]),
                lead=int(z["t"][idx]) - use["gen"] if use else None,
                head=int(z["head"][idx]),
            )
        )
    save(HERE / "bfs_fixed3_witness.csv", results, list(results[0]))
    rows = []
    for p in ("stride", "multi", "spp", "shape"):
        paths_for_join = {
            "nopf": paths["nopf"],
            "alternative": paths[p],
            "shape": paths["sparse3"],
        }
        counts = compressed_join(paths_for_join, "alternative")
        rows.append(dict(actual_arm="sparse3", reference=p, **counts))
    save(HERE / "bfs_fixed3_alignment.csv", rows, list(rows[0]))


if __name__ == "__main__":
    main()
