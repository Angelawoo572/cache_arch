"""Check that published comparisons, fixed-work inputs, indices and citations agree."""

from urllib.parse import unquote
import re
from catalog import HERE, register, rows


def check_links():
    docs = [
        p
        for folder in HERE.iterdir()
        if folder.is_dir() and folder.name != "raw"
        for p in folder.glob("*.md")
    ]
    docs.extend(HERE.glob("*.md"))
    checked = 0
    for doc in docs:
        text = doc.read_text()
        if re.search(r"[\u3400-\u9fff]", text):
            raise ValueError(f"Expected English prose in {doc.name}")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            path = unquote(link.split("#")[0])
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            if not (doc.parent / path).exists():
                raise ValueError(f"{doc.name} links to missing {path}")
            checked += 1
    return checked


def check():
    actual = rows(HERE / "data_registry.csv")
    expected = list(register())
    assert actual == [{k: str(v) for k, v in row.items()} for row in expected]
    distinct = {(r["case"], r["policy"]) for r in actual}
    assert len(distinct) == len(actual) == 104
    assert sum(bool(r["rich_run"]) for r in actual) == 47
    expansion = rows(HERE / "expansion/results/performance.csv") + rows(
        HERE / "domains/results/performance.csv"
    )
    assert len(expansion) == 68
    idx = {(r["case"], r["policy"]): r for r in actual}
    for r in expansion:
        view = idx[r["case"], r["policy"]]
        for key1, key2 in (
            ("warmup", "warmup"),
            ("measured_instructions", "measured"),
            ("llc_demand_ingress", "llc_ingress"),
        ):
            assert int(r[key1]) == int(view[key2])
        assert int(r["proposed"]) == int(r["accepted"]) + int(r["rejected"])
        assert int(r["nopf_original_misses"]) == sum(
            int(r[k])
            for k in (
                "orig_to_timely",
                "orig_to_late",
                "orig_to_ordinary_hit",
                "orig_to_persistent_miss",
                "orig_without_arm_tag",
            )
        )
    labels = rows(HERE / "labels/datasets.csv")
    assert {(r["case"], r["policy"]) for r in labels} >= distinct
    effects = rows(HERE / "trials/effects.csv")
    assert len(effects) == 6 and all(
        int(r["matched"]) <= int(r["planned"]) for r in effects
    )
    assert all(
        int(r["cycle_savings"]) == int(r["baseline_cycles"]) - int(r["treated_cycles"])
        for r in effects
    )
    successful = rows(HERE / "probes/prospective_offers.csv")
    interventions = rows(HERE / "one_offer/effects.csv")
    assert {r["scenario"] for r in successful} == {r["scenario"] for r in interventions}
    pressure = rows(HERE / "resources/at_ingress.csv")
    assert len(pressure) == 36
    assert len(rows(HERE / "resources/external_at_ingress.csv")) == 11
    timing = rows(HERE / "resources/request_service.csv")
    assert len(timing) == 18
    for r in timing:
        assert (
            int(r["earlier_slack_ge32"])
            <= int(r["earlier_slack_ge8"])
            <= int(r["earlier_slack_ge0"])
            <= int(r["comparable_service"])
        )
    links = check_links()
    print(
        f"Verified {len(actual)} history/label joins, {len(expansion)} expansion metrics, six oracle timing arms, "
        f"{len(interventions)} prospective replay, {len(timing)} timing cohorts and {links} local document links."
    )


if __name__ == "__main__":
    check()
