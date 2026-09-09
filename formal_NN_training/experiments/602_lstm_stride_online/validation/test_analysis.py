"""Counter/interval fixtures; these are software validation, not research runs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "python/analyze.py"
SPEC = importlib.util.spec_from_file_location("online_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class MetricFixtures(unittest.TestCase):
    def test_exact_legacy_meanings(self):
        c = dict(instructions=1000, cycles=2000, l2_loads=100, l2_load_miss=30,
                 pf_requested=20, pf_issued=18, pq_merged_duplicate_proxy=3,
                 pf_useful=9, pf_late=3)
        m = analysis.metrics(c, {"l2_load_miss": 50})
        self.assertEqual(m["useful_prefetch_coverage"], .18)
        self.assertEqual(m["miss_reduction"], .4)
        self.assertEqual(m["raw_accuracy"], .5)
        self.assertEqual(m["selected_accuracy"], .6)
        self.assertEqual(m["legacy_timeliness"], .75)
        self.assertEqual(m["request_pressure"], .2)
        self.assertEqual(m["ipc"], .5)
        self.assertEqual(m["l2_mpki"], 30)

    def test_undefined_ratios_are_na_not_zero(self):
        c = {key: 0 for key in analysis.COUNTERS + ("instructions", "cycles")}
        self.assertTrue(all(v is None for v in analysis.metrics(c, c).values()))

    def test_request_crosses_window_then_timely(self):
        # An issue belongs to window 1; first useful demand belongs to window 2.
        first = {key: 0 for key in analysis.COUNTERS + analysis.PROGRESS}
        first.update(instructions=1_000_000, cycles=2_000_000,
                     pf_requested=1, pf_issued=1, l2_loads=20, l2_load_miss=4)
        second = dict(first, instructions=2_000_000, cycles=4_000_000,
                      pf_useful=1, l2_loads=40, l2_load_miss=8)
        wins = analysis.windows([first, second])
        self.assertEqual(analysis.metrics(wins[0])["raw_accuracy"], 0)
        # No new issue in window 2: undefined interval accuracy, not failure.
        self.assertIsNone(analysis.metrics(wins[1])["raw_accuracy"])
        self.assertEqual(wins[1]["pf_useful"], 1)

    def test_timely_late_unused_unresolved_remain_distinct(self):
        counts = {"enqueued": 7, "timely": 1, "late": 1, "unused": 1,
                  "pending": 2, "resident_unused": 2}
        censored = counts["pending"] + counts["resident_unused"]
        self.assertEqual(censored, 4)
        self.assertEqual(sum(counts[k] for k in ("timely", "late", "unused")) + censored,
                         counts["enqueued"])
        # Legacy timeliness is U/(U+L), with no unused or pending terms.
        self.assertEqual(analysis.metrics({"pf_useful": 1, "pf_late": 1})["legacy_timeliness"], .5)

    def test_first_sustained_needs_three_consecutive(self):
        rows = [{"window": i + 1, "saving": v} for i, v in enumerate([2, -1, 3, 4, 5])]
        first, confirm = analysis.first_sustained(rows, lambda r: r["saving"] > 0)
        self.assertEqual((first["window"], confirm["window"]), (3, 5))
        self.assertIsNone(analysis.first_sustained(rows[:4], lambda r: r["saving"] > 0))

    def test_positive_window_does_not_recover_startup(self):
        rows = [{"window": i, "window_saved": 10, "cumulative_saved": -40 + i * 10}
                for i in range(1, 4)]
        self.assertIsNotNone(analysis.first_sustained(rows, lambda r: r["window_saved"] > 0))
        self.assertIsNone(analysis.first_sustained(rows, lambda r: r["cumulative_saved"] > 0))

    def test_normalizes_observer_without_redefining_counter(self):
        row = analysis.normalize(dict(NL=10, M=4, P=5, Ipf=4, Q=1, U=2, L=1,
                                      instructions=100, cycles=200, capacity_lines=8))
        self.assertEqual(row["l2_loads"], 10)
        self.assertEqual(row["pq_merged_duplicate_proxy"], 1)
        self.assertEqual(row["l2_capacity_lines"], 8)

    def test_occupancy_exact_first_attainment_and_not_reached(self):
        snapshots = [dict(instructions=100, cycles=200, occupancy_lines=4, l2_capacity_lines=8)]
        exact = {"50": {"instructions": 62, "cycles": 124}}
        rows = analysis.occupancy_milestones(snapshots, exact)
        self.assertEqual(rows[0]["instructions"], 62)
        self.assertEqual(rows[0]["measurement"], "exact fill event")
        self.assertEqual(rows[-1]["status"], "NOT REACHED")

    def test_missing_snapshot_does_not_infer_future_counter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "snapshots.jsonl"
            p.write_text('{"event":"snapshot","instructions":100,"cycles":200,"U":1}\n{"event":"snapshot"')
            rows, warnings = analysis.read_snapshots(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pf_useful"], 1)
            self.assertEqual(len(warnings), 1)

    def test_arm_identity(self):
        identity = analysis.identity(dict(arm="frozen_i1m", h=8, macs=4, state_capacity=64,
                                          protocol="cold", skip_instructions=25_000_000), "example")
        self.assertEqual(identity["arm"], "frozen")
        self.assertEqual(identity["checkpoint"], "i1m")
        self.assertEqual(identity["service_case"], 4)

    def test_real_parser_and_end_to_end_fixture(self):
        # Deliberately named software_fixture; these numbers never enter plots
        # or final cold/historical comparison tables.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, cycles_per_window in (("none", 4_000_000), ("stride", 3_000_000), ("online_random", 2_000_000)):
                run = root / name
                run.mkdir()
                (run / "run.json").write_text(json.dumps(dict(arm=name, protocol="software_fixture", h=8, seed=7, macs=0, state_capacity=0)))
                final = {"NL": 300, "M": 60, "P": 30, "Ipf": 30, "Q": 0, "U": 15, "L": 3, "pf_filled": 18, "pf_useless": 0, "pf_dropped": 0}
                names = {"NL": "loads", "M": "load_miss", "P": "prefetch_requested", "Ipf": "prefetch_issued", "Q": "pq_merged", "U": "prefetch_useful", "L": "prefetch_late", "pf_filled": "prefetch_filled", "pf_useless": "prefetch_useless", "pf_dropped": "prefetch_dropped"}
                lines = [f"Core_0_L2C_{names[k]} {v}" for k, v in final.items()]
                lines += [f"Finished CPU 0 instructions: 3000000 cycles: {3*cycles_per_window} cumulative IPC: 0.5", "Core_0_instructions 3000000", f"Core_0_cycles {3*cycles_per_window}"]
                (run / "run.log").write_text("\n".join(lines))
                samples = []
                for i in range(4):
                    samples.append(dict(event="snapshot", instructions=i*1_000_000, cycles=i*cycles_per_window, **{k: v*i/3 for k, v in final.items()}))
                (run / "snapshots.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
            payload = analysis.analyze(root, root / "tables")
            self.assertEqual(payload["completed_runs"], 3)
            attained = next(m for m in payload["learning_milestones"] if m["run_id"] == "online_random" and m["milestone"] == "positive_cumulative_cycle_saving_vs_stride")
            self.assertEqual(attained["first_window_end"], 1_000_000)
            self.assertEqual(attained["confirmed_at_instructions"], 3_000_000)


if __name__ == "__main__":
    unittest.main()
