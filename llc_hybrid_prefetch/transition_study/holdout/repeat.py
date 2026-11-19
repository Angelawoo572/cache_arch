"""Repeat the four registered arms in a separate directory and check compact results."""
from concurrent.futures import ThreadPoolExecutor
import run

cases = ("gap_bfs3", "gap_bfs8")
run.RAW = run.RAW / "repeat"
with ThreadPoolExecutor(4) as pool:
    jobs = [pool.submit(run.run, case, mode) for case in cases for mode in ("nopf", "stride")]
    for job in jobs:
        job.result()
with ThreadPoolExecutor(2) as pool:
    measured = list(pool.map(run.summarize, cases))
expected = run.table(run.HERE / "results.csv")
assert len(measured) == len(expected)
for r, ref in zip(measured, expected):
    assert {k: str(v) for k, v in r.items()} == ref, (r, ref)
print("Reproduced all four native arms and both aligned summaries in an independent run directory.")
