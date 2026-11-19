"""Check first references of native timely examples against the original trace from ordinal one."""

import csv
import gzip
from collections import defaultdict
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
ROOT = FULL.parents[1]


def rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    exe = FULL / "raw/native/first_reference"
    subprocess.run(
        ["c++", "-O3", "-std=c++14", str(HERE / "first_reference.cc"), "-o", str(exe)],
        check=True,
    )
    examples = rows(FULL / "discovery/full_native_examples.csv")
    cases = defaultdict(list)
    for row in examples:
        case = row["case"]
        if case.endswith("_far"):
            continue
        cases[case].append(row)
    out = []
    for case, subset in cases.items():
        folder = FULL / "raw/native/runs" / (case + "__nopf")
        with gzip.open(folder / "pages.csv.gz", "rt") as f:
            reverse = {
                int(r["physical_page"]): int(r["virtual_page"])
                for r in csv.DictReader(f)
            }
        end = max(int(r["target_instr"]) for r in subset)

        def virtual_line(r):
            line = int(r["target_line"])
            return (reverse[line >> 6] << 6) + (line & 63)

        requested = set(map(virtual_line, subset))
        trace = FULL / "raw/traces" / (case + ".champsimtrace")
        p = subprocess.run(
            [str(exe), str(trace), str(end)],
            input="\n".join(map(str, requested)),
            capture_output=True,
            text=True,
            check=True,
        )
        first = dict(
            map(lambda line: tuple(map(int, line.split())), p.stdout.splitlines())
        )
        for r in subset:
            cursor, target = int(r["cue_instr"]), int(r["target_instr"])
            earliest = first[virtual_line(r)]
            if not 0 < earliest <= target:
                raise ValueError("Native target absent at or before demand ordinal")
            if earliest <= cursor:
                status = "data_reference_at_or_before_anchor_ordinal"
            elif earliest <= cursor + 256:
                status = "new_in_program_order_ingress_order_uncertain"
            else:
                status = "first_data_reference_beyond_256_entry_ROB"
            out.append(
                dict(
                    case=case,
                    actual_arm=r["actual_arm"],
                    prior_arm=r["prior_arm"],
                    cue_instr=cursor,
                    target_instr=target,
                    target_operand=r["target_op"],
                    target_physical_line=r["target_line"],
                    target_virtual_line=virtual_line(r),
                    first_data_reference_ordinal=earliest,
                    classification=status,
                )
            )
    with (HERE / "first_reference_examples.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=out[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
