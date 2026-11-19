"""Cross-check the independent parser against code compiled with Pythia's native input type."""
import argparse
import csv
import os
import subprocess
from register import HERE, STUDY, write_csv
from audit import inputs

RAW = STUDY / "raw/intake"


def main(long_google=False):
    discovery, windows, length, prefix = inputs(long_google)
    executable = RAW / (prefix + "check_reader")
    source = STUDY.parent / "raw/build-2/pythia/inc"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++14", "-O2", "-I", str(source),
                    str(HERE / "check_reader.cc"), "-o", str(executable)], check=True)
    with (HERE / (prefix + "format_audit.csv")).open() as f:
        expected = {(r["case"], r["window"]): r for r in csv.DictReader(f)}
    result = []
    for row, path in discovery:
        for window, (start, stop) in windows.items():
            output = subprocess.check_output([str(executable), str(path), str(start), str(stop),
                                              str(length)], text=True).strip()
            record = list(map(int, output.split(",")))
            a = expected[row["case"], window]
            fields = ["records", "branch_count", "taken_count", "load_operands", "store_operands", "load_instructions"]
            assert record[:len(fields)] == [int(a[k]) for k in fields]
            result.append(dict(case=row["case"], window=window, verified_fields=len(fields), instructions=record[0],
                               loads=record[5], loads_reading_a_recent_register_writer=record[6],
                               loads_reading_a_recent_loads_register=record[7],
                               instructions_reading_a_recent_loads_register=record[8]))
    write_csv(prefix + "reader_compatibility.csv", result)
    print(f"Matched native and independent parsers for all {len(result)} windows.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--long-google", action="store_true")
    main(p.parse_args().long_google)
