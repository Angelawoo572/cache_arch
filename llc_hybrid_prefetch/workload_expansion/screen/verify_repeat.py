"""Compare independent native executions at the same registered instruction windows."""
import argparse
import csv
from pathlib import Path
from run import STUDY, POLICIES, select


def numeric_table(file):
    with file.open() as f:
        rows = csv.DictReader(f)
        if file.name == "stats.csv":
            return {r["key"]:int(r["value"]) for r in rows}
        return [tuple(int(r[k]) for k in rows.fieldnames) for r in rows]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat",type=Path,default=STUDY/"raw/remote-repeat")
    args=parser.parse_args()
    count=0
    for label,folder in (("A","screen"),("B","replication"),("C","long_google")):
        for r in select(None):
            if label=="C" and r["collection"]!="gtrace_v2":
                continue
            for policy in POLICIES:
                suffix=Path("runs")/r["case"]/policy
                original=STUDY/"raw"/folder/suffix
                repeat=args.repeat/label/suffix
                for name in ("stats.csv","cores.csv"):
                    if numeric_table(original/name)!=numeric_table(repeat/name):
                        raise ValueError((label,r["case"],policy,name))
                count+=1
    assert count==156
    print(f"All {count} independent fixed-work native configurations reproduce core retirement and every recorded system statistic.")


if __name__=="__main__":
    main()
