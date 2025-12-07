#!/usr/bin/env python
from pathlib import Path

import pandas as pd

TABLE_DIRS = [
    Path("newlogs/output/01_tables/shift"),
    Path("newlogs/output/01_tables/tcc"),
    Path("newlogs/output/01_tables/tcc_slip"),
]


def write_headerless(df: pd.DataFrame, dest: Path) -> None:
    numeric = df.iloc[:, 1:]
    numeric.to_csv(dest, sep="\t", index=False, header=False, encoding="utf-8")


def main() -> None:
    for table_dir in TABLE_DIRS:
        if not table_dir.exists():
            continue
        for path in sorted(table_dir.glob("*.tsv")):
            if path.name.endswith("__NOHDR.tsv"):
                continue
            df = pd.read_csv(path, sep="\t", engine="python")
            if df.shape[1] < 2:
                continue
            dest = path.with_name(path.stem + "__NOHDR.tsv")
            write_headerless(df, dest)
            print(f"[INFO] Wrote headerless version: {dest.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
