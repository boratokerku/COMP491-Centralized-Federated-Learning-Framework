"""Quick EDA for a tabular dataset (default: UCI Bank Marketing).

Usage:
    python scripts/eda_tabular.py [path/to/file.csv] [--sep ";"] [--label y]
"""

from __future__ import annotations

import argparse

import pandas as pd

DEFAULT_PATH = "data/tabular/raw/bank-additional/bank-additional-full.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=DEFAULT_PATH)
    p.add_argument("--sep", default=";")
    p.add_argument("--label", default="y")
    p.add_argument("--missing-token", default="unknown")
    args = p.parse_args()

    df = pd.read_csv(args.path, sep=args.sep)

    print(f"# Samples: {len(df)}   Columns: {df.shape[1]}\n")

    print("## Columns and dtypes")
    print(df.dtypes.to_string(), "\n")

    print("## Missing values (NaN / '%s')" % args.missing_token)
    nan = df.isna().sum()
    tok = (df.astype(str) == args.missing_token).sum()
    miss = pd.DataFrame({"nan": nan, args.missing_token: tok})
    print(miss[(miss.T != 0).any()].to_string() or "none", "\n")

    print(f"## Label distribution ({args.label})")
    vc = df[args.label].value_counts()
    print(pd.DataFrame({"count": vc, "ratio": (vc / len(df)).round(4)}).to_string(), "\n")

    print("## Categorical columns")
    for col in df.select_dtypes(include="object").columns:
        if col == args.label:
            continue
        print(f"- {col} ({df[col].nunique()}): {df[col].value_counts().to_dict()}")
    print()

    print("## Numeric summary")
    print(df.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
