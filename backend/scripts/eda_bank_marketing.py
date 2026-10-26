"""Exploratory data analysis for the UCI Bank Marketing dataset.

Prints a plain-text summary (shape, types, "unknown" values, label balance, categories,
numeric ranges). Run download_bank_marketing.py first.

Usage (from the repository root):
    python backend/scripts/eda_bank_marketing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "data" / "tabular" / "raw" / "bank-additional-full.csv"
LABEL = "y"


def main() -> int:
    if not CSV.exists():
        print(f"{CSV} not found; run download_bank_marketing.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV, sep=";")

    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Empty cells: {int(df.isna().sum().sum())}\n")

    print("Label balance:")
    counts = df[LABEL].value_counts()
    for value, n in counts.items():
        print(f"  {value}: {n} ({n / len(df):.1%})")

    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != LABEL]
    numeric = [c for c in df.columns if c not in categorical and c != LABEL]

    print("\nCategorical columns (values; share of 'unknown'):")
    for col in categorical:
        unknown = (df[col] == "unknown").mean()
        values = sorted(df[col].unique())
        print(f"  {col}: {len(values)} values, unknown={unknown:.1%}")
        print(f"    {values}")

    print("\nNumeric columns (min / max / mean / std):")
    for col in numeric:
        s = df[col]
        print(f"  {col}: {s.min()} / {s.max()} / {s.mean():.4f} / {s.std():.4f}")

    print(f"\npdays == 999 (not previously contacted): {(df['pdays'] == 999).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
