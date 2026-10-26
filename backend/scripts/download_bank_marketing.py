"""Download the UCI Bank Marketing dataset and verify it with SHA-256.

Only `bank-additional-full.csv` is extracted, into data/tabular/raw/ (gitignored).
The data is never committed to the repository.

Usage (from the repository root):
    python backend/scripts/download_bank_marketing.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import httpx

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
INNER_ZIP = "bank-additional.zip"
CSV_MEMBER = "bank-additional/bank-additional-full.csv"
CSV_NAME = "bank-additional-full.csv"
CSV_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "tabular" / "raw"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(out_dir: Path) -> Path:
    target = out_dir / CSV_NAME
    if target.exists() and sha256(target.read_bytes()) == CSV_SHA256:
        print(f"Already present and verified: {target}")
        return target

    print(f"Downloading {URL}")
    response = httpx.get(URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    archive = response.content
    if sha256(archive) != ZIP_SHA256:
        raise RuntimeError("Checksum mismatch for the downloaded archive (source changed?)")

    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        inner_bytes = outer.read(INNER_ZIP)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        csv_bytes = inner.read(CSV_MEMBER)
    if sha256(csv_bytes) != CSV_SHA256:
        raise RuntimeError(f"Checksum mismatch for {CSV_NAME}")

    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(csv_bytes)
    print(f"Saved and verified: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()
    download(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
