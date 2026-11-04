"""Download the UCI SMS Spam Collection and verify it with SHA-256.

Only the `SMSSpamCollection` file is extracted, into data/text/raw/ (gitignored).
The data is never committed to the repository.

Usage (from the repository root):
    python backend/scripts/download_sms_spam.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import httpx

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
MEMBER = "SMSSpamCollection"
MEMBER_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "text" / "raw"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(out_dir: Path) -> Path:
    target = out_dir / MEMBER
    if target.exists() and sha256(target.read_bytes()) == MEMBER_SHA256:
        print(f"Already present and verified: {target}")
        return target

    print(f"Downloading {URL}")
    response = httpx.get(URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    archive = response.content
    if sha256(archive) != ZIP_SHA256:
        raise RuntimeError("Checksum mismatch for the downloaded archive (source changed?)")

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        data = zf.read(MEMBER)
    if sha256(data) != MEMBER_SHA256:
        raise RuntimeError(f"Checksum mismatch for {MEMBER}")

    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
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
