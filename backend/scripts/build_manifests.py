"""Generate the conformance manifests from code and validate them against the JSON Schema.

Output: planning/examples/bank_marketing_manifest.json, planning/examples/sms_spam_manifest.json
Never edit those files by hand; change the model, tokenizer or dataset config and rerun.

Usage (from the repository root):
    python backend/scripts/build_manifests.py           # write the manifests
    python backend/scripts/build_manifests.py --check   # fail if a manifest is out of date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jsonschema

from cfl.core.manifest import same_manifest
from cfl.tasks.tabular.manifest import build_manifest as build_tabular
from cfl.tasks.tabular.preprocessing import load_dataset_config
from cfl.tasks.text.manifest import build_manifest as build_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = REPO_ROOT / "planning" / "conformance-manifest.schema.json"
EXAMPLES = REPO_ROOT / "planning" / "examples"
BANK_CONFIG = REPO_ROOT / "backend" / "configs" / "datasets" / "bank_marketing.yaml"


def to_json(value, indent: int = 0) -> str:
    """Like json.dumps(indent=2), but keeps lists of scalars on one line."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{json.dumps(k)}: {to_json(v, indent + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and any(isinstance(v, dict | list) for v in value):
        items = [f"{inner}{to_json(v, indent + 1)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value, ensure_ascii=False)


def manifests() -> dict[Path, dict]:
    return {
        EXAMPLES / "bank_marketing_manifest.json": build_tabular(load_dataset_config(BANK_CONFIG)),
        EXAMPLES / "sms_spam_manifest.json": build_text(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check files are up to date")
    args = parser.parse_args()

    schema = json.loads(SCHEMA.read_text())
    stale = []
    for path, manifest in manifests().items():
        jsonschema.validate(manifest, schema)
        text = to_json(manifest) + "\n"
        rel = path.relative_to(REPO_ROOT)
        if args.check:
            if not path.exists() or not same_manifest(json.loads(path.read_text()), manifest):
                stale.append(rel)
            continue
        path.write_text(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"{rel}  model={manifest['model']['fingerprint'][:12]}  sha256={digest}")

    if stale:
        print("out of date (run build_manifests.py):", *stale, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
