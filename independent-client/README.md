# Independent-implementation client

A second client for the CFL gateway, written **only from the specification**
([`planning/api-spec.md`](../planning/api-spec.md) and
[`planning/conformance-manifest.md`](../planning/conformance-manifest.md)). It is one Python file,
[`independent_client.py`](independent_client.py), that uses `requests`, `numpy` and `pandas` and
**does not import the `cfl` package**: registration, manifest download with checksum check,
pre-processing from the manifest, the `reference_sin_v1` forward test, the conformance report,
polling, safetensors reading and writing, NumPy training of the 65 → 64 → 32 → 2 MLP, the
multipart update and verification of the final model are all implemented here.

Its purpose is to show that the protocol does not depend on our Python code: whatever this client
needs and the specification does not say is a gap in the specification and goes into it.
It also holds the deliberately incompatible variants used in the live demo.

## Setup

```bash
cd independent-client
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need three things from the federation:

| What | How |
|---|---|
| Server URL | `https://fl.<server-ip>.sslip.io` (local test: `https://fl.localhost:<CFL_HTTPS_PORT>`) |
| Client id and token | the administrator runs `cfl-admin add-client <client_id> --token-file token.txt` and sends you both |
| Your data | a folder with `train.csv` and `val.csv`: raw Bank Marketing rows, `;` separated, with the `y` label, as written by `backend/scripts/partition_tabular.py` (for example `data/tabular/partitions/bank_marketing/client_0/`) |

For a server with its own CA (local Caddy), pass its root certificate with `--ca-bundle`
(`deploy/scripts/caddy_ca.sh` exports it). Let's Encrypt certificates need nothing.

## Run

```bash
python independent_client.py run \
    --server-url https://fl.203-0-113-7.sslip.io --client-id bank-b \
    --token-file token.txt --partition-dir ../data/tabular/partitions/bank_marketing/client_1 \
    --state-dir state --exit-after-final
```

`CFL_SERVER_URL`, `CFL_CLIENT_ID`, `CFL_TOKEN_FILE` and `CFL_CA_BUNDLE` can replace the options;
`CFL_TOKEN` holds the token itself. The three commands:

| Command | What it does |
|---|---|
| `report --manifest FILE` | builds the conformance report from a manifest file and prints it; no network, no token (for example `../planning/examples/bank_marketing_manifest.json`) |
| `conformance` | registers, downloads and verifies the manifest, submits the report and prints the gateway's decision; exit code 0 for `PASSED`, 2 for `FAILED` |
| `run` | `conformance`, then polls for tasks, downloads the global model, trains locally, uploads the update and verifies the final model; `--exit-after-final` stops after the first run it took part in has finished |

The state directory keeps the pending update on disk before it is uploaded. After a crash or a
lost response the same `update_id` is sent again and the gateway counts it once.

## Incompatible variants

`--variant NAME` makes the client wrong in exactly one way. The gateway must reject its
conformance report with the code in the table, and no task is ever issued to it. Without
`--variant` (or with `--variant compatible`) the client is conformant and takes part in rounds.

| `--variant` | What it does wrong | Expected rejection |
|---|---|---|
| `feature-order` | swaps the first two features (`campaign`, `age`) | `feature_order_mismatch` |
| `unit` | counts `pdays` in months instead of days | `unit_mismatch` |
| `label-semantics` | declares `yes` as 0 and `no` as 1 | `label_semantics_mismatch` |
| `label-range` | declares the encoded labels as `[1, 2]` instead of `[0, 1]` | `label_range_mismatch` |
| `preprocessing-version` | declares preprocessing version `2.0` | `preprocessing_version_mismatch` |
| `model-shape` | uses 128 instead of 64 units in the first hidden layer | `model_shape_mismatch` |
| `encoding` | lists the `job` categories in reverse order | `encoding_mismatch` |
| `weights-layout` | reads weights as `[in, out]` instead of `[out, in]` | `forward_output_mismatch` |

```bash
python independent_client.py conformance --variant unit --server-url ... --client-id ... --token-file token.txt
# REJECTED by the gateway: unit_mismatch: feature units differ from the manifest
# expected for variant 'unit': unit_mismatch -> as expected
```

`backend/tests/test_independent_client.py` checks every variant against the gateway's checker
without a server. A rejected client stays `FAILED` until it submits a passing report for the
same manifest.

## What the client implements, in one place

* **Pre-processing** (from `input_schema.features`): numeric `(value - mean) / std`; with a
  `sentinel`, the value is first replaced by `sentinel_fill` and a 0/1 indicator column follows the
  scaled column; categorical one-hot in the manifest's category order. Nothing is fitted locally.
* **Model**: `x → fc1 → ReLU → dropout(0.2) → fc2 → ReLU → fc3`, logits `[no, yes]`, weights `[out, in]`.
* **Local training**: Adam, class-weighted cross-entropy, L2 weight decay, and the FedProx term
  `(mu/2)·‖w − w_global‖²` when `proximal_mu > 0`, exactly the round config. A check against the
  PyTorch reference with dropout off agrees to about 1e-6 on the weights.
* **Final model**: SHA-256 of `model.onnx` against `X-Final-Model-SHA256`, every file against
  `artifact.json`, and the reference predictions reproduced in NumPy from `model.safetensors`.
