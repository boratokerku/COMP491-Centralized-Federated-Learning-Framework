# Client Conformance Manifest Specification

**Status:** v1 proposal; review together with the API and database owners before implementation.  
**Schema:** [`conformance-manifest.schema.json`](conformance-manifest.schema.json)  
**Examples (generated, do not edit by hand):** [`examples/bank_marketing_manifest.json`](examples/bank_marketing_manifest.json), [`examples/sms_spam_manifest.json`](examples/sms_spam_manifest.json)

## Purpose and versioning

The manifest is the versioned contract for one run. It defines the model parameter layout and the exact interpretation of inputs and labels, plus synthetic reference cases. A client is eligible only after its submitted report passes the manifest version pinned in `runs.manifest_version`.

Create a new manifest version whenever the dataset meaning/source, feature names/order/types/units/categories, preprocessing or encoding, label definition, model architecture/parameter layout, protocol-visible reference cases, or expected outputs change. Do not edit a published manifest in place. The gateway serves the immutable JSON artifact and its SHA-256; the report stores the manifest version and checksum for audit.

Both manifests are generated from code by `python backend/scripts/build_manifests.py` (`--check` fails if a checked-in file is out of date; `tests/test_manifest.py` runs the same check). Change the model, tokenizer or dataset config, never the JSON.

- **Bank Marketing** (`bank_mlp` 1.0.0, `65 -> 64 -> 32 -> 2`): built from `backend/configs/datasets/bank_marketing.yaml`, `cfl.tasks.tabular.model` and `cfl.tasks.tabular.manifest`.
- **SMS Spam** (`sms_hash_embed` 1.0.0, hashed-token embedding `16384 x 64 -> 32 -> 2`): built from `cfl.tasks.text.tokenizer`, `cfl.tasks.text.model` and `cfl.tasks.text.manifest`. No vocabulary is built from client data; `preprocessing.tokenizer` fully specifies the hashing tokenizer.

Parameter layout follows PyTorch: `Linear.weight` is `[out, in]`, row-major, all parameters float32, no buffers (no BatchNorm). The raw data remains at the client. All reference records and messages are synthetic and may be shared.

## Manifest fields

| Field | Meaning |
|---|---|
| `manifest_version` | Immutable semantic version for this complete contract. |
| `dataset` | Dataset name, dataset/config version, label column and provenance identifier (not private records). |
| `model` | Architecture ID/version, framework-neutral input/output dimensions, task, exact parameter names/shapes/dtypes, and `fingerprint`: SHA-256 of the canonical JSON (sorted keys, no spaces) of `architecture_id`, `architecture_version` and `parameters`. The gateway and aggregator compare fingerprints. |
| `preprocessing` | Version, implementation contract, encoding and scaling policies. Values are fixed by the manifest; clients must not fit a different scaler or category map on private data. Text manifests add `tokenizer` (normalization, token pattern, n-gram order, FNV-1a hash, bucket count, padding, max length). |
| `input_schema.features` | Ordered list of raw features with type, unit and any fixed numeric or categorical semantics. Array order is authoritative. |
| `input_schema.model_input` | Ordered post-preprocessing feature names and tensor shape/dtype. This must match the model's input dimension. |
| `label` | Positive/negative meanings, raw values, and encoded output range. |
| `reference_tests` | Synthetic raw inputs and exact expected preprocessed arrays (feature vectors, or token ids for text). The client runs them locally and returns only test IDs and outputs. |
| `forward_reference_tests` | Model inputs and expected logits under the reference weights `reference_sin_v1`. Catches transposed/reordered weights and wrong layers that shape checks cannot see. |
| `validation` | Tolerances (`reference_atol` for preprocessing outputs, `forward_atol` for logits) and ordered checks. All checks fail closed; no training task is issued on failure. |

### Reference weights `reference_sin_v1`

Walk `model.parameters` in manifest order and each tensor in row-major order, with one running index `j` across all parameters starting at 0. Element `j` is `float32(0.1 * sin(j + 1))`, computed in float64. Any language can generate these weights, so no weight file has to be shipped for the forward test. Reference implementation: `cfl.core.manifest.reference_weights`.

## Validation order and stable rejection codes

Authentication and approved-client lookup happen before conformance. The gateway then applies these checks in order and records the first failure (all discovered failures may also be returned for debugging):

1. `protocol_version_mismatch` — client protocol is supported by the server.
2. `manifest_version_mismatch` — report and client are checking the exact active run manifest.
3. `model_version_mismatch` — architecture ID/version match the manifest.
4. `model_shape_mismatch` — parameter names, tensor dimensions and dtypes match exactly; no missing or extra parameter is allowed.
5. `feature_schema_mismatch` — feature count, names, types or input/output shapes differ.
6. `feature_order_mismatch` — ordered feature-name array differs, even if the set is the same.
7. `unit_mismatch` — a numeric feature unit differs (e.g. days vs. months).
8. `encoding_mismatch` — categorical vocabulary/order or encoding differs.
9. `preprocessing_version_mismatch` — preprocessing/encoding version differs.
10. `label_semantics_mismatch` — label column or positive/negative meaning differs.
11. `label_range_mismatch` — encoded labels can fall outside the declared values/range.
12. `reference_output_mismatch` — one or more synthetic test outputs differ beyond `reference_atol`.
13. `forward_output_mismatch` — one or more forward reference logits differ beyond `forward_atol`.
14. `invalid_report` — required report fields are missing, malformed, non-finite, or refer to unknown test IDs.

Use exact-match comparison for names, order, versions, units, categories, dimensions and dtypes. Use `validation.reference_atol` and `validation.forward_atol` only for floating-point outputs. Never use the tolerance to excuse metadata differences.

## Client-to-gateway flow

1. Client authenticates and registers. Gateway checks the approved client and protocol/version policy.
2. Client requests the active manifest JSON and synthetic reference inputs. Gateway returns the immutable artifact, `manifest_version`, and artifact SHA-256.
3. Client checks the manifest checksum, runs metadata checks, preprocessing on the supplied synthetic inputs and its forward pass under the reference weights locally, and constructs a report. It does not upload raw local data or local training examples.
4. Client submits the report to the conformance-report endpoint. The report includes client/protocol/manifest versions, declared model and input metadata, pass/fail, test IDs, produced synthetic outputs, and local failure codes.
5. Gateway independently validates the report against the stored manifest and expected outputs. A client-reported `passed=true` is not trusted by itself.
6. Gateway stores the submitted payload and final server decision in `conformance_reports`; it returns accepted or a stable rejection code. The coordinator selects only clients with a latest `PASSED` record for the run's exact `manifest_version`.

The endpoint path and HTTP status mapping must be finalized with the REST API owner (Enes). The database column names intentionally align with `conformance_reports` in [`db-schema.md`](db-schema.md): `manifest_version`, `client_version`, `protocol_version`, `status`, `reason_code`, `report_payload`, and `checked_at`.

## Report payload contract

The JSON Schema below validates the manifest, not the submitted report. The report payload stored in `conformance_reports.report_payload` should follow this shape:

```json
{
  "manifest_version": "1.0.0",
  "manifest_sha256": "<64 lowercase hex characters>",
  "client_id": "client-0",
  "client_version": "independent-client/0.1.0",
  "protocol_version": "1.0",
  "declared_model": {
    "architecture_id": "bank_mlp",
    "architecture_version": "1.0.0",
    "fingerprint": "<64 lowercase hex characters>",
    "parameters": [
      {"name": "fc1.weight", "shape": [64, 65], "dtype": "float32"}
    ]
  },
  "declared_input": {
    "feature_names": ["age", "campaign"],
    "feature_types": ["numeric", "numeric"],
    "feature_units": ["years", "contacts"],
    "preprocessing_version": "1.0"
  },
  "client_passed": true,
  "checks": [{"code": "reference_output_match", "passed": true}],
  "reference_outputs": [
    {"test_id": "bank-marketing-reference-001", "output_features": [0.0, 0.0]}
  ],
  "forward_outputs": [
    {"test_id": "bank-marketing-forward-001", "logits": [-0.0301, -0.0656]}
  ],
  "client_failure_codes": []
}
```

The abbreviated feature list above is illustrative; real reports include all manifest features in order. The gateway's stored `status` and `reason_code` are authoritative. Do not accept client-supplied status as the final decision.

## Implementation acceptance criteria

- The manifest validates against the checked-in JSON Schema and has a stable SHA-256.
- A matching independent client passes every check and is persisted as `PASSED` for the exact version.
- Each listed mismatch has a deterministic failure code and is persisted as `FAILED`; the client cannot receive a task for that run.
- A client that passed an older manifest is rejected for a newer run until it passes the new version.
- Synthetic reference outputs are verified by the gateway; no raw client data is requested or stored.
- The example's 65 ordered model inputs agree with the preprocessing configuration (guaranteed by generation from the config).
- A client that loads `reference_sin_v1` with a transposed or reordered parameter fails `forward_output_mismatch`.
