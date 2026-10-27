# Client Conformance Manifest Specification

**Status:** v1 proposal; review together with the API and database owners before implementation.  
**Schema:** [`conformance-manifest.schema.json`](conformance-manifest.schema.json)  
**Bank Marketing example:** [`examples/bank_marketing_manifest.json`](examples/bank_marketing_manifest.json)

## Purpose and versioning

The manifest is the versioned contract for one run. It defines the model parameter layout and the exact interpretation of inputs and labels, plus synthetic reference cases. A client is eligible only after its submitted report passes the manifest version pinned in `runs.manifest_version`.

Create a new manifest version whenever the dataset meaning/source, feature names/order/types/units/categories, preprocessing or encoding, label definition, model architecture/parameter layout, protocol-visible reference cases, or expected outputs change. Do not edit a published manifest in place. The gateway serves the immutable JSON artifact and its SHA-256; the report stores the manifest version and checksum for audit.

The example is filled from the current Bank Marketing preprocessing configuration. It proposes a small `65 -> 64 -> 2` MLP only to make the parameter contract concrete; the team must freeze that architecture and its exact parameter names in code before declaring this example executable. The raw data remains at the client. The example's reference record is synthetic and may be shared.

## Manifest fields

| Field | Meaning |
|---|---|
| `manifest_version` | Immutable semantic version for this complete contract. |
| `dataset` | Dataset name, dataset/config version, label column and provenance identifier (not private records). |
| `model` | Architecture ID/version, framework-neutral input/output dimensions, task and exact parameter names/shapes/dtypes. |
| `preprocessing` | Version, implementation contract, encoding and scaling policies. Values are fixed by the manifest; clients must not fit a different scaler or category map on private data. |
| `input_schema.features` | Ordered list of raw features with type, unit and any fixed numeric or categorical semantics. Array order is authoritative. |
| `input_schema.model_input` | Ordered post-preprocessing feature names and tensor shape/dtype. This must match the model's input dimension. |
| `label` | Positive/negative meanings, raw values, and encoded output range. |
| `reference_tests` | Synthetic raw inputs and exact expected preprocessed arrays. The client runs them locally and returns only test IDs and outputs. |
| `validation` | Tolerance and ordered checks. All checks fail closed; no training task is issued on failure. |

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
12. `reference_output_mismatch` — one or more synthetic test outputs differ beyond the declared absolute tolerance.
13. `invalid_report` — required report fields are missing, malformed, non-finite, or refer to unknown test IDs.

Use exact-match comparison for names, order, versions, units, categories, dimensions and dtypes. Use `validation.reference_atol` only for floating-point reference outputs. Never use the tolerance to excuse metadata differences.

## Client-to-gateway flow

1. Client authenticates and registers. Gateway checks the approved client and protocol/version policy.
2. Client requests the active manifest JSON and synthetic reference inputs. Gateway returns the immutable artifact, `manifest_version`, and artifact SHA-256.
3. Client checks the manifest checksum, runs metadata checks and preprocessing on the supplied synthetic inputs locally, and constructs a report. It does not upload raw local data or local training examples.
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
    "parameters": [
      {"name": "network.0.weight", "shape": [64, 65], "dtype": "float32"}
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
- The example's 65 ordered model inputs agree with the preprocessing configuration. The MLP dimensions/parameter names remain a team decision until model implementation is fixed.
