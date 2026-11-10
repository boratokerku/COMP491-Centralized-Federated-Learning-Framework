#!/usr/bin/env python3
"""Independent-implementation client for the CFL gateway (Bank Marketing, tabular).

Written only from planning/api-spec.md and planning/conformance-manifest.md. It does not import
the `cfl` package; its only dependencies are requests, numpy and pandas. Everything a
participating site does is implemented here by hand: registration, manifest download with
checksum verification, pre-processing from the manifest, the reference-weights forward test, the
conformance report, task polling, safetensors reading and writing, local training of the
65 -> 64 -> 32 -> 2 MLP in NumPy, the multipart update upload and final model verification.

Commands (run with --help for every option):

    report        build the conformance report from a manifest file and print it (no network)
    conformance   register, fetch the manifest, submit the report, print the gateway's decision
    run           conformance, then take part in training rounds until stopped

`--variant` switches on one deliberate incompatibility so a rejection can be shown live; the
gateway must answer with the reason code named in README.md. Without it the client is conformant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CLIENT_VERSION = "independent-client/0.1.0"
PROTOCOL_VERSION = "0.1"
ARCHITECTURE_ID = "bank_mlp"
ARCHITECTURE_VERSION = "1.0.0"
DATASET = "bank_marketing"
CSV_SEPARATOR = ";"  # the partition files written by the federation's partitioner
LABEL_COLUMN = "y"
DROPOUT = 0.2

# Every incompatibility this client can simulate, and the code the gateway must answer with.
VARIANTS = {
    "compatible": None,
    "feature-order": "feature_order_mismatch",
    "unit": "unit_mismatch",
    "label-semantics": "label_semantics_mismatch",
    "label-range": "label_range_mismatch",
    "preprocessing-version": "preprocessing_version_mismatch",
    "model-shape": "model_shape_mismatch",
    "encoding": "encoding_mismatch",
    "weights-layout": "forward_output_mismatch",
}

# Answers to an update after which resending it cannot help.
FINAL_REJECTIONS = frozenset(
    {
        "stale_round",
        "deadline_passed",
        "model_version_mismatch",
        "duplicate_update",
        "not_selected",
        "unknown_round",
        "checksum_mismatch",
        "invalid_update_id",
    }
)


class ClientError(RuntimeError):
    """Something this client cannot recover from (bad checksum, rejected conformance, ...)."""


class GatewayError(ClientError):
    """An error answer in the gateway's error schema (api-spec.md, section 2.5)."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.details = status, code, message, details or {}


class GatewayUnavailable(ClientError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------------------------
# safetensors: 8-byte little-endian header length N, N bytes of JSON header, raw tensor bytes.
# The header maps each tensor name to {"dtype": "F32", "shape": [...], "data_offsets": [a, b]},
# with offsets relative to the start of the raw bytes. All tensors here are float32.
# ---------------------------------------------------------------------------------------------
def read_safetensors(data: bytes) -> dict[str, np.ndarray]:
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len])
    body = memoryview(data)[8 + header_len :]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ClientError(f"tensor {name} has dtype {info['dtype']}, expected F32")
        start, end = info["data_offsets"]
        array = np.frombuffer(body[start:end], dtype="<f4").reshape(info["shape"])
        tensors[name] = array.astype(np.float64)
    return tensors


def write_safetensors(tensors: dict[str, np.ndarray]) -> bytes:
    header, chunks, offset = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # the data section starts on an 8-byte boundary
    return struct.pack("<Q", len(blob)) + blob + b"".join(chunks)


# ---------------------------------------------------------------------------------------------
# Model definition: this client's own idea of the architecture (manifest model.parameters).
# Linear weights are [out, in], row-major.
# ---------------------------------------------------------------------------------------------
def parameter_layout(input_dim: int, variant: str) -> list[dict]:
    hidden1 = 128 if variant == "model-shape" else 64
    sizes = [(input_dim, hidden1), (hidden1, 32), (32, 2)]
    layout = []
    for i, (n_in, n_out) in enumerate(sizes, start=1):
        layout.append({"name": f"fc{i}.weight", "shape": [n_out, n_in], "dtype": "float32"})
        layout.append({"name": f"fc{i}.bias", "shape": [n_out], "dtype": "float32"})
    return layout


def fingerprint(layout: list[dict]) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no spaces) of id, version and parameters."""
    canonical = json.dumps(
        {
            "architecture_id": ARCHITECTURE_ID,
            "architecture_version": ARCHITECTURE_VERSION,
            "parameters": layout,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_hex(canonical.encode())


def reference_sin_v1(layout: list[dict], variant: str) -> dict[str, np.ndarray]:
    """Deterministic test weights: element j (one running index over all parameters, in order)
    is float32(0.1 * sin(j + 1)), computed in float64."""
    weights, offset = {}, 0
    for spec in layout:
        n = math.prod(spec["shape"])
        j = np.arange(offset + 1, offset + n + 1, dtype=np.float64)
        flat = (0.1 * np.sin(j)).astype(np.float32).astype(np.float64)
        shape = spec["shape"]
        if variant == "weights-layout" and len(shape) == 2:
            shape = shape[::-1]  # the deliberate mistake: filled as [in, out]
        weights[spec["name"]] = flat.reshape(shape)
        offset += n
    return weights


def forward(weights: dict[str, np.ndarray], x: np.ndarray, variant: str = "compatible"):
    """Logits of the MLP in inference mode (dropout off)."""
    h = np.asarray(x, dtype=np.float64)
    for i in (1, 2, 3):
        w, b = weights[f"fc{i}.weight"], weights[f"fc{i}.bias"]
        h = h @ w + b if variant == "weights-layout" else h @ w.T + b
        if i < 3:
            h = np.maximum(h, 0.0)
    return h


# ---------------------------------------------------------------------------------------------
# Pre-processing, driven only by the manifest's input_schema.features.
#   numeric      (value - mean) / std; with a `sentinel`, the value is replaced by
#                `sentinel_fill` first and a 0/1 indicator column follows the scaled column
#   categorical  one-hot, columns in the manifest's category order
# Columns follow the feature order; nothing is fitted on local data.
# ---------------------------------------------------------------------------------------------
def schema_features(manifest: dict, variant: str) -> list[dict]:
    features = [dict(f) for f in manifest["input_schema"]["features"]]
    if variant == "feature-order":
        features[0], features[1] = features[1], features[0]
    if variant == "unit":
        for f in features:
            if f["name"] == "pdays":
                f["unit"] = "months"  # this client counts pdays in months
    if variant == "encoding":
        for f in features:
            if f["name"] == "job":
                f["categories"] = list(reversed(f["categories"]))
    return features


def preprocess(df: pd.DataFrame, features: list[dict], variant: str = "compatible") -> np.ndarray:
    blocks = []
    for f in features:
        name = f["name"]
        if name not in df.columns:
            raise ClientError(f"missing column {name!r}")
        if f["type"] == "numeric":
            values = pd.to_numeric(df[name]).to_numpy(dtype=np.float64)
            if np.isnan(values).any():
                raise ClientError(f"{name}: missing values")
            flag = None
            if f.get("sentinel") is not None:
                flag = values == f["sentinel"]
                values = np.where(flag, f["sentinel_fill"], values)
            if f["unit"] == "months" and name == "pdays":
                values = values / 30.0
            blocks.append(((values - f["mean"]) / f["std"])[:, None])
            if flag is not None:
                blocks.append(flag.astype(np.float64)[:, None])
        else:
            values = df[name].astype(str).to_numpy()
            unknown = sorted(set(values) - set(f["categories"]))
            if unknown:
                raise ClientError(f"{name}: values outside the manifest categories: {unknown}")
            blocks.append((values[:, None] == np.array(f["categories"])[None, :]).astype(float))
    return np.hstack(blocks)


def encode_labels(df: pd.DataFrame, manifest: dict, variant: str = "compatible") -> np.ndarray:
    label = manifest["label"]
    values = df[label["column"]].astype(str).to_numpy()
    unexpected = sorted(set(values) - {label["positive_value"], label["negative_value"]})
    if unexpected:
        raise ClientError(f"label values outside the manifest: {unexpected}")
    return (values == label["positive_value"]).astype(np.int64)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEPARATOR, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------------------------
# Conformance report (api-spec.md, section 3.3)
# ---------------------------------------------------------------------------------------------
def build_report(
    manifest: dict, manifest_sha256: str, client_id: str, variant: str = "compatible"
) -> dict:
    features = schema_features(manifest, variant)
    # Model input width comes from this client's own pre-processing of a reference record.
    sample = pd.DataFrame([manifest["reference_tests"][0]["synthetic_raw_input"]])
    input_dim = preprocess(sample, features, variant).shape[1]
    layout = parameter_layout(input_dim, variant)

    reference_outputs = [
        {
            "test_id": t["test_id"],
            "output_features": preprocess(
                pd.DataFrame([t["synthetic_raw_input"]]), features, variant
            )[0].tolist(),
        }
        for t in manifest["reference_tests"]
    ]
    weights = reference_sin_v1(layout, variant)
    forward_outputs = [
        {
            "test_id": t["test_id"],
            "logits": forward(weights, np.array([t["model_input"]]), variant)[0].tolist(),
        }
        for t in manifest["forward_reference_tests"]
    ]

    label = manifest["label"]
    positive, negative, encoded = label["positive_value"], label["negative_value"], [0, 1]
    if variant == "label-semantics":
        positive, negative = negative, positive  # yes=0, no=1
    if variant == "label-range":
        encoded = [1, 2]
    version = manifest["preprocessing"]["version"]
    if variant == "preprocessing-version":
        version = "2.0"

    report = {
        "manifest_version": f"{manifest['dataset']['name']}@{manifest['manifest_version']}",
        "manifest_sha256": manifest_sha256,
        "client_id": client_id,
        "client_version": CLIENT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "declared_model": {
            "architecture_id": ARCHITECTURE_ID,
            "architecture_version": ARCHITECTURE_VERSION,
            "fingerprint": fingerprint(layout),
            "parameters": layout,
        },
        "declared_input": {
            "feature_names": [f["name"] for f in features],
            "feature_types": [f["type"] for f in features],
            "feature_units": [
                f["unit"] if f["type"] == "numeric" else "category" for f in features
            ],
            "preprocessing_version": version,
            "categories": {f["name"]: f["categories"] for f in features if "categories" in f},
            "tokenizer": None,
        },
        "declared_label": {
            "column": label["column"],
            "positive_value": positive,
            "negative_value": negative,
            "encoded_values": encoded,
        },
        "client_passed": False,
        "checks": [],
        "reference_outputs": reference_outputs,
        "forward_outputs": forward_outputs,
        "client_failure_codes": [],
    }
    problems = self_check(manifest, report)
    report["client_passed"] = not problems
    report["checks"] = [{"code": problems[0] if problems else "conformant", "passed": not problems}]
    report["client_failure_codes"] = problems[:1]
    return report


def self_check(manifest: dict, report: dict) -> list[str]:
    """This client's own verdict, for its log. The gateway re-checks everything."""
    found = []
    expected_names = [f["name"] for f in manifest["input_schema"]["features"]]
    if report["declared_input"]["feature_names"] != expected_names:
        found.append("feature_order_mismatch")
    if report["declared_model"]["fingerprint"] != manifest["model"]["fingerprint"]:
        found.append("model_shape_mismatch")
    atol = manifest["validation"]
    for key, tests, want_key, out_key, tol, code in (
        ("reference_outputs", manifest["reference_tests"], "expected_model_input",
         "output_features", atol["reference_atol"], "reference_output_mismatch"),
        ("forward_outputs", manifest["forward_reference_tests"], "expected_logits",
         "logits", atol["forward_atol"], "forward_output_mismatch"),
    ):  # fmt: skip
        got = {o["test_id"]: o[out_key] for o in report[key]}
        for t in tests:
            want, have = t[want_key], got.get(t["test_id"], [])
            if (
                len(want) != len(have)
                or max(abs(a - b) for a, b in zip(want, have, strict=True)) > tol
            ):
                found.append(code)
                break
    return found


# ---------------------------------------------------------------------------------------------
# Metrics and local training (NumPy)
# ---------------------------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos = int(y.sum())
    neg = len(y) - pos
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(score) != 0)[0], len(y) - 1]  # end of each tied block
    precision, recall = tp[last] / (tp[last] + fp[last]), tp[last] / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(weights: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    logits = forward(weights, x)
    prob = softmax(logits)
    pred = logits.argmax(axis=1)
    tp = float(((pred == 1) & (y == 1)).sum())
    precision = tp / max(float((pred == 1).sum()), 1.0)
    recall = tp / max(float((y == 1).sum()), 1.0)
    metrics = {
        "loss": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, None)).mean()),
        "accuracy": float((pred == y).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    if 0 < y.sum() < len(y):
        metrics["roc_auc"] = roc_auc(y, prob[:, 1])
        metrics["pr_auc"] = average_precision(y, prob[:, 1])
    return metrics


def train(
    weights: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    config: dict,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Local training with exactly the round config: Adam, class-weighted cross-entropy,
    dropout after the first hidden layer, L2 weight decay, and the FedProx term
    (mu / 2) * ||w - w_global||^2 when proximal_mu > 0. Returns (new weights, mean batch loss)."""
    global_weights = {k: v.copy() for k, v in weights.items()}
    w = {k: v.copy() for k, v in weights.items()}
    mu = float(config.get("proximal_mu") or 0.0)
    weight_decay = float(config.get("weight_decay") or 0.0)
    lr = float(config["learning_rate"])
    class_w = np.array(config.get("class_weights") or [1.0, 1.0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v2 = {k: np.zeros_like(v) for k, v in w.items()}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    total, batches = 0.0, 0

    for _ in range(int(config["local_epochs"])):
        order = rng.permutation(len(y))
        for start in range(0, len(order), int(config["batch_size"])):
            idx = order[start : start + int(config["batch_size"])]
            xb, yb = x[idx], y[idx]
            # forward (training mode)
            z1 = xb @ w["fc1.weight"].T + w["fc1.bias"]
            a1 = np.maximum(z1, 0.0)
            mask = (rng.random(a1.shape) >= DROPOUT) / (1.0 - DROPOUT)
            h1 = a1 * mask
            z2 = h1 @ w["fc2.weight"].T + w["fc2.bias"]
            a2 = np.maximum(z2, 0.0)
            logits = a2 @ w["fc3.weight"].T + w["fc3.bias"]
            prob = softmax(logits)
            sample_w = class_w[yb]
            norm = sample_w.sum()
            loss = float(
                -(sample_w * np.log(np.clip(prob[np.arange(len(yb)), yb], 1e-12, None))).sum()
                / norm
            )
            # backward
            dz3 = prob.copy()
            dz3[np.arange(len(yb)), yb] -= 1.0
            dz3 *= (sample_w / norm)[:, None]
            grads = {"fc3.weight": dz3.T @ a2, "fc3.bias": dz3.sum(axis=0)}
            dz2 = (dz3 @ w["fc3.weight"]) * (z2 > 0)
            grads |= {"fc2.weight": dz2.T @ h1, "fc2.bias": dz2.sum(axis=0)}
            dz1 = ((dz2 @ w["fc2.weight"]) * mask) * (z1 > 0)
            grads |= {"fc1.weight": dz1.T @ xb, "fc1.bias": dz1.sum(axis=0)}
            if mu > 0:
                loss += 0.5 * mu * sum(float(((w[k] - global_weights[k]) ** 2).sum()) for k in w)
            # Adam (PyTorch semantics: weight decay is added to the gradient)
            step += 1
            for k in w:
                g = grads[k]
                if mu > 0:
                    g = g + mu * (w[k] - global_weights[k])
                if weight_decay:
                    g = g + weight_decay * w[k]
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v2[k] = beta2 * v2[k] + (1 - beta2) * g * g
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v2[k] / (1 - beta2**step)
                w[k] = w[k] - lr * m_hat / (np.sqrt(v_hat) + eps)
            total += loss
            batches += 1
    return w, total / max(batches, 1)


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
class Gateway:
    """The client side of api-spec.md, section 3. Transient failures (connection errors,
    timeouts, 429, 502/503/504) are retried with exponential backoff and full jitter."""

    RETRY_STATUSES = frozenset({429, 502, 503, 504})

    def __init__(self, base_url: str, token: str, verify: str | bool = True, attempts: int = 6):
        if not base_url.startswith("https://"):
            raise ClientError("the gateway is HTTPS only; the server URL must start with https://")
        self.base = base_url.rstrip("/")
        self.attempts = attempts
        self.http = requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {token}", "User-Agent": CLIENT_VERSION})
        self.http.verify = verify

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                response = self.http.request(method, self.base + path, timeout=60, **kwargs)
                if response.status_code not in self.RETRY_STATUSES:
                    return response
                last = ClientError(f"{response.status_code} from {path}")
            except requests.exceptions.SSLError as exc:
                raise ClientError(
                    f"TLS certificate not trusted ({exc}); pass --ca-bundle for a private CA"
                ) from exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
            time.sleep(random.uniform(0, min(30.0, 0.5 * 2**attempt)))
        raise GatewayUnavailable(f"{method} {path} failed after {self.attempts} attempts: {last}")

    @staticmethod
    def _error(response: requests.Response) -> GatewayError:
        try:
            body = response.json()["error"]
            return GatewayError(
                response.status_code, body["code"], body["message"], body.get("details")
            )
        except (ValueError, KeyError, TypeError):
            return GatewayError(response.status_code, "unexpected_response", response.text[:200])

    def _json(self, method: str, path: str, ok=(200,), **kwargs):
        response = self._request(method, path, **kwargs)
        if response.status_code not in ok:
            raise self._error(response)
        return response

    def register(self, client_id: str) -> dict:
        body = {
            "client_id": client_id,
            "client_version": CLIENT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
        }
        return self._json("POST", "/register", json=body).json()

    def manifest_for(self, dataset: str) -> tuple[bytes, str, str] | None:
        """(manifest bytes, sha256, "<dataset>@<version>") of the active run for `dataset`."""
        listing = self._json("GET", "/conformance").json()
        mine = [m for m in listing if m["dataset"] == dataset]
        if not mine:
            return None
        identity = mine[0]["manifest_version"]
        response = self._json("GET", f"/conformance/{identity}")
        digest = sha256_hex(response.content)
        published = response.headers.get("X-Manifest-Checksum-SHA256")
        if digest != published or digest != mine[0]["manifest_sha256"]:
            raise ClientError(f"manifest {identity} does not match its published checksum")
        return response.content, digest, identity

    def submit_report(self, report: dict) -> dict:
        response = self._request("POST", "/conformance/report", json=report)
        if response.status_code not in (200, 422) or "status" not in response.json():
            raise self._error(response)  # 401/403, or a 422 invalid_request
        return response.json()

    def poll_task(self) -> dict | None:
        response = self._json("GET", "/task", ok=(200, 204))
        return None if response.status_code == 204 else response.json()

    def download_model(self, run_id: str, version: int) -> bytes:
        response = self._json("GET", f"/model/{run_id}/{version}")
        if sha256_hex(response.content) != response.headers.get("X-Checksum-SHA256"):
            raise ClientError(f"model v{version} does not match X-Checksum-SHA256")
        return response.content

    def upload_update(self, metadata: dict, weights: bytes) -> dict:
        response = self._request(
            "POST",
            "/update",
            data={"metadata": json.dumps(metadata)},
            files={"weights": ("update.safetensors", weights, "application/octet-stream")},
        )
        body = response.json()
        if "error" in body:  # invalid_metadata, identity_mismatch, 401/403 auth errors
            raise self._error(response)
        return body

    def final_file(
        self, run_id: str, fmt: str
    ) -> tuple[bytes, requests.structures.CaseInsensitiveDict] | None:
        """None while the run is still going; GatewayError no_final_model when it failed."""
        response = self._request("GET", f"/model/{run_id}/final", params={"format": fmt})
        if response.status_code == 409:
            return None
        if response.status_code != 200:
            raise self._error(response)
        if sha256_hex(response.content) != response.headers.get("X-Checksum-SHA256"):
            raise ClientError(f"final {fmt} does not match X-Checksum-SHA256")
        return response.content, response.headers


# ---------------------------------------------------------------------------------------------
# Participation
# ---------------------------------------------------------------------------------------------
def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class Participant:
    gateway: Gateway
    client_id: str
    variant: str
    dataset: str
    partition_dir: Path
    state_dir: Path
    poll_interval: float
    seed: int

    # -- state kept across restarts: rounds done, runs trained in, finals verified ------------
    @property
    def _state_file(self) -> Path:
        return self.state_dir / "state.json"

    def _load(self) -> dict:
        try:
            return json.loads(self._state_file.read_text())
        except FileNotFoundError:
            return {"rounds_done": [], "runs": [], "finals_verified": []}

    def _save(self, state: dict) -> None:
        atomic_write(self._state_file, json.dumps(state, indent=2).encode())

    def conformance(self) -> tuple[dict, dict]:
        """Register, wait for the dataset's manifest, submit the report. Returns (manifest,
        decision); the decision is the gateway's response, PASSED or FAILED."""
        registration = self.gateway.register(self.client_id)
        log(f"registered as {self.client_id}: {registration['status']}")
        while True:
            found = self.gateway.manifest_for(self.dataset)
            if found is not None:
                break
            log(f"no running run for {self.dataset} yet; waiting")
            time.sleep(self.poll_interval)
        data, digest, identity = found
        manifest = json.loads(data)
        log(f"manifest {identity} (sha256 {digest[:12]}) verified")
        report = build_report(manifest, digest, self.client_id, self.variant)
        own = "conformant" if report["client_passed"] else report["client_failure_codes"]
        log(f"own check: {own}")
        decision = self.gateway.submit_report(report)
        return manifest, decision

    def run(self, manifest: dict, exit_after_final: bool) -> int:
        features = schema_features(manifest, self.variant)
        train_df = load_split(self.partition_dir / "train.csv")
        val_df = load_split(self.partition_dir / "val.csv")
        x_train, y_train = preprocess(train_df, features), encode_labels(train_df, manifest)
        x_val, y_val = preprocess(val_df, features), encode_labels(val_df, manifest)
        log(
            f"local data: {len(y_train)} train rows, {len(y_val)} val rows, "
            f"{x_train.shape[1]} inputs"
        )
        state = self._load()
        verified_at_start = set(state["finals_verified"])

        while True:
            if self._resend_pending(state):
                time.sleep(self.poll_interval)
                continue
            task = self.gateway.poll_task()
            if task is not None and [task["run_id"], task["round_id"]] not in state["rounds_done"]:
                self._round(task, state, x_train, y_train, x_val, y_val)
                continue
            if self._fetch_finals(state):
                latest = state["runs"][-1] if state["runs"] else None
                if (
                    exit_after_final
                    and latest in state["finals_verified"]
                    and latest not in verified_at_start
                ):
                    return 0
                continue
            time.sleep(self.poll_interval)

    # -- one round -----------------------------------------------------------------------------
    def _round(self, task: dict, state: dict, x_train, y_train, x_val, y_val) -> None:
        run_id, round_id = task["run_id"], task["round_id"]
        log(f"round {round_id} of run {run_id[:8]}: training on v{task['model_version']}")
        global_weights = read_safetensors(
            self.gateway.download_model(run_id, task["model_version"])
        )
        before = evaluate(global_weights, x_val, y_val)
        started = time.perf_counter()
        weights, train_loss = train(
            global_weights, x_train, y_train, task["config"], seed=self.seed * 100_003 + round_id
        )
        train_s = time.perf_counter() - started
        after = evaluate(weights, x_val, y_val)

        blob = write_safetensors(weights)
        metrics = {"train_loss": train_loss, "train_s": train_s}
        metrics |= {f"val_{k}_before": v for k, v in before.items()}
        metrics |= {f"val_{k}_after": v for k, v in after.items()}
        metadata = {
            "run_id": run_id,
            "round_id": round_id,
            "client_id": self.client_id,
            "update_id": str(uuid.uuid4()),  # UUIDv4; a retry reuses it
            "model_version": task["model_version"],
            "num_samples": int(len(y_train)),
            "checksum_sha256": sha256_hex(blob),
            "metrics": {k: float(v) for k, v in metrics.items() if math.isfinite(v)},
        }
        # Written before the upload: after a crash the same update (same id) is sent again.
        atomic_write(self.state_dir / "pending" / f"{metadata['update_id']}.safetensors", blob)
        atomic_write(
            self.state_dir / "pending" / f"{metadata['update_id']}.json",
            json.dumps(metadata).encode(),
        )
        if run_id in state["runs"]:
            state["runs"].remove(run_id)
        state["runs"].append(run_id)
        self._save(state)
        self._upload(metadata, blob, state)

    def _upload(self, metadata: dict, blob: bytes, state: dict) -> bool:
        """True once the gateway has decided (accepted or finally rejected)."""
        try:
            ack = self.gateway.upload_update(metadata, blob)
        except GatewayError as exc:  # e.g. identity_mismatch: resending cannot help
            log(f"update refused: {exc}")
            ack = None
        if ack is not None and ack["accepted"]:
            note = " (retry acknowledged)" if ack.get("reason") == "duplicate_idempotent" else ""
            log(f"round {metadata['round_id']}: update accepted{note}")
        elif ack is not None and ack.get("reason") not in FINAL_REJECTIONS:
            log(f"round {metadata['round_id']}: not decided ({ack.get('reason')}); will resend")
            return False
        elif ack is not None:
            log(f"round {metadata['round_id']}: update rejected ({ack['reason']})")
        for suffix in (".json", ".safetensors"):
            (self.state_dir / "pending" / f"{metadata['update_id']}{suffix}").unlink(
                missing_ok=True
            )
        state["rounds_done"].append([metadata["run_id"], metadata["round_id"]])
        self._save(state)
        return True

    def _resend_pending(self, state: dict) -> bool:
        for meta_file in sorted((self.state_dir / "pending").glob("*.json")):
            metadata = json.loads(meta_file.read_text())
            blob = meta_file.with_suffix(".safetensors").read_bytes()
            self._upload(metadata, blob, state)
            return True
        return False

    # -- final model -----------------------------------------------------------------------------
    def _fetch_finals(self, state: dict) -> bool:
        for run_id in state["runs"]:
            if run_id in state["finals_verified"]:
                continue
            try:
                got = self.gateway.final_file(run_id, "onnx")
            except GatewayError as exc:
                if exc.code == "no_final_model":
                    log(f"run {run_id[:8]} ended without a final model")
                    state["finals_verified"].append(run_id)
                    self._save(state)
                    return True
                raise
            if got is None:  # still running
                continue
            onnx, headers = got
            files = {"model.onnx": onnx}
            for fmt, name in (("artifact", "artifact.json"), ("safetensors", "model.safetensors")):
                files[name] = self.gateway.final_file(run_id, fmt)[0]  # type: ignore[index]
            self._verify_final(files, headers["X-Final-Model-SHA256"])
            for name, data in files.items():
                atomic_write(self.state_dir / "final" / run_id / name, data)
            state["finals_verified"].append(run_id)
            self._save(state)
            saved = self.state_dir / "final" / run_id
            log(f"run {run_id[:8]} finished: final model verified, saved under {saved}")
            return True
        return False

    @staticmethod
    def _verify_final(files: dict[str, bytes], published_onnx_sha256: str) -> None:
        """Checksums of every file, then the reference predictions in NumPy from the weights
        (no ONNX runtime needed: the safetensors file holds the same parameters)."""
        if sha256_hex(files["model.onnx"]) != published_onnx_sha256:
            raise ClientError("model.onnx does not match the checksum published by the server")
        descriptor = json.loads(files["artifact.json"])
        for name, info in descriptor["files"].items():
            if name in files and sha256_hex(files[name]) != info["sha256"]:
                raise ClientError(f"{name} does not match artifact.json")
        weights = read_safetensors(files["model.safetensors"])
        for ref in descriptor["reference_predictions"]:
            logits = forward(weights, np.array([ref["model_input"]]))[0]
            if np.abs(logits - np.array(ref["logits"])).max() > descriptor["verification"]["atol"]:
                raise ClientError(f"final weights do not reproduce {ref['test_id']}")


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["report", "conformance", "run"])
    p.add_argument(
        "--server-url", default=os.environ.get("CFL_SERVER_URL"), help="https://fl.<ip>.sslip.io"
    )
    p.add_argument("--client-id", default=os.environ.get("CFL_CLIENT_ID"))
    p.add_argument(
        "--token-file",
        type=Path,
        default=os.environ.get("CFL_TOKEN_FILE"),
        help="file with the bearer token (or set CFL_TOKEN)",
    )
    p.add_argument(
        "--ca-bundle",
        default=os.environ.get("CFL_CA_BUNDLE"),
        help="CA certificate for a server with a private CA",
    )
    p.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="compatible",
        help="simulate one incompatibility; see README.md",
    )
    p.add_argument("--dataset", default=DATASET)
    p.add_argument(
        "--manifest", type=Path, help="report: manifest JSON file to build the report for (offline)"
    )
    p.add_argument(
        "--partition-dir",
        type=Path,
        default=Path("partition"),
        help="folder with train.csv and val.csv",
    )
    p.add_argument("--state-dir", type=Path, default=Path("state"))
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exit-after-final",
        action="store_true",
        help="run: stop once the final model of a run was verified",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "report":
        if args.manifest is None:
            raise SystemExit("report needs --manifest FILE")
        data = args.manifest.read_bytes()
        manifest = json.loads(data)
        report = build_report(
            manifest, sha256_hex(data), args.client_id or "independent-test", args.variant
        )
        print(json.dumps(report, indent=2))
        return 0

    missing = [n for n in ("server_url", "client_id") if not getattr(args, n)]
    if missing:
        raise SystemExit(f"missing: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    token = os.environ.get("CFL_TOKEN") or (
        args.token_file.read_text().strip() if args.token_file else ""
    )
    if not token:
        raise SystemExit("no token: use --token-file FILE or set CFL_TOKEN")

    try:
        participant = Participant(
            Gateway(args.server_url, token, verify=args.ca_bundle or True),
            args.client_id, args.variant, args.dataset,
            args.partition_dir, args.state_dir, args.poll_interval, args.seed,
        )  # fmt: skip
        manifest, decision = participant.conformance()
        if decision["status"] != "PASSED":
            expected = VARIANTS[args.variant]
            log(f"REJECTED by the gateway: {decision['reason_code']}: {decision['message']}")
            log(f"details: {json.dumps(decision.get('details', {}))[:400]}")
            if expected:
                verdict = "as expected" if expected == decision["reason_code"] else "UNEXPECTED"
                log(f"expected for variant {args.variant!r}: {expected} -> {verdict}")
            return 2
        log(f"conformance PASSED: {decision['message']}")
        if args.command == "conformance":
            return 0
        return participant.run(manifest, args.exit_after_final)
    except KeyboardInterrupt:
        return 130
    except ClientError as exc:
        log(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
