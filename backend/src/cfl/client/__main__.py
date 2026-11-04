"""Managed-code client entrypoint (the cfl-client image): `python -m cfl.client <command>`.

    check        verify this site's setup: token, partition files, task, state directory,
                 and that the gateway is reachable over HTTPS with a matching protocol version
    conformance  run the manifest's reference and forward tests against this site's TaskSpec
                 and show what the gateway would decide, without submitting anything;
                 --manifest FILE works offline (e.g. planning/examples/*_manifest.json)
    run          take part in training runs (the client runtime; see planning/interfaces.md)

Settings come from command-line options, CFL_* environment variables or a YAML config file
(--config), in that order of precedence (cfl.client.settings; `--help` lists them all):

    --server-url     CFL_SERVER_URL     gateway base URL, e.g. https://fl.203-0-113-7.sslip.io
    --client-id      CFL_CLIENT_ID      this site's approved client id
    --task           CFL_TASK           tabular | text
    --task-spec      CFL_TASK_SPEC      optional module:Class of the site's TaskSpec
    --token-file     CFL_TOKEN_FILE     file with the bearer token (default /run/secrets/cfl_token)
    --partition-dir  CFL_PARTITION_DIR  this site's train.csv, val.csv, test.csv (default /data)
    --state-dir      CFL_STATE_DIR      writable folder for pending updates (default /state)
    --ca-bundle      CFL_CA_BUNDLE      CA certificate to trust, for a server with its own CA
    --poll-interval  CFL_POLL_INTERVAL  seconds between task polls when idle (default 3)
    --seed           CFL_SEED           seed for local training (default 0)

The container image sets its defaults through the environment; outside a container, a config
file is the easiest (deploy/client/client.yaml.example). The token itself is never an option.

`run --exit-after-final` stops once the final model of a run this site trained in has been
downloaded and verified (useful for demos and tests); without it the client keeps serving.
"""

from __future__ import annotations

import argparse
import importlib
import signal
import sys
from pathlib import Path

import httpx

from cfl.client.settings import OPTIONS, ClientSettings, SettingsError, load_settings
from cfl.protocol import PROTOCOL_VERSION
from cfl.tasks.registry import get_task

SPLITS = ("train", "val", "test")


def check(s: ClientSettings) -> int:
    s.require("client_id", "task", "server_url")
    problems: list[str] = []

    def ok(message: str) -> None:
        print(f"  ok    {message}")

    def fail(message: str) -> None:
        print(f"  FAIL  {message}")
        problems.append(message)

    print(f"{s.client_version} as {s.client_id}")

    token_file = s.token_file
    try:
        token = token_file.read_text().strip()
        ok(f"token file {token_file}") if token else fail(f"token file {token_file} is empty")
    except OSError as exc:
        fail(f"token file {token_file}: {exc.strerror}")

    try:
        task = get_task(s.task)
        ok(f"task {task.name} ({task.dataset})")
    except ValueError as exc:
        fail(str(exc))
        task = None

    spec_path = s.task_spec
    if spec_path:
        module, _, cls = spec_path.partition(":")
        try:
            getattr(importlib.import_module(module), cls)
            ok(f"TaskSpec {spec_path}")
        except (ImportError, AttributeError) as exc:
            fail(f"TaskSpec {spec_path}: {exc}")

    partition = s.partition_dir
    for split in SPLITS:
        path = partition / f"{split}.csv"
        if not path.is_file():
            fail(f"missing {path}")
        elif task is not None:
            try:
                inputs, labels = task.load_split(path)
                ok(f"{path}: {len(labels)} rows, input {tuple(inputs.shape[1:])}")
            except Exception as exc:  # report any parsing problem instead of a traceback
                fail(f"{path}: {exc}")

    state = s.state_dir
    probe = state / ".write-test"
    try:
        state.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink()
        ok(f"state directory {state} is writable")
    except OSError as exc:
        fail(f"state directory {state}: {exc.strerror}")

    server = s.server_url  # https:// is enforced by ClientSettings
    if server:
        try:
            response = httpx.get(f"{server}/health", timeout=10, verify=s.verify)
            response.raise_for_status()
            remote = response.json().get("protocol_version")
            if remote == PROTOCOL_VERSION:
                ok(f"gateway {server} reachable, protocol {remote}")
            else:
                fail(f"gateway speaks protocol {remote}, this client {PROTOCOL_VERSION}")
        except httpx.HTTPError as exc:
            fail(f"gateway {server}: {exc}")

    print("ready" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def _load_task_spec(s: ClientSettings):
    """task_spec = module:Class (constructed without arguments), else the reference spec."""
    from cfl.client.reference import ReferenceTaskSpec

    spec_path = s.task_spec
    if not spec_path:
        return ReferenceTaskSpec(s.task)
    module, _, cls = spec_path.partition(":")
    return getattr(importlib.import_module(module), cls)()


def conformance(s: ClientSettings, manifest_file: Path | None) -> int:
    """Self-test for TaskSpec authors: every reference and forward test with its largest
    difference, then the checker's verdict on the full report."""
    import json

    import torch

    from cfl.conformance.checker import check_report
    from cfl.conformance.report import build_report
    from cfl.core.contract import ModelContract
    from cfl.core.manifest import manifest_id, reference_weights
    from cfl.core.serialization import sha256

    s.require("task")
    task_name = s.task
    spec = _load_task_spec(s)
    if manifest_file is not None:
        data = manifest_file.read_bytes()
        source = str(manifest_file)
    else:
        from cfl.client.http import HttpTransport

        s.require("server_url", "client_id")
        token = s.token_file.read_text().strip()
        transport = HttpTransport(s.server_url, token, verify=s.verify)
        from cfl.protocol.messages import RegisterRequest

        # GET /conformance needs a registered client; registering again is harmless.
        transport.register(
            RegisterRequest(
                client_id=s.client_id,
                client_version=s.client_version,
            )
        )
        dataset = get_task(task_name).dataset
        mine = [m for m in transport.list_manifests() if m.dataset == dataset]
        if not mine:
            print(f"no running run uses {dataset}; pass --manifest FILE to test offline")
            return 1
        data, _ = transport.get_manifest(mine[0].manifest_version)
        source = f"{s.server_url} ({mine[0].manifest_version})"
    manifest = json.loads(data)
    print(f"manifest {manifest_id(manifest)} from {source}")
    print(f"TaskSpec {type(spec).__module__}.{type(spec).__name__}")

    failures = 0
    for test in manifest["reference_tests"]:
        try:
            got = list(spec.preprocess(test["synthetic_raw_input"]))
        except Exception as exc:  # show the author's exception, not a traceback
            print(f"  FAIL  preprocess {test['test_id']}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        want = test["expected_model_input"]
        if len(got) != len(want):
            print(f"  FAIL  preprocess {test['test_id']}: {len(got)} values, expected {len(want)}")
            failures += 1
            continue
        diffs = [abs(a - b) for a, b in zip(want, got, strict=True)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        ok = diffs[worst] <= manifest["validation"]["reference_atol"]
        failures += not ok
        names = manifest["input_schema"]["model_input"]["feature_names"]
        where = (
            "" if ok else f" (worst at {names[worst]}: expected {want[worst]}, got {got[worst]})"
        )
        print(
            f"  {'ok  ' if ok else 'FAIL'}  preprocess {test['test_id']}: "
            f"max diff {diffs[worst]:.2e}{where}"
        )

    model = manifest["model"]
    contract = ModelContract.from_model(
        spec.build_model(), model["architecture_id"], model["architecture_version"]
    )
    dtype = (
        torch.int64
        if manifest["input_schema"]["model_input"]["dtype"] == "int64"
        else (torch.float32)
    )
    expected_params = [(p["name"], list(p["shape"]), p["dtype"]) for p in model["parameters"]]
    own_params = [(p.name, list(p.shape), p.dtype) for p in contract.params]
    if own_params == expected_params:
        print(f"  ok    model parameters: {len(own_params)} match the manifest")
    else:
        failures += 1
        wrong = next(
            (
                i
                for i, (a, b) in enumerate(zip(expected_params, own_params, strict=False))
                if a != b
            ),
            min(len(expected_params), len(own_params)),
        )
        print(
            f"  FAIL  model parameters: first difference at #{wrong}: expected "
            f"{expected_params[wrong] if wrong < len(expected_params) else None}, got "
            f"{own_params[wrong] if wrong < len(own_params) else None}"
        )
    weights = reference_weights(contract)
    for test in manifest["forward_reference_tests"]:
        net = spec.build_model()
        try:
            net.load_state_dict(weights)
        except RuntimeError as exc:
            print(f"  FAIL  forward {test['test_id']}: weights do not fit the model: {exc}")
            failures += 1
            break
        net.eval()
        with torch.no_grad():
            got = net(torch.tensor([test["model_input"]], dtype=dtype))[0].tolist()
        diff = max(abs(a - b) for a, b in zip(test["expected_logits"], got, strict=False))
        ok = (
            len(got) == len(test["expected_logits"])
            and diff <= manifest["validation"]["forward_atol"]
        )
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  forward {test['test_id']}: max diff {diff:.2e}")

    report = build_report(
        manifest=manifest,
        manifest_sha256=sha256(data),
        client_id=s.client_id or "self-test",
        client_version=s.client_version,
        preprocess=spec.preprocess,
        build_model=spec.build_model,
        contract=contract,
    )
    result = check_report(manifest, sha256(data), report)
    if result.passed:
        print("conformant: the gateway would accept this report")
        return 0
    print(f"NOT conformant: {result.reason_code}: {result.message}")
    if result.details:
        print(f"  details: {json.dumps(result.details, default=str)[:400]}")
    return 1


def run(s: ClientSettings, exit_after_final: bool = False) -> int:
    from cfl.client.http import GatewayHTTPError, HttpTransport
    from cfl.client.runtime import ClientRuntime, ConformanceFailed

    # As PID 1 in a container, SIGTERM is ignored unless handled. Stopping is always safe:
    # a pending update stays on disk and is resent after the restart.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    s.require("server_url", "client_id", "task")
    try:
        token = s.token_file.read_text().strip()
    except OSError as exc:
        print(f"error: token file {s.token_file}: {exc.strerror}", file=sys.stderr)
        return 2
    print(f"cfl-client settings: {s.describe()}", flush=True)
    transport = HttpTransport(s.server_url, token, verify=s.verify, user_agent=s.client_version)
    runtime = ClientRuntime(
        transport=transport,
        task=_load_task_spec(s),
        client_id=s.client_id,
        client_version=s.client_version,
        dataset=get_task(s.task).dataset,
        partition_dir=s.partition_dir,
        state_dir=s.state_dir,
        poll_interval_s=s.poll_interval,
        seed=s.seed,
    )
    try:
        runtime.run(exit_after_final=exit_after_final)
    except ConformanceFailed as exc:
        print(f"error: conformance check failed: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return 1
    except GatewayHTTPError as exc:
        print(f"error: the gateway refused this client: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("stopped; start again to continue")
    finally:
        transport.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cfl-client",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["check", "conformance", "run"])
    parser.add_argument("--config", type=Path, default=None, help="YAML settings file")
    for name, option in OPTIONS.items():
        parser.add_argument(
            option.flag,
            dest=name,
            type=option.type,
            default=None,
            help=f"{option.help} [{option.env}]",
        )
    parser.add_argument("--exit-after-final", action="store_true", help="with run: see above")
    parser.add_argument("--manifest", type=Path, default=None, help="with conformance: see above")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config, {name: getattr(args, name) for name in OPTIONS})
        if args.command == "check":
            return check(settings)
        if args.command == "conformance":
            return conformance(settings, args.manifest)
        return run(settings, args.exit_after_final)
    except SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
