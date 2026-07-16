"""Destructive-path tests for queue recovery and parity contracts."""

import copy
import http.server
import json
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from . import PARITY_SCHEMA, QUEUE_SCHEMA
from .artifacts import (ArtifactError, atomic_write_bytes, atomic_write_json,
                        load_json)
from .host import process_identity, stable_process_identity
from .parity import run_parity
from .runner import run_queue


FAKE_ADAPTER = r'''import base64, hashlib, json, os, sys, time
envelope = json.load(sys.stdin)
request = envelope.get("request", {})
marker = request.get("sleepOnceMarker")
if marker:
    try:
        with open(marker, "x"):
            pass
    except FileExistsError:
        pass
    else:
        time.sleep(float(request.get("sleepOnceSeconds", 3)))
time.sleep(float(request.get("sleepSeconds", 0)))
if request.get("exitCode"):
    raise SystemExit(int(request["exitCode"]))
lane = os.environ.get("PICCHIO_TEST_LANE", envelope.get("lane", "queue"))
first = 30 if lane == "product" else 10
usage_prompt = int(os.environ.get("PICCHIO_TEST_USAGE_PROMPT", "10"))
received_prompt = int(os.environ.get("PICCHIO_TEST_RECEIVED_PROMPT", "10"))
sampling = json.loads(os.environ.get(
    "PICCHIO_TEST_SAMPLING", json.dumps(request.get(
        "sampling", {"temperature": 0}))))
wire_body = json.dumps({
    "request": request, "maxOutputTokens": envelope.get("maxOutputTokens")
}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
wire_body += os.environ.get("PICCHIO_TEST_WIRE_SUFFIX", "").encode("utf-8")
wire_sha = hashlib.sha256(wire_body).hexdigest()
out = {
    "ok": not request.get("runtimeFail", False),
    "text": request.get("text", "PASS " + envelope.get("caseId", "?")),
    "usage": {"promptTokens": usage_prompt, "completionTokens": 5},
    "timings": {
        "routeAdmissionMs": 1, "runtimeLoadMs": 2,
        "serverTtftMs": 4, "serverPrefillTokensPerSecond": 100,
        "serverDecodeTokensPerSecond": 20,
        "adapterFirstChunkMs": first, "serverTotalMs": 12
    },
    "received": {
        "inputSha256": envelope.get("inputSha256"),
        "promptTokens": received_prompt,
        "modelSha256": os.environ.get("PICCHIO_TEST_MODEL_SHA", "a" * 64),
        "context": request.get("context", 4096),
        "runtimeBuild": request.get("runtimeBuild", "fake-1"),
        "kv": request.get("kv", "f16"),
        "parallel": int(os.environ.get("PICCHIO_TEST_PARALLEL", "1")),
        "sampling": sampling,
        "maxOutputTokens": envelope.get("maxOutputTokens"),
        "launchArgs": request.get("launchArgs", ["fake-server", "-c", "4096"]),
        "runtimePid": request.get("runtimePid"),
        "runtimeStartTime": request.get("runtimeStartTime"),
        "launchSequence": int(os.environ.get(
            "PICCHIO_TEST_LAUNCH_SEQUENCE", request.get("launchSequence", 1))),
        "projector": request.get("projector", False),
        "cacheN": int(os.environ.get("PICCHIO_TEST_CACHE_N", "0")),
        "toolRounds": int(os.environ.get("PICCHIO_TEST_TOOL_ROUNDS", "0"))
    },
    "wireEvidence": {
        "kind": "actual_engine_request_body",
        "provenance": os.environ.get(
            "PICCHIO_TEST_WIRE_PROVENANCE",
            "product_engine_boundary_capture" if lane == "product"
            else "direct_engine_send"),
        "bodyBase64": base64.b64encode(wire_body).decode("ascii"),
        "sha256": os.environ.get("PICCHIO_TEST_WIRE_SHA", wire_sha)
    },
}
if request.get("agentTrace"):
    trace_rounds = []
    for index, (purpose, prompt, completion) in enumerate([
        ("planning", 6000, 4), ("tool", 7000, 3), ("final", 8000, 5)
    ]):
        round_body = json.dumps({
            "round": index, "purpose": purpose, "promptTokens": prompt
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        trace_rounds.append({
            "index": index,
            "purpose": purpose,
            "visible": purpose == "final",
            "usage": {
                "promptTokens": prompt,
                "completionTokens": completion
            },
            "timings": {
                "serverTtftMs": 1000 + index * 100,
                "serverPrefillTokensPerSecond": 500 - index * 10,
                "serverDecodeTokensPerSecond": 20 + index
            },
            "wireEvidence": {
                "kind": "actual_engine_request_body",
                "provenance": "application_engine_boundary_capture",
                "bodyBase64": base64.b64encode(round_body).decode("ascii"),
                "sha256": hashlib.sha256(round_body).hexdigest()
            }
        })
    out["agentTrace"] = {
        "schema": "picchio.agent-trace.v1",
        "components": [
            {"id": "user", "kind": "user_input", "characters": 45,
             "bytes": 45, "tokens": 12, "countSource": "measured",
             "tokenSource": "tokenized"},
            {"id": "system", "kind": "system_instructions",
             "label": "UNTRUSTED CUSTOM LABEL",
             "characters": 61404, "bytes": 61404,
             "countSource": "measured"},
            {"id": "document", "kind": "document_context",
             "characters": 46801, "bytes": 46801,
             "countSource": "measured", "includedIn": "system"},
            {"id": "history", "kind": "conversation_history",
             "characters": 0, "bytes": 0, "items": 0,
             "countSource": "measured"}
        ],
        "rounds": trace_rounds,
        "context": {
            "capacityTokens": 32768,
            "compactAtTokens": 16384,
            "reportedCompactTriggered": True
        },
        "firstVisibleTokenMs": 93200,
        "visibleDecodeTokensPerSecond": 25.23
    }
json.dump(out, sys.stdout)
'''


def _case(case_id, identity, **request):
    body = {
        "runtimePid": identity["pid"],
        "runtimeStartTime": identity["startTime"],
        "context": 4096,
        "runtimeBuild": "fake-1",
        "kv": "f16",
        "sampling": {"temperature": 0},
        "launchArgs": ["fake-server", "-c", "4096"],
        "launchSequence": 1,
        "projector": False,
    }
    body.update(request)
    return {"id": case_id, "category": request.get("category", "general"),
            "request": body, "maxOutputTokens": 32,
            "rubric": {"contains": ["PASS"]}, "telemetry": False}


def _queue(adapter, identity, cases):
    return {"schema": QUEUE_SCHEMA, "adapter": adapter,
            "runtime": {"pid": identity["pid"]}, "cases": cases}


def _hooks():
    return {"version": "selftest", "protocol": "selftest",
            "telemetry_start": lambda disabled=False: {"off": "selftest"}}


def _quiet(_message):
    pass


class _HttpFail(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"synthetic"}')

    def log_message(self, _format, *_args):
        pass


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _receipts(root):
    rows = []
    for base, _directories, files in os.walk(os.path.join(root, "cases")):
        if "result.json" in files:
            rows.append(load_json(os.path.join(base, "result.json")))
    return rows


def run_selftests(entry_argv):
    checks = []
    failures = []

    def check(name, condition):
        checks.append(name)
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="picchio-selftest-") as root:
        adapter_path = os.path.join(root, "fake_adapter.py")
        with open(adapter_path, "w", encoding="utf-8") as handle:
            handle.write(FAKE_ADAPTER)
        adapter = {"type": "command",
                   "command": [sys.executable, adapter_path],
                   "timeoutSeconds": 10}
        identity = process_identity(os.getpid())
        if not identity:
            return 0, 1, ["process_identity"]

        # Normal completion covers four task categories in one serial queue.
        normal_cases = [_case("reason", identity, category="reasoning"),
                        _case("code", identity, category="code"),
                        _case("extract", identity, category="extraction"),
                        _case("format", identity, category="format")]
        normal = run_queue(_queue(adapter, identity, normal_cases),
                           os.path.join(root, "normal"), _hooks(), _quiet)
        check("queue_normal", normal["exitCode"] == 0 and
              normal["completedCases"] == 4 and len(normal["rollups"]) >= 1)

        # A generic multi-round trace keeps current context separate from total
        # prompt work, persists every engine-boundary body, and renders only
        # neutral component names supplied by Picchio's fixed vocabulary.
        trace_root = os.path.join(root, "agent-trace")
        traced = run_queue(_queue(adapter, identity, [
            _case("agent-trace", identity, agentTrace=True)]),
            trace_root, _hooks(), _quiet)
        trace_receipts = _receipts(trace_root)
        trace = trace_receipts[0].get("agentTrace") \
            if trace_receipts else {}
        trace_summary = trace.get("summary") or {}
        markdown_path = os.path.join(trace_root,
                                     trace.get("markdownPath", "missing"))
        try:
            with open(markdown_path, encoding="utf-8") as handle:
                trace_markdown = handle.read()
        except OSError:
            trace_markdown = ""
        trace_wires = [
            (round_row.get("wireEvidence") or {}).get("valid")
            for round_row in trace.get("rounds", [])
        ]
        check("agent_trace_accounting", traced["exitCode"] == 0 and
              trace.get("status") == "complete" and
              trace_summary.get("currentPromptTokens") == 8000 and
              trace_summary.get("peakPromptTokens") == 8000 and
              trace_summary.get("totalPromptTokensProcessed") == 21000 and
              trace_summary.get("compactDecision") == "false_positive" and
              traced["rollups"][0].get("promptTokens") == 21000 and
              traced["rollups"][0].get("currentPromptTokensMedian") == 8000 and
              trace_wires == [True, True, True])
        check("agent_trace_neutral_markdown",
              "System instructions" in trace_markdown and
              "Document context" in trace_markdown and
              "FALSE POSITIVE" in trace_markdown and
              "UNTRUSTED CUSTOM LABEL" not in trace_markdown)
        checkout_root = os.path.dirname(os.path.abspath(entry_argv[-1]))
        trace_example_path = os.path.join(checkout_root, "examples",
                                          "agent-trace.md")
        if os.path.isfile(trace_example_path):
            try:
                with open(trace_example_path, encoding="utf-8") as handle:
                    trace_example = handle.read()
            except OSError:
                trace_example = ""
            check("agent_trace_example_fresh", trace_markdown == trace_example)

        quality_manifest = _queue(adapter, identity,
                                  [_case("quality", identity, text="WRONG")])
        quality = run_queue(quality_manifest, os.path.join(root, "quality"),
                            _hooks(), _quiet)
        check("quality_failure", quality["exitCode"] == 4 and
              quality["quality"]["fail"] == 1)

        # Real HTTP 500, not a command-process stand-in.
        server = socketserver.TCPServer(("127.0.0.1", 0), _HttpFail)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        http_manifest = {"schema": QUEUE_SCHEMA,
                         "adapter": {"type": "http", "url":
                             "http://127.0.0.1:{}/fail".format(
                                 server.server_address[1])},
                         "cases": [_case("http", identity)]}
        http_result = run_queue(http_manifest, os.path.join(root, "http"),
                                _hooks(), _quiet)
        server.shutdown()
        server.server_close()
        check("http_failure", http_result["exitCode"] == 3 and
              http_result["runtime"]["fail"] == 1)

        # SIGKILL the runner after case one committed and case two started.
        marker = os.path.join(root, "kill.marker")
        kill_manifest = _queue(adapter, identity, [
            _case("done", identity),
            _case("resume", identity, sleepOnceMarker=marker,
                  sleepOnceSeconds=3)])
        manifest_path = os.path.join(root, "kill.json")
        artifact = os.path.join(root, "kill-run")
        atomic_write_json(manifest_path, kill_manifest)
        command = list(entry_argv) + ["run", manifest_path,
                                     "--artifact", artifact]
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

        def second_started():
            checkpoint = os.path.join(artifact, "checkpoint.json")
            try:
                completed = load_json(checkpoint).get("completed", {})
            except ArtifactError:
                return False
            return os.path.exists(marker) and len(completed) == 1

        started = _wait_until(second_started)
        if started:
            os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        resumed = subprocess.run(command, capture_output=True, text=True,
                                 timeout=15)
        try:
            resumed_json = json.loads(resumed.stdout)
        except ValueError:
            resumed_json = {}
        case_root = os.path.join(artifact, "cases")
        attempt_counts = [len([name for name in os.listdir(
            os.path.join(case_root, item)) if name.startswith("attempt-")])
            for item in os.listdir(case_root)] if os.path.isdir(case_root) \
            else []
        check("runner_kill_resume", started and resumed.returncode == 0 and
              resumed_json.get("completedCases") == 2 and
              sorted(attempt_counts) == [1, 2])

        # Runtime exits by itself mid-case. Picchio observes, never signals.
        server_exit = _queue(adapter, identity, [
            _case("server-exit", identity, sleepSeconds=1.3)])
        server_exit["runtime"] = {"launchCommand": [
            sys.executable, "-c", "import time; time.sleep(1)"]}
        exited = run_queue(server_exit, os.path.join(root, "server-exit"),
                           _hooks(), _quiet)
        check("server_early_exit", exited["exitCode"] == 3 and
              exited.get("stopReason") == "server_exited")

        blocker = os.path.join(root, "blocker")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a directory")
        try:
            run_queue(_queue(adapter, identity,
                             [_case("disk", identity)]),
                      os.path.join(blocker, "run"), _hooks(), _quiet)
        except ArtifactError:
            disk_failed = True
        else:
            disk_failed = False
        check("disk_write_failure", disk_failed)

        safety_manifest = _queue(adapter, identity,
                                 [_case("safety", identity),
                                  _case("never", identity)])
        safety_manifest["safety"] = {"maxRssBytes": 0}
        safety = run_queue(safety_manifest, os.path.join(root, "safety"),
                           _hooks(), _quiet)
        check("safety_stop", safety["exitCode"] == 6 and
              safety["completedCases"] == 0 and safety["safetyAbort"] and
              safety["safetyAbort"].get("stage") == "before_case")

        fingerprint = {"modelSha256": "a" * 64,
                       "runtimeBuild": "fake-1", "context": 4096,
                       "kv": "f16", "parallel": 1,
                       "sampling": {"temperature": 0},
                       "maxOutputTokens": 32, "projector": False}
        lane = {"adapter": adapter, "runtime": {"pid": identity["pid"]},
                "fingerprint": copy.deepcopy(fingerprint)}
        product = copy.deepcopy(lane)
        product["adapter"]["env"] = {"PICCHIO_TEST_LANE": "product"}
        parity_manifest = {
            "schema": PARITY_SCHEMA, "fingerprint": fingerprint,
            "lanes": {"baseline": lane, "product": product},
            "calibrationCases": [_case("cal", identity)],
            "cases": [_case("normal-1", identity),
                      _case("normal-2", identity)]}
        parity_root = os.path.join(root, "parity")
        parity = run_parity(parity_manifest, parity_root,
                            _hooks(), _quiet)
        check("parity_causal", parity["exitCode"] == 0 and
              parity["comparability"] == "causal" and
              parity["causalAttribution"] == "stream_bridge" and
              parity["completedLaneCases"] == 6)
        normal_order = [row["lane"] for row in sorted(
            (row for row in _receipts(parity_root)
             if row.get("phase") == "normal"),
            key=lambda row: row["executionIndex"])]
        check("parity_interleaved_order",
              normal_order == ["baseline", "product", "product",
                               "baseline"])
        first_receipt = _receipts(parity_root)[0]
        wire_path = os.path.join(
            parity_root, first_receipt["wireEvidence"]["bodyPath"])
        atomic_write_bytes(wire_path, b"tampered")
        tampered = run_parity(parity_manifest, parity_root, _hooks(), _quiet)
        check("wire_artifact_tamper_gate", tampered["exitCode"] == 5 and
              "wire_body_artifact_digest_mismatch" in {
                  row["code"] for row in tampered["comparabilityIssues"]})

        abba_manifest = copy.deepcopy(parity_manifest)
        abba_manifest["executionOrder"] = "abba"
        abba_manifest["calibrationCases"] = []
        abba_manifest["cases"] = [abba_manifest["cases"][0]]
        abba_root = os.path.join(root, "abba")
        abba = run_parity(abba_manifest, abba_root, _hooks(), _quiet)
        abba_rows = sorted(_receipts(abba_root),
                           key=lambda row: row["executionIndex"])
        check("parity_abba_order", abba["exitCode"] == 0 and
              abba["completedLaneCases"] == 4 and
              [row["lane"] for row in abba_rows] ==
              ["baseline", "product", "product", "baseline"] and
              [row["replicate"] for row in abba_rows] == [1, 1, 2, 2])

        directional_manifest = copy.deepcopy(parity_manifest)
        directional_manifest["lanes"]["product"]["fingerprint"][
            "context"] = 8192
        directional = run_parity(
            directional_manifest, os.path.join(root, "directional"),
            _hooks(), _quiet)
        check("parity_directional", directional["verdict"] == "DIRECTIONAL"
              and directional["exitCode"] == 5 and
              directional["causalAttribution"] is None)

        def gate_run(name, env):
            gate_manifest = copy.deepcopy(parity_manifest)
            gate_manifest["calibrationCases"] = []
            gate_manifest["cases"] = [gate_manifest["cases"][0]]
            gate_manifest["lanes"]["product"]["adapter"].setdefault(
                "env", {}).update(env)
            return run_parity(gate_manifest, os.path.join(root, name),
                              _hooks(), _quiet)

        provenance = gate_run("wire-provenance", {
            "PICCHIO_TEST_WIRE_PROVENANCE":
                "product_reconstructed_from_contract"})
        check("wire_provenance_gate", provenance["exitCode"] == 5 and
              "wire_body_provenance_mismatch" in {
                  row["code"] for row in provenance["comparabilityIssues"]})

        wire = gate_run("wire-body", {"PICCHIO_TEST_WIRE_SUFFIX": "changed"})
        check("wire_body_gate", wire["exitCode"] == 5 and
              "lane_wire_body_difference" in {
                  row["code"] for row in wire["comparabilityIssues"]})

        prompt = gate_run("prompt-usage", {
            "PICCHIO_TEST_RECEIVED_PROMPT": "11"})
        check("prompt_usage_gate", prompt["exitCode"] == 5 and
              "prompt_tokens_usage_mismatch" in {
                  row["code"] for row in prompt["comparabilityIssues"]})

        actual = gate_run("actual-fingerprint", {
            "PICCHIO_TEST_MODEL_SHA": "b" * 64,
            "PICCHIO_TEST_PARALLEL": "2"})
        actual_codes = {row["code"] for row in
                        actual["comparabilityIssues"]}
        check("actual_fingerprint_gate", actual["exitCode"] == 5 and
              "received_field_mismatch" in actual_codes and
              "lane_model_sha256_difference" in actual_codes and
              "lane_parallel_difference" in actual_codes)

        runtime_gate = gate_run("runtime-identity", {
            "PICCHIO_TEST_LAUNCH_SEQUENCE": "2"})
        check("cross_lane_runtime_gate", runtime_gate["exitCode"] == 5 and
              "lane_launch_sequence_difference" in {
                  row["code"] for row in
                  runtime_gate["comparabilityIssues"]})

        cache = gate_run("cache-state", {"PICCHIO_TEST_CACHE_N": "19"})
        check("cache_state_gate", cache["exitCode"] == 5 and
              "lane_cache_state_difference" in {
                  row["code"] for row in cache["comparabilityIssues"]})

        sampling = gate_run("sampling", {
            "PICCHIO_TEST_SAMPLING": '{"temperature":0.4}'})
        check("requested_sampling_gate", sampling["exitCode"] == 5 and
              any(row.get("code") == "received_field_mismatch" and
                  row.get("field") == "sampling"
                  for row in sampling["comparabilityIssues"]))

        reuse_manifest = copy.deepcopy(parity_manifest)
        for case in reuse_manifest["cases"]:
            case["request"]["runtimeStartTime"] = "reused-pid-wrong-start"
        reuse = run_parity(reuse_manifest, os.path.join(root, "pid-reuse"),
                           _hooks(), _quiet)
        codes = [row.get("code") for row in reuse["comparabilityIssues"]]
        check("pid_reuse", reuse["comparability"] == "directional" and
              "runtime_start_time_mismatch" in codes)

        child = subprocess.Popen([
            sys.executable, "-c", "import time; time.sleep(.2)",
            "/tmp/llama-server"])
        child_identity = process_identity(child.pid)
        child.wait(timeout=2)
        check("argv_identity", child_identity is not None and
              child_identity.get("executablePath") and
              "llama-server" not in child_identity["executablePath"])

        caffeinate = "/usr/bin/caffeinate"
        if os.path.exists(caffeinate):
            wrapped = subprocess.Popen([
                caffeinate, "-i", sys.executable, "-c",
                "import time; time.sleep(.2)", "/tmp/llama-server"])
            wrapped_identity = stable_process_identity(wrapped.pid)
            wrapped.wait(timeout=2)
            wrapper_ok = wrapped_identity is not None and \
                wrapped_identity.get("executablePath") and \
                "llama-server" not in wrapped_identity["executablePath"] and \
                wrapped_identity.get("parentPid") == os.getpid()
        else:
            wrapper_ok = True
        check("caffeinate_wrapper_identity", wrapper_ok)

        source_root = os.path.join(checkout_root, "src", "picchio_core")
        if os.path.isdir(source_root):
            archive_path = os.path.join(checkout_root, "public",
                                        "picchio.pyz")
            archive_ok = os.path.isfile(archive_path)
            if archive_ok:
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        with open(os.path.join(checkout_root, "picchio.py"),
                                  "rb") as handle:
                            archive_ok = archive.read("picchio.py") == \
                                handle.read()
                        for name in sorted(os.listdir(source_root)):
                            if not name.endswith(".py"):
                                continue
                            with open(os.path.join(source_root, name),
                                      "rb") as handle:
                                archive_ok = archive_ok and archive.read(
                                    "picchio_core/" + name) == handle.read()
                except (OSError, KeyError, zipfile.BadZipFile):
                    archive_ok = False
            check("published_zipapp_fresh", archive_ok)

    return len(checks) - len(failures), len(checks), failures
