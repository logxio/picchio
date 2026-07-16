"""Serial long-queue execution with resumable, independently reducible proof."""

import json
import os
import statistics
import time

from . import ADAPTER_REQUEST_SCHEMA, CASE_RESULT_SCHEMA, QUEUE_SCHEMA
from .agent_trace import persist_agent_trace
from .adapters import AdapterError, call_adapter
from .artifacts import (ArtifactError, ArtifactStore, atomic_write_bytes,
                        atomic_write_json, canonical_bytes, sha256_json,
                        utc_now)
from .host import (evaluate_safety, memory_delta, memory_snapshot,
                   process_identity)
from .evidence import capture_wire_evidence
from .quality import evaluate_quality
from .runtime import RuntimeError, ensure_runtime, runtime_alive
from .summary import queue_summary


class ManifestError(Exception):
    pass


def validate_cases(cases, label="cases"):
    if not isinstance(cases, list) or not cases:
        raise ManifestError("{} must be a non-empty list".format(label))
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ManifestError("{}[{}] must be an object".format(
                label, index))
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ManifestError("{}[{}] needs a string id".format(
                label, index))
        if case_id in seen:
            raise ManifestError("duplicate case id: {}".format(case_id))
        seen.add(case_id)
        if not isinstance(case.get("request"), dict):
            raise ManifestError("case {} request must be an object".format(
                case_id))
    return cases


def validate_queue_manifest(manifest):
    if manifest.get("schema") != QUEUE_SCHEMA:
        raise ManifestError("queue manifest schema must be {}".format(
            QUEUE_SCHEMA))
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict):
        raise ManifestError("queue manifest requires adapter")
    validate_cases(manifest.get("cases"))
    return manifest


def _gpu_summary(samples, request_started):
    rows = [row for row in samples
            if row.get("t") is not None and row["t"] >= request_started]
    util = [row.get("dev") for row in rows
            if row.get("dev") is not None]
    watts = [row.get("gpu_w") for row in rows
             if row.get("gpu_w") is not None]
    memory = [row.get("mem") for row in rows
              if row.get("mem") is not None]
    return {
        "schema": "picchio.gpu-window.v1",
        "attribution": "whole_gpu",
        "sampleCount": len(rows),
        "utilizationPercent": {
            "median": statistics.median(util) if util else None,
            "peak": max(util) if util else None,
        },
        "powerWatts": {
            "median": statistics.median(watts) if watts else None,
            "peak": max(watts) if watts else None,
        },
        "memoryBytes": {"peak": max(memory) if memory else None},
        # Flat aliases keep hourly rollup independent from nested styling.
        "work_med": statistics.median(util) if util else None,
        "work_w": statistics.median(watts) if watts else None,
        "warnings": ["GPU metrics are whole-GPU, not per-process."],
    }


def _stop_gpu(sampler, request_started, attempt_dir):
    if isinstance(sampler, dict):
        return {"summary": dict(sampler, attribution="whole_gpu"),
                "samplesPath": None}
    samples = []
    if sampler is not None:
        try:
            sampler.stop()
        finally:
            samples = list(getattr(sampler, "samples", []))
    path = os.path.join(attempt_dir, "gpu.samples.jsonl")
    payload = b"".join(canonical_bytes({
        "monotonicSeconds": row.get("t"),
        "gpuUtilizationPercent": row.get("dev"),
        "gpuPowerWatts": row.get("gpu_w"),
        "gpuMemoryBytes": row.get("mem"),
    }) + b"\n" for row in samples)
    atomic_write_bytes(path, payload)
    return {"summary": _gpu_summary(samples, request_started),
            "samplesPath": path}


def _relative_paths(store, value):
    if isinstance(value, dict):
        return {key: _relative_paths(store, item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_paths(store, item) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        try:
            if os.path.commonpath([store.root, value]) == store.root:
                return store.relative(value)
        except ValueError:
            pass
    return value


def _start_gpu(hooks, disabled=False):
    if disabled:
        return {"off": "disabled by case"}
    factory = hooks.get("telemetry_start") if hooks else None
    if not factory:
        return {"off": "no telemetry hook"}
    try:
        return factory(False)
    except Exception as exc:
        return {"off": "gpu sampler failed: {}".format(exc)}


def safety_allows_case(store, runtime, case, lane, phase, limits):
    if not limits:
        return True
    target_pid = ((runtime or {}).get("identity") or {}).get("pid")
    snapshot = memory_snapshot(target_pid)
    reasons = evaluate_safety(snapshot, snapshot, limits)
    if not reasons:
        return True
    store.state["safetyAbort"] = {
        "caseId": case["id"], "lane": lane, "phase": phase,
        "stage": "before_case", "reasons": reasons,
        "memory": snapshot}
    store.save_state()
    store.event("safety_stop", **store.state["safetyAbort"])
    return False


def execute_case(store, case, adapter, runtime, key, lane, phase, order,
                 safety, hooks, progress, replicate=1,
                 execution_index=None, execution_order=None):
    attempt_dir, attempt = store.new_attempt(key)
    input_sha = sha256_json(case["request"])
    envelope = {
        "schema": ADAPTER_REQUEST_SCHEMA,
        "manifestSha256": store.manifest_sha,
        "caseId": case["id"],
        "category": case.get("category"),
        "lane": lane,
        "phase": phase,
        "order": order,
        "replicate": replicate,
        "executionIndex": execution_index,
        "executionOrder": execution_order,
        "inputSha256": input_sha,
        "maxOutputTokens": case.get("maxOutputTokens"),
        "request": case["request"],
    }
    atomic_write_json(os.path.join(attempt_dir, "input.json"), envelope)
    progress("{} {} {}".format(lane, phase, case["id"]))
    started_at = utc_now()
    started = time.monotonic()
    sampler = _start_gpu(hooks, case.get("telemetry") is False)
    if not isinstance(sampler, dict):
        time.sleep(0.3)
    request_started = time.monotonic()
    target_pid = ((runtime or {}).get("identity") or {}).get("pid")
    before = memory_snapshot(target_pid)
    response = None
    error = None
    try:
        response = call_adapter(adapter, envelope, attempt_dir)
    except (AdapterError, TypeError, ValueError) as exc:
        error = {"message": str(exc),
                 "details": exc.details if isinstance(exc, AdapterError)
                 else {"kind": "adapter_configuration"}}
    elapsed = time.monotonic() - started
    gpu = _stop_gpu(sampler, request_started, attempt_dir)
    after = memory_snapshot(target_pid)
    safety_reasons = evaluate_safety(before, after, safety)
    runtime_is_alive = runtime_alive(runtime)
    if response is not None:
        text_path = os.path.join(attempt_dir, "model.txt")
        atomic_write_bytes(text_path, response["text"].encode("utf-8"))
        atomic_write_json(os.path.join(attempt_dir, "response.json"), {
            "runtimeOk": response["runtimeOk"],
            "usage": response["usage"],
            "timings": response["timings"],
            "received": response["received"],
            "reportedQuality": response["reportedQuality"],
            "agentTracePresent": response["agentTrace"] is not None,
        })
        quality = evaluate_quality(case, response, attempt_dir)
        runtime_status = "pass" if response["runtimeOk"] \
            and runtime_is_alive else "fail"
        received_pid = response["received"].get("runtimePid")
        received_process = process_identity(received_pid) \
            if received_pid is not None else None
        usage = response["usage"]
        timings = response["timings"]
        received = response["received"]
        wire_evidence = capture_wire_evidence(
            response.get("wireEvidence"), attempt_dir, store)
        agent_trace = persist_agent_trace(
            response.get("agentTrace"), attempt_dir, store)
        transport = _relative_paths(store, response["transport"])
        text_rel = store.relative(text_path)
    else:
        quality = {"status": "not_evaluated", "checks": []}
        runtime_status = "fail"
        received_process = None
        usage = {"promptTokens": None, "completionTokens": None,
                 "totalTokens": None}
        timings = {"endToEndMs": elapsed * 1000.0}
        received, transport, text_rel = {}, {}, None
        wire_evidence = None
        agent_trace = None
    if not runtime_is_alive:
        error = error or {"message": "runtime exited during case",
                          "details": {}}
    receipt = {
        "schema": CASE_RESULT_SCHEMA,
        "manifestSha256": store.manifest_sha,
        "key": key,
        "caseId": case["id"],
        "category": case.get("category"),
        "lane": lane,
        "phase": phase,
        "order": order,
        "replicate": replicate,
        "executionIndex": execution_index,
        "executionOrder": execution_order,
        "attempt": attempt,
        "startedAt": started_at,
        "endedAt": utc_now(),
        "elapsedSeconds": elapsed,
        "inputSha256": input_sha,
        "maxOutputTokens": case.get("maxOutputTokens"),
        "requestPath": store.relative(os.path.join(attempt_dir, "input.json")),
        "modelTextPath": text_rel,
        "transport": transport,
        "serverStderrPath": (runtime or {}).get("stderrPath"),
        "runtimeProcess": (runtime or {}).get("identity"),
        "receivedProcess": received_process,
        "wireEvidence": wire_evidence,
        "agentTrace": agent_trace,
        "launchSequence": (runtime or {}).get("launchSequence"),
        "usage": usage,
        "timings": timings,
        "received": received,
        "runtimeStatus": runtime_status,
        "runtimeError": _relative_paths(store, error),
        "qualityStatus": quality["status"],
        "quality": _relative_paths(store, quality),
        "memory": {"before": before, "after": after,
                   "swapoutDeltaBytes": memory_delta(before, after)},
        "gpu": {"summary": gpu["summary"],
                "samplesPath": store.relative(gpu["samplesPath"])
                if gpu["samplesPath"] else None},
        "safetyAbort": safety_reasons or None,
    }
    store.commit_result(key, attempt_dir, receipt)
    if safety_reasons and not store.state.get("safetyAbort"):
        store.state["safetyAbort"] = {
            "caseId": case["id"], "lane": lane, "phase": phase,
            "reasons": safety_reasons}
        store.save_state()
        store.event("safety_stop", **store.state["safetyAbort"])
    return receipt


def _write_current_queue_summary(store, interrupted=False):
    summary = queue_summary(store, store.completed_results(), interrupted)
    if store.state.get("stopReason") and not summary.get("stopReason"):
        summary["stopReason"] = store.state["stopReason"]
        if summary["exitCode"] == 0:
            summary.update(status="partial", verdict="INCOMPLETE", exitCode=2)
    store.write_summary(summary)
    return summary


def run_queue(manifest, artifact_dir, hooks, progress):
    validate_queue_manifest(manifest)
    tool = {"name": "picchio", "version": hooks.get("version"),
            "protocol": hooks.get("protocol")}
    with ArtifactStore(artifact_dir, manifest, tool) as store:
        cases = manifest["cases"]
        store.state["expectedCases"] = len(cases)
        store.save_state()
        store.event("run_opened", mode="queue",
                    resumed=bool(store.state["completed"]))
        interrupted = False
        runtime = None
        try:
            runtime = ensure_runtime(store, manifest.get("runtime"), "queue")
            for order, case in enumerate(cases, 1):
                key = "queue:normal:{}".format(case["id"])
                if store.is_complete(key):
                    progress("resume skip {}".format(case["id"]))
                    continue
                if store.state.get("safetyAbort"):
                    break
                if runtime and not runtime_alive(runtime):
                    store.state["stopReason"] = "server_exited"
                    store.save_state()
                    break
                if not safety_allows_case(
                        store, runtime, case, "queue", "normal",
                        manifest.get("safety")):
                    break
                execute_case(store, case, manifest["adapter"], runtime, key,
                             "queue", "normal", order,
                             manifest.get("safety"), hooks, progress)
                _write_current_queue_summary(store)
                if runtime and not runtime_alive(runtime):
                    store.state["stopReason"] = "server_exited"
                    store.save_state()
                    break
                if store.state.get("safetyAbort"):
                    break
                if manifest.get("stopOnRuntimeFailure"):
                    latest = store.completed_results()[-1]
                    if latest.get("runtimeStatus") != "pass":
                        store.state["stopReason"] = "runtime_failure"
                        store.save_state()
                        break
        except KeyboardInterrupt:
            interrupted = True
            store.event("run_interrupted")
        except RuntimeError as exc:
            store.state["stopReason"] = "runtime_unavailable"
            store.state["runtimeError"] = str(exc)
            store.save_state()
            store.event("runtime_unavailable", message=str(exc))
        summary = _write_current_queue_summary(store, interrupted)
        store.event("run_closed", verdict=summary["verdict"],
                    exitCode=summary["exitCode"])
        return summary
