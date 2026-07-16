"""Same-build bare/product parity built on the queue receipt protocol."""

import hashlib
import os
import statistics

from . import PARITY_SCHEMA
from .artifacts import ArtifactStore
from .runner import (ManifestError, execute_case, safety_allows_case,
                     validate_cases)
from .runtime import RuntimeError, ensure_runtime, runtime_alive
from .summary import metric_median, write_rollups


FINGERPRINT_FIELDS = (
    "modelSha256", "runtimeBuild", "context", "kv", "parallel",
    "sampling", "maxOutputTokens", "projector")
EXECUTION_ORDERS = ("interleaved", "abba")
WIRE_PROVENANCE = {
    "baseline": "direct_engine_send",
    "product": "product_engine_boundary_capture",
}


def validate_parity_manifest(manifest):
    if manifest.get("schema") != PARITY_SCHEMA:
        raise ManifestError("parity manifest schema must be {}".format(
            PARITY_SCHEMA))
    if not isinstance(manifest.get("fingerprint"), dict):
        raise ManifestError("parity manifest requires fingerprint")
    lanes = manifest.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != {"baseline", "product"}:
        raise ManifestError("parity lanes must be baseline and product")
    for name in ("baseline", "product"):
        if not isinstance(lanes[name], dict) \
                or not isinstance(lanes[name].get("adapter"), dict):
            raise ManifestError("lane {} requires adapter".format(name))
    validate_cases(manifest.get("cases"))
    calibration = manifest.get("calibrationCases", [])
    if calibration:
        validate_cases(calibration, "calibrationCases")
    execution_order = manifest.get("executionOrder", "interleaved")
    if execution_order not in EXECUTION_ORDERS:
        raise ManifestError("parity executionOrder must be interleaved or "
                            "abba")
    return manifest


def _static_issues(manifest):
    expected = manifest["fingerprint"]
    issues = []
    lane_values = {}
    for lane in ("baseline", "product"):
        actual = manifest["lanes"][lane].get("fingerprint")
        if not isinstance(actual, dict):
            issues.append({"code": "lane_fingerprint_missing", "lane": lane})
            actual = {}
        lane_values[lane] = actual
        for field in FINGERPRINT_FIELDS:
            if field not in expected:
                issues.append({"code": "expected_fingerprint_missing",
                               "field": field})
            if field not in actual:
                issues.append({"code": "lane_fingerprint_field_missing",
                               "lane": lane, "field": field})
            elif field in expected and actual[field] != expected[field]:
                issues.append({"code": "fingerprint_mismatch", "lane": lane,
                               "field": field, "expected": expected[field],
                               "actual": actual[field]})
    for field in FINGERPRINT_FIELDS:
        left = lane_values["baseline"].get(field)
        right = lane_values["product"].get(field)
        if left != right:
            issues.append({"code": "lane_fingerprint_difference",
                           "field": field, "baseline": left,
                           "product": right})
    return issues


def _contract_issues(result, fingerprint, lane, artifact_root):
    received = result.get("received") or {}
    issues = []

    def need(field, expected=None):
        if field not in received:
            issues.append({"code": "received_field_missing", "field": field})
        elif expected is not None and received[field] != expected:
            issues.append({"code": "received_field_mismatch", "field": field,
                           "expected": expected, "actual": received[field]})

    need("inputSha256", result.get("inputSha256"))
    need("promptTokens")
    need("modelSha256", fingerprint.get("modelSha256"))
    need("context", fingerprint.get("context"))
    need("runtimeBuild", fingerprint.get("runtimeBuild"))
    need("kv", fingerprint.get("kv"))
    need("parallel", fingerprint.get("parallel"))
    need("sampling", fingerprint.get("sampling"))
    need("maxOutputTokens", result.get("maxOutputTokens"))
    need("launchArgs")
    need("runtimePid")
    need("runtimeStartTime")
    need("launchSequence")
    need("projector", fingerprint.get("projector"))
    need("cacheN")
    need("toolRounds", 0)
    prompt_usage = (result.get("usage") or {}).get("promptTokens")
    if not isinstance(prompt_usage, (int, float)):
        issues.append({"code": "usage_prompt_tokens_missing"})
    elif received.get("promptTokens") != prompt_usage:
        issues.append({"code": "prompt_tokens_usage_mismatch",
                       "received": received.get("promptTokens"),
                       "usage": prompt_usage})
    cache_n = received.get("cacheN")
    if "cacheN" in received and not isinstance(cache_n, (int, float)):
        issues.append({"code": "cache_state_invalid", "actual": cache_n})
    wire = result.get("wireEvidence")
    if not isinstance(wire, dict):
        issues.append({"code": "wire_body_evidence_missing"})
    else:
        if not wire.get("valid"):
            issues.append({"code": "wire_body_evidence_invalid",
                           "error": wire.get("error")})
        expected_provenance = WIRE_PROVENANCE[lane]
        if wire.get("provenance") != expected_provenance:
            issues.append({"code": "wire_body_provenance_mismatch",
                           "expected": expected_provenance,
                           "actual": wire.get("provenance")})
        if not wire.get("bodySha256") or not wire.get("bodyPath"):
            issues.append({"code": "wire_body_artifact_missing"})
        else:
            path = os.path.abspath(os.path.join(
                artifact_root, wire["bodyPath"]))
            try:
                within_artifact = os.path.commonpath(
                    [artifact_root, path]) == artifact_root
            except ValueError:
                within_artifact = False
            if not within_artifact:
                issues.append({"code": "wire_body_artifact_outside_run"})
            else:
                try:
                    with open(path, "rb") as handle:
                        digest = hashlib.sha256(handle.read()).hexdigest()
                except OSError as exc:
                    issues.append({"code": "wire_body_artifact_unreadable",
                                   "error": str(exc)})
                else:
                    if digest != wire.get("bodySha256"):
                        issues.append({
                            "code": "wire_body_artifact_digest_mismatch",
                            "expected": wire.get("bodySha256"),
                            "actual": digest})
    actual = result.get("receivedProcess")
    if received.get("runtimePid") is not None:
        if not actual:
            issues.append({"code": "runtime_process_not_observable"})
        else:
            if actual.get("pid") != received.get("runtimePid"):
                issues.append({"code": "runtime_pid_mismatch",
                               "actual": actual.get("pid")})
            if actual.get("startTime") != received.get("runtimeStartTime"):
                issues.append({"code": "runtime_start_time_mismatch",
                               "actual": actual.get("startTime")})
            if not actual.get("executablePath"):
                issues.append({"code": "runtime_executable_unresolved"})
            if actual.get("parentPid") is None:
                issues.append({"code": "runtime_parent_unresolved"})
            observed = result.get("runtimeProcess")
            if isinstance(observed, dict):
                for field in ("pid", "startTime", "executablePath",
                              "parentPid"):
                    if observed.get(field) != actual.get(field):
                        issues.append({
                            "code": "observed_runtime_identity_mismatch",
                            "field": field, "expected": observed.get(field),
                            "actual": actual.get(field)})
    return issues


def _schedule(cases, execution_order):
    scheduled = []
    for case_order, case in enumerate(cases, 1):
        if execution_order == "abba":
            visits = (("baseline", 1), ("product", 1),
                      ("product", 2), ("baseline", 2))
        else:
            lanes = ("baseline", "product") if case_order % 2 else \
                ("product", "baseline")
            visits = tuple((lane, 1) for lane in lanes)
        for lane, replicate in visits:
            scheduled.append({
                "lane": lane,
                "case": case,
                "caseOrder": case_order,
                "replicate": replicate,
                "executionIndex": len(scheduled) + 1,
            })
    return scheduled


def _pair_differences(left, right, case_id, phase, replicate):
    issues = []

    def difference(code, baseline, product):
        if baseline != product:
            issues.append({"code": code, "caseId": case_id,
                           "phase": phase, "replicate": replicate,
                           "baseline": baseline, "product": product})

    difference("lane_input_hash_difference", left.get("inputSha256"),
               right.get("inputSha256"))
    left_received = left.get("received") or {}
    right_received = right.get("received") or {}
    for field, code in (
            ("promptTokens", "lane_prompt_tokens_difference"),
            ("modelSha256", "lane_model_sha256_difference"),
            ("parallel", "lane_parallel_difference"),
            ("context", "lane_context_difference"),
            ("runtimeBuild", "lane_runtime_build_difference"),
            ("kv", "lane_kv_difference"),
            ("sampling", "lane_sampling_difference"),
            ("maxOutputTokens", "lane_max_output_difference"),
            ("launchArgs", "lane_launch_args_difference"),
            ("runtimePid", "lane_runtime_pid_difference"),
            ("runtimeStartTime", "lane_runtime_start_time_difference"),
            ("launchSequence", "lane_launch_sequence_difference"),
            ("projector", "lane_projector_difference"),
            ("cacheN", "lane_cache_state_difference"),
            ("toolRounds", "lane_tool_rounds_difference")):
        difference(code, left_received.get(field), right_received.get(field))
    difference("lane_usage_prompt_tokens_difference",
               (left.get("usage") or {}).get("promptTokens"),
               (right.get("usage") or {}).get("promptTokens"))
    difference("lane_wire_body_difference",
               (left.get("wireEvidence") or {}).get("bodySha256"),
               (right.get("wireEvidence") or {}).get("bodySha256"))
    left_process = left.get("receivedProcess") or {}
    right_process = right.get("receivedProcess") or {}
    for field in ("pid", "startTime", "executablePath", "parentPid"):
        baseline = left_process.get(field)
        product = right_process.get(field)
        if baseline != product:
            issues.append({"code": "lane_runtime_process_difference",
                           "field": field, "caseId": case_id,
                           "phase": phase, "replicate": replicate,
                           "baseline": baseline, "product": product})
    return issues


def _pair_issues(results, fingerprint, phases, execution_order,
                 artifact_root):
    issues = []
    by_key = {}
    for row in results:
        key = (row.get("lane"), row.get("phase"), row.get("caseId"),
               row.get("replicate", 1))
        if key in by_key:
            issues.append({"code": "duplicate_case_result", "lane": key[0],
                           "phase": key[1], "caseId": key[2],
                           "replicate": key[3]})
        else:
            by_key[key] = row
    expected_keys = set()
    for phase, cases in phases:
        scheduled = _schedule(cases, execution_order)
        for item in scheduled:
            lane = item["lane"]
            case = item["case"]
            replicate = item["replicate"]
            key = (lane, phase, case["id"], replicate)
            expected_keys.add(key)
            row = by_key.get(key)
            if not row:
                issues.append({"code": "case_missing", "lane": lane,
                               "phase": phase, "caseId": case["id"],
                               "replicate": replicate})
                continue
            for field, expected in (
                    ("order", item["caseOrder"]),
                    ("executionIndex", item["executionIndex"]),
                    ("executionOrder", execution_order)):
                if row.get(field) != expected:
                    issues.append({"code": "case_order_mismatch",
                                   "lane": lane, "phase": phase,
                                   "caseId": case["id"],
                                   "replicate": replicate, "field": field,
                                   "actual": row.get(field),
                                   "expected": expected})
            for issue in _contract_issues(row, fingerprint, lane,
                                          artifact_root):
                issue.update(lane=lane, phase=phase, caseId=case["id"],
                             replicate=replicate)
                issues.append(issue)
        repeats = (1, 2) if execution_order == "abba" else (1,)
        for case in cases:
            for replicate in repeats:
                left = by_key.get(("baseline", phase, case["id"], replicate))
                right = by_key.get(("product", phase, case["id"], replicate))
                if left and right:
                    issues.extend(_pair_differences(
                        left, right, case["id"], phase, replicate))
    for key in sorted(set(by_key) - expected_keys, key=str):
        issues.append({"code": "unexpected_case_result", "lane": key[0],
                       "phase": key[1], "caseId": key[2],
                       "replicate": key[3]})
    return issues


def _pct(product, baseline, lower_is_better=True):
    if product is None or baseline in (None, 0):
        return None
    raw = (product - baseline) * 100.0 / baseline
    return raw if lower_is_better else -raw


def _performance(results, lane, phase="normal"):
    rows = [row for row in results if row.get("lane") == lane
            and row.get("phase") == phase]
    fields = ("routeAdmissionMs", "runtimeLoadMs", "serverTtftMs",
              "serverPrefillTokensPerSecond",
              "serverDecodeTokensPerSecond", "adapterFirstChunkMs",
              "endToEndMs")
    return {field: metric_median(rows, field) for field in fields}


def _lifecycle(results, lane):
    rows = [row for row in results if row.get("lane") == lane]
    sequences = [(row.get("received") or {}).get("launchSequence")
                 for row in rows]
    known = [value for value in sequences if value is not None]
    processes = {(row.get("received") or {}).get("runtimePid")
                 for row in rows
                 if (row.get("received") or {}).get("runtimePid") is not None}
    loads = [row for row in rows
             if isinstance((row.get("timings") or {}).get("runtimeLoadMs"),
                           (int, float))
             and (row.get("timings") or {}).get("runtimeLoadMs") > 0]
    return {"observedLaunchSequences": sorted(set(known)),
            "distinctRuntimePids": len(processes) if processes else None,
            "casesReportingLoad": len(loads),
            "warmReuseCases": len(known) - len(set(known)) if known else None}


def _latency_regressed(product, baseline, relative_threshold,
                       absolute_threshold):
    relative = _pct(product, baseline)
    return relative is not None and relative >= relative_threshold and \
        product - baseline >= absolute_threshold


def _attribution(performance, results, threshold, absolute_threshold):
    baseline, product = performance["baseline"], performance["product"]
    quality_regression = any(
        left.get("qualityStatus") == "pass" and
        right.get("qualityStatus") == "fail"
        for left in results for right in results
        if left.get("lane") == "baseline" and
        right.get("lane") == "product" and
        left.get("phase") == right.get("phase") == "normal" and
        left.get("caseId") == right.get("caseId"))
    if quality_regression:
        return "quality", "PRODUCT QUALITY REGRESSION"
    if _latency_regressed(product.get("runtimeLoadMs"),
                          baseline.get("runtimeLoadMs"), threshold,
                          absolute_threshold):
        return "runtime_lifecycle", "PRODUCT OVERHEAD"
    if _latency_regressed(product.get("routeAdmissionMs"),
                          baseline.get("routeAdmissionMs"), threshold,
                          absolute_threshold):
        return "route_admission", "PRODUCT OVERHEAD"
    decode = _pct(product.get("serverDecodeTokensPerSecond"),
                  baseline.get("serverDecodeTokensPerSecond"), False)
    ttft = _latency_regressed(product.get("serverTtftMs"),
                              baseline.get("serverTtftMs"), threshold,
                              absolute_threshold)
    if (decode is not None and decode >= threshold) or \
            ttft:
        return "engine", "PRODUCT OVERHEAD"
    first = _latency_regressed(product.get("adapterFirstChunkMs"),
                               baseline.get("adapterFirstChunkMs"), threshold,
                               absolute_threshold)
    total = _latency_regressed(product.get("endToEndMs"),
                               baseline.get("endToEndMs"), threshold,
                               absolute_threshold)
    if first or total:
        return "stream_bridge", "PRODUCT OVERHEAD"
    return "none", "NO MATERIAL OVERHEAD"


def parity_summary(store, manifest, results, interrupted=False):
    execution_order = manifest.get("executionOrder", "interleaved")
    phases = (("calibration", manifest.get("calibrationCases", [])),
              ("normal", manifest["cases"]))
    static = _static_issues(manifest)
    contract = _pair_issues(results, manifest["fingerprint"], phases,
                            execution_order, store.root)
    issues = static + contract
    performance = {lane: _performance(results, lane)
                   for lane in ("baseline", "product")}
    calibration_performance = {lane: _performance(
        results, lane, "calibration") for lane in ("baseline", "product")}
    lifecycle = {lane: _lifecycle(results, lane)
                 for lane in ("baseline", "product")}
    expected = store.state.get("expectedCases", 0)
    runtime_fail = sum(row.get("runtimeStatus") != "pass" for row in results)
    quality_fail = sum(row.get("qualityStatus") == "fail" for row in results)
    quality_missing = sum(row.get("qualityStatus") == "not_evaluated"
                          for row in results)
    safety = store.state.get("safetyAbort")
    if issues:
        location, verdict = None, "DIRECTIONAL"
        comparability = "directional"
    else:
        relative_threshold = float(manifest.get(
            "relativeRegressionPercent", 10))
        absolute_threshold = float(manifest.get(
            "absoluteLatencyRegressionMs", 10))
        location, verdict = _attribution(
            performance, results, relative_threshold, absolute_threshold)
        comparability = "causal"
    if interrupted:
        status, code, verdict = "interrupted", 130, "INTERRUPTED"
    elif safety:
        status, code, verdict = "stopped", 6, "SAFETY STOP"
    elif len(results) < expected:
        status, code = "partial", 2
    elif runtime_fail:
        status, code = "completed", 3
    elif quality_fail:
        status, code = "completed", 4
    elif issues:
        status, code = "completed", 5
    else:
        status, code = "completed", 0
    rollups = write_rollups(store, [row for row in results
                                    if row.get("phase") == "normal"])
    launched = store.state.get("loadCount", 0)
    observed_loads = {(row.get("lane"),
                       (row.get("received") or {}).get("launchSequence"))
                      for row in results
                      if (row.get("received") or {}).get("launchSequence")
                      is not None}
    load_count = launched if launched else (
        len(observed_loads) if observed_loads else None)
    load_source = "picchio_launches" if launched else (
        "adapter_received" if observed_loads else "unavailable")
    return {
        "schema": "picchio.parity.summary.v1",
        "manifestSha256": store.manifest_sha,
        "status": status,
        "verdict": verdict,
        "exitCode": code,
        "artifactDirectory": store.root,
        "comparability": comparability,
        "comparabilityIssues": issues,
        "causalAttribution": location,
        "executionOrder": execution_order,
        "regressionThresholds": {
            "relativePercent": float(manifest.get(
                "relativeRegressionPercent", 10)),
            "absoluteLatencyMs": float(manifest.get(
                "absoluteLatencyRegressionMs", 10)),
        },
        "performance": performance,
        "calibrationPerformance": calibration_performance,
        "lifecycle": lifecycle,
        "runtime": {"pass": len(results) - runtime_fail,
                    "fail": runtime_fail},
        "quality": {"pass": len(results) - quality_fail - quality_missing,
                    "fail": quality_fail,
                    "notEvaluated": quality_missing},
        "expectedLaneCases": expected,
        "completedLaneCases": len(results),
        "loadCount": load_count,
        "loadCountSource": load_source,
        "safetyAbort": safety,
        "rollups": rollups,
        "evidence": {"manifest": "manifest.json",
                     "checkpoint": "checkpoint.json",
                     "events": "events.jsonl", "summary": "summary.json",
                     "cases": "cases/", "rollups": "rollups/"},
    }


def run_parity(manifest, artifact_dir, hooks, progress):
    validate_parity_manifest(manifest)
    tool = {"name": "picchio", "version": hooks.get("version"),
            "protocol": hooks.get("protocol")}
    with ArtifactStore(artifact_dir, manifest, tool) as store:
        phases = (("calibration", manifest.get("calibrationCases", [])),
                  ("normal", manifest["cases"]))
        execution_order = manifest.get("executionOrder", "interleaved")
        expected = sum(len(_schedule(cases, execution_order))
                       for _phase, cases in phases)
        store.state["expectedCases"] = expected
        store.state["executionOrder"] = execution_order
        store.save_state()
        store.event("run_opened", mode="parity",
                    resumed=bool(store.state["completed"]),
                    executionOrder=execution_order)
        interrupted = False
        runtimes = {}
        try:
            for phase, cases in phases:
                for item in _schedule(cases, execution_order):
                    lane = item["lane"]
                    case = item["case"]
                    replicate = item["replicate"]
                    lane_spec = manifest["lanes"][lane]
                    runtime = runtimes.get(lane)
                    if runtime is None:
                        runtime = ensure_runtime(store, lane_spec.get("runtime"),
                                                 lane)
                        runtimes[lane] = runtime
                    key = "parity:{}:{}:{}".format(phase, lane, case["id"])
                    if execution_order == "abba":
                        key += ":r{}".format(replicate)
                    if store.is_complete(key):
                        progress("resume skip {} {} r{}".format(
                            lane, case["id"], replicate))
                        continue
                    if store.state.get("safetyAbort"):
                        break
                    if not runtime_alive(runtime):
                        store.state["stopReason"] = "server_exited"
                        store.save_state()
                        break
                    if not safety_allows_case(
                            store, runtime, case, lane, phase,
                            manifest.get("safety")):
                        break
                    execute_case(
                        store, case, lane_spec["adapter"], runtime, key, lane,
                        phase, item["caseOrder"], manifest.get("safety"),
                        hooks, progress, replicate=replicate,
                        execution_index=item["executionIndex"],
                        execution_order=execution_order)
                    current = parity_summary(
                        store, manifest, store.completed_results())
                    store.write_summary(current)
                    if runtime and not runtime_alive(runtime):
                        store.state["stopReason"] = "server_exited"
                        store.save_state()
                        break
                if store.state.get("safetyAbort") \
                        or store.state.get("stopReason"):
                    break
        except KeyboardInterrupt:
            interrupted = True
            store.event("run_interrupted")
        except RuntimeError as exc:
            store.state["stopReason"] = "runtime_unavailable"
            store.state["runtimeError"] = str(exc)
            store.save_state()
            store.event("runtime_unavailable", message=str(exc))
        summary = parity_summary(store, manifest, store.completed_results(),
                                 interrupted)
        if store.state.get("stopReason"):
            summary["stopReason"] = store.state["stopReason"]
        store.write_summary(summary)
        store.event("run_closed", verdict=summary["verdict"],
                    exitCode=summary["exitCode"])
        return summary
