"""Summaries are pure reductions of committed case receipts."""

import os
import statistics

from .artifacts import atomic_write_json


def _numbers(rows):
    return [float(value) for value in rows
            if isinstance(value, (int, float))]


def _median(rows):
    values = _numbers(rows)
    return statistics.median(values) if values else None


def _sum(rows):
    values = _numbers(rows)
    return sum(values) if values else 0


def _hour(value):
    return value[:13] + ":00:00Z" if isinstance(value, str) \
        and len(value) >= 13 else "unknown"


def _gpu_value(result, key):
    gpu = result.get("gpu") or {}
    summary = gpu.get("summary") or {}
    return summary.get(key)


def _work_usage(result):
    usage = result.get("usage") or {}
    trace = result.get("agentTrace") or {}
    trace_summary = trace.get("summary") or {}
    prompt = trace_summary.get("totalPromptTokensProcessed")
    completion = trace_summary.get("totalCompletionTokensProcessed")
    return {
        "promptTokens": prompt if isinstance(prompt, (int, float))
        else usage.get("promptTokens"),
        "completionTokens": completion
        if isinstance(completion, (int, float))
        else usage.get("completionTokens"),
        "currentPromptTokens": trace_summary.get("currentPromptTokens"),
        "peakPromptTokens": trace_summary.get("peakPromptTokens"),
    }


def rollup_rows(results):
    buckets = {}
    for result in results:
        buckets.setdefault(_hour(result.get("endedAt")), []).append(result)
    first_decode = None
    output = []
    for hour in sorted(buckets):
        rows = buckets[hour]
        elapsed = _sum(row.get("elapsedSeconds") for row in rows)
        usage = [_work_usage(row) for row in rows]
        timing = [row.get("timings") or {} for row in rows]
        memory = [row.get("memory") or {} for row in rows]
        after = [row.get("after") or {} for row in memory]
        decode = _median(row.get("serverDecodeTokensPerSecond")
                         for row in timing)
        if first_decode is None and decode is not None:
            first_decode = decode
        drift = ((decode - first_decode) * 100.0 / first_decode) \
            if decode is not None and first_decode else None
        prompt_tokens = _sum(row.get("promptTokens") for row in usage)
        completion_tokens = _sum(row.get("completionTokens")
                                 for row in usage)
        hours = elapsed / 3600.0 if elapsed > 0 else None
        output.append({
            "schema": "picchio.hour-rollup.v1",
            "hour": hour,
            "tasks": len(rows),
            "validTasks": sum(1 for row in rows
                              if row.get("qualityStatus") == "pass"),
            "runtimePass": sum(1 for row in rows
                               if row.get("runtimeStatus") == "pass"),
            "promptTokens": prompt_tokens,
            "totalPromptTokensProcessed": prompt_tokens,
            "completionTokens": completion_tokens,
            "currentPromptTokensMedian": _median(
                row.get("currentPromptTokens") for row in usage),
            "peakPromptTokensMedian": _median(
                row.get("peakPromptTokens") for row in usage),
            "tasksPerHour": len(rows) / hours if hours else None,
            "validTasksPerHour": sum(
                1 for row in rows if row.get("qualityStatus") == "pass") /
            hours if hours else None,
            "tokensPerHour": (prompt_tokens + completion_tokens) / hours
            if hours else None,
            "ttftMsMedian": _median(row.get("serverTtftMs")
                                    for row in timing),
            "prefillTokensPerSecondMedian": _median(
                row.get("serverPrefillTokensPerSecond") for row in timing),
            "decodeTokensPerSecondMedian": decode,
            "decodeDriftFromFirstHourPercent": drift,
            "endToEndMsMedian": _median(row.get("endToEndMs")
                                        for row in timing),
            "rssBytesPeak": max(_numbers(row.get("rssBytes")
                                         for row in after), default=None),
            "wiredBytesMedian": _median(row.get("wiredBytes")
                                        for row in after),
            "compressedBytesMedian": _median(row.get("compressedBytes")
                                             for row in after),
            "reclaimableBytesMinimum": min(_numbers(
                row.get("reclaimableBytes") for row in after), default=None),
            "swapoutDeltaBytes": _sum(row.get("swapoutDeltaBytes")
                                      for row in memory),
            "gpuUtilizationPercentMedian": _median(
                _gpu_value(row, "work_med") for row in rows),
            "gpuPowerWattsMedian": _median(
                _gpu_value(row, "work_w") for row in rows),
            "thermalRaised": any((row.get("after") or {}).get(
                "thermalRaised") is True for row in memory),
        })
    return output


def write_rollups(store, rows):
    directory = os.path.join(store.root, "rollups")
    os.makedirs(directory, exist_ok=True)
    rollups = rollup_rows(rows)
    for index, row in enumerate(rollups, 1):
        atomic_write_json(os.path.join(
            directory, "hour-{:04d}.json".format(index)), row)
    return rollups


def queue_summary(store, results, interrupted=False):
    rollups = write_rollups(store, results)
    runtime_fail = sum(1 for row in results
                       if row.get("runtimeStatus") != "pass")
    quality_fail = sum(1 for row in results
                       if row.get("qualityStatus") == "fail")
    quality_missing = sum(1 for row in results
                          if row.get("qualityStatus") == "not_evaluated")
    expected = store.state.get("expectedCases", len(results))
    safety = store.state.get("safetyAbort")
    if interrupted:
        status, code, verdict = "interrupted", 130, "INTERRUPTED"
    elif safety:
        status, code, verdict = "stopped", 6, "SAFETY STOP"
    elif runtime_fail:
        status, code, verdict = "completed", 3, "RUNTIME FAIL"
    elif quality_fail:
        status, code, verdict = "completed", 4, "QUALITY FAIL"
    elif len(results) < expected:
        status, code, verdict = "partial", 2, "INCOMPLETE"
    else:
        status, code, verdict = "completed", 0, "PASS"
    launched = store.state.get("loadCount", 0)
    observed_loads = {(row.get("received") or {}).get("launchSequence")
                      for row in results
                      if (row.get("received") or {}).get("launchSequence")
                      is not None}
    load_count = launched if launched else (
        len(observed_loads) if observed_loads else None)
    load_source = "picchio_launches" if launched else (
        "adapter_received" if observed_loads else "unavailable")
    agent_traces = []
    for row in results:
        trace = row.get("agentTrace")
        if not isinstance(trace, dict):
            continue
        trace_summary = trace.get("summary") or {}
        agent_traces.append({
            "caseId": row.get("caseId"),
            "status": trace.get("status"),
            "currentPromptTokens": trace_summary.get(
                "currentPromptTokens"),
            "totalPromptTokensProcessed": trace_summary.get(
                "totalPromptTokensProcessed"),
            "compactDecision": trace_summary.get("compactDecision"),
            "jsonPath": trace.get("jsonPath"),
            "markdownPath": trace.get("markdownPath"),
        })
    return {
        "schema": "picchio.queue.summary.v1",
        "manifestSha256": store.manifest_sha,
        "status": status,
        "verdict": verdict,
        "exitCode": code,
        "artifactDirectory": store.root,
        "expectedCases": expected,
        "completedCases": len(results),
        "runtime": {"pass": len(results) - runtime_fail,
                    "fail": runtime_fail},
        "quality": {"pass": len(results) - quality_fail - quality_missing,
                    "fail": quality_fail,
                    "notEvaluated": quality_missing},
        "loadCount": load_count,
        "loadCountSource": load_source,
        "safetyAbort": safety,
        "agentTraces": agent_traces,
        "rollups": rollups,
        "evidence": {
            "manifest": "manifest.json",
            "checkpoint": "checkpoint.json",
            "events": "events.jsonl",
            "summary": "summary.json",
            "cases": "cases/",
            "rollups": "rollups/",
        },
    }


def metric_median(results, name):
    return _median((row.get("timings") or {}).get(name) for row in results)
