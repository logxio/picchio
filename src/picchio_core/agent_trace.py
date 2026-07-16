"""Generic multi-round agent request accounting and human evidence tables."""

import math
import os

from . import AGENT_TRACE_SCHEMA
from .artifacts import atomic_write_bytes, atomic_write_json
from .evidence import capture_wire_evidence

COMPONENT_LABELS = {
    "user_input": "User input",
    "system_instructions": "System instructions",
    "document_context": "Document context",
    "conversation_history": "Conversation history",
    "retrieval_evidence": "Retrieval evidence",
    "tool_schemas": "Tool schemas",
    "tool_results": "Tool results",
    "attachments": "Attachments",
    "chat_template": "Chat template",
    "other": "Other context",
}

ROUND_PURPOSES = {
    "initial", "planning", "tool", "final", "repair", "verification",
    "other",
}

COUNT_SOURCES = {
    "measured", "tokenized", "derived", "estimated", "reported", "unknown",
}


def _number(value, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if value < (1 if positive else 0):
        return None
    if integer and isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if integer else value


def _usage(value):
    value = value if isinstance(value, dict) else {}
    prompt = value.get("promptTokens", value.get("prompt_tokens"))
    completion = value.get("completionTokens", value.get(
        "completion_tokens"))
    return {
        "promptTokens": _number(prompt, integer=True),
        "completionTokens": _number(completion, integer=True),
    }


def _timings(value):
    value = value if isinstance(value, dict) else {}

    def first(*names):
        for name in names:
            if name in value:
                return _number(value[name])
        return None

    return {
        "serverTtftMs": first("serverTtftMs", "ttftMs"),
        "serverPrefillTokensPerSecond": first(
            "serverPrefillTokensPerSecond", "prefillTokensPerSecond"),
        "serverDecodeTokensPerSecond": first(
            "serverDecodeTokensPerSecond", "decodeTokensPerSecond"),
        "serverTotalMs": first("serverTotalMs", "totalMs"),
    }


def _normalize_components(raw_components, warnings):
    if raw_components is None:
        return []
    if not isinstance(raw_components, list):
        warnings.append("components_not_array")
        return []
    raw_ids = {}
    for index, raw in enumerate(raw_components, 1):
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) \
                or not raw["id"].strip():
            continue
        if raw["id"] in raw_ids:
            warnings.append("component_{}_id_duplicate".format(index))
        else:
            raw_ids[raw["id"]] = "component-{}".format(index)
    components = []
    for index, raw in enumerate(raw_components, 1):
        if not isinstance(raw, dict):
            warnings.append("component_{}_not_object".format(index))
            continue
        kind = raw.get("kind")
        if kind not in COMPONENT_LABELS:
            warnings.append("component_{}_kind_unknown".format(index))
            kind = "other"
        source = raw.get("countSource", "reported")
        if source not in COUNT_SOURCES:
            warnings.append("component_{}_count_source_unknown".format(index))
            source = "unknown"
        token_source = raw.get("tokenSource", "unknown")
        if token_source not in COUNT_SOURCES:
            warnings.append("component_{}_token_source_unknown".format(index))
            token_source = "unknown"
        component_id = "component-{}".format(index)
        included_in = raw_ids.get(raw.get("includedIn"))
        if raw.get("includedIn") is not None and included_in is None:
            warnings.append("component_{}_parent_missing".format(index))
        components.append({
            "id": component_id,
            "kind": kind,
            "characters": _number(raw.get("characters"), integer=True),
            "bytes": _number(raw.get("bytes"), integer=True),
            "tokens": _number(raw.get("tokens"), integer=True),
            "items": _number(raw.get("items"), integer=True),
            "countSource": source,
            "tokenSource": token_source,
            "includedIn": included_in,
        })
    return components


def _normalize_rounds(raw_rounds, warnings):
    if not isinstance(raw_rounds, list) or not raw_rounds:
        warnings.append("rounds_missing")
        return []
    rounds = []
    seen = set()
    for position, raw in enumerate(raw_rounds):
        if not isinstance(raw, dict):
            warnings.append("round_{}_not_object".format(position))
            continue
        index = _number(raw.get("index", position), integer=True)
        if index is None or index in seen:
            warnings.append("round_{}_index_invalid".format(position))
            index = position
            while index in seen:
                index += 1
        seen.add(index)
        purpose = raw.get("purpose", "other")
        if purpose not in ROUND_PURPOSES:
            warnings.append("round_{}_purpose_unknown".format(index))
            purpose = "other"
        rounds.append({
            "index": index,
            "purpose": purpose,
            "visible": raw.get("visible")
            if isinstance(raw.get("visible"), bool) else None,
            "usage": _usage(raw.get("usage")),
            "timings": _timings(raw.get("timings")),
            "wireEvidenceRaw": raw.get("wireEvidence")
            if isinstance(raw.get("wireEvidence"), dict) else None,
        })
    rounds.sort(key=lambda row: row["index"])
    if [row["index"] for row in rounds] != list(range(len(rounds))):
        warnings.append("round_indexes_not_contiguous")
    return rounds


def _summary(trace, warnings):
    rounds = trace["rounds"]
    prompts = [row["usage"]["promptTokens"] for row in rounds]
    completions = [row["usage"]["completionTokens"] for row in rounds]
    all_prompts = bool(prompts) and all(value is not None for value in prompts)
    all_completions = bool(completions) and all(
        value is not None for value in completions)
    if not all_prompts:
        warnings.append("round_prompt_tokens_incomplete")
    current = prompts[-1] if prompts else None
    total = sum(prompts) if all_prompts else None
    peak = max(prompts) if all_prompts else None
    total_completion = sum(completions) if all_completions else None
    context = trace["context"]
    capacity = context["capacityTokens"]
    compact_at = context["compactAtTokens"]
    utilization = round(current * 100.0 / capacity, 2) \
        if current is not None and capacity else None
    should_compact = current >= compact_at \
        if current is not None and compact_at is not None else None
    reported = context["reportedCompactTriggered"]
    if should_compact is None:
        decision = "not_evaluated"
    elif reported is None:
        decision = "not_reported"
    elif reported == should_compact:
        decision = "correct"
    elif reported:
        decision = "false_positive"
    else:
        decision = "missed"
    visible_decode = trace["visibleDecodeTokensPerSecond"]
    if visible_decode is None:
        candidates = [row for row in rounds if row["visible"] is True]
        if not candidates:
            candidates = [row for row in rounds if row["purpose"] == "final"]
        if not candidates and rounds:
            candidates = [rounds[-1]]
        if candidates:
            visible_decode = candidates[-1]["timings"][
                "serverDecodeTokensPerSecond"]
    return {
        "roundCount": len(rounds),
        "firstRoundPromptTokens": prompts[0] if prompts else None,
        "currentPromptTokens": current,
        "peakPromptTokens": peak,
        "totalPromptTokensProcessed": total,
        "totalCompletionTokensProcessed": total_completion,
        "contextCapacityTokens": capacity,
        "compactAtTokens": compact_at,
        "contextUtilizationPercent": utilization,
        "reportedCompactTriggered": reported,
        "compactShouldTrigger": should_compact,
        "compactDecision": decision,
        "firstVisibleTokenMs": trace["firstVisibleTokenMs"],
        "visibleDecodeTokensPerSecond": visible_decode,
    }


def normalize_agent_trace(value):
    warnings = []
    if not isinstance(value, dict):
        return None
    schema_ok = value.get("schema") == AGENT_TRACE_SCHEMA
    if not schema_ok:
        warnings.append("schema_invalid")
    context_value = value.get("context") \
        if isinstance(value.get("context"), dict) else {}
    capacity = _number(context_value.get("capacityTokens"), integer=True,
                       positive=True)
    compact_at = _number(context_value.get("compactAtTokens"), integer=True,
                         positive=True)
    if capacity is None:
        warnings.append("context_capacity_missing")
    if compact_at is None:
        warnings.append("compact_threshold_missing")
    elif capacity is not None and compact_at > capacity:
        warnings.append("compact_threshold_exceeds_capacity")
        compact_at = None
    reported = context_value.get("reportedCompactTriggered")
    if not isinstance(reported, bool):
        reported = None
    trace = {
        "schema": AGENT_TRACE_SCHEMA,
        "components": _normalize_components(value.get("components"),
                                             warnings),
        "rounds": _normalize_rounds(value.get("rounds"), warnings),
        "context": {
            "capacityTokens": capacity,
            "compactAtTokens": compact_at,
            "reportedCompactTriggered": reported,
        },
        "firstVisibleTokenMs": _number(value.get("firstVisibleTokenMs")),
        "visibleDecodeTokensPerSecond": _number(
            value.get("visibleDecodeTokensPerSecond")),
        "warnings": warnings,
    }
    trace["summary"] = _summary(trace, warnings)
    trace["schemaValid"] = schema_ok
    return trace


def _format_number(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, int):
        return "{:,}".format(value)
    return ("{:,.%df}" % digits).format(value).rstrip("0").rstrip(".")


def _format_bool(value):
    return "yes" if value is True else "no" if value is False else "—"


def _format_ms(value):
    if value is None:
        return "—"
    if value >= 1000:
        return "{} s".format(_format_number(value / 1000.0))
    return "{} ms".format(_format_number(value))


def render_agent_trace_markdown(trace):
    components = trace["components"]
    by_id = {row["id"]: COMPONENT_LABELS[row["kind"]]
             for row in components}
    lines = [
        "# Agent trace",
        "",
        "## Request composition",
        "",
        "| Component | Characters | Bytes | Tokens | Evidence | Included in |",
        "|---|---:|---:|---:|---|---|",
    ]
    if components:
        for row in components:
            evidence = row["countSource"]
            if row["tokens"] is not None:
                evidence += "; tokens " + row["tokenSource"]
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                COMPONENT_LABELS[row["kind"]],
                _format_number(row["characters"]),
                _format_number(row["bytes"]),
                _format_number(row["tokens"]),
                evidence,
                by_id.get(row["includedIn"], "—"),
            ))
    else:
        lines.append("| No component evidence | — | — | — | — | — |")
    if any(row["includedIn"] for row in components):
        lines.extend([
            "",
            "Rows with `Included in` are subsets and must not be added to "
            "their parent.",
        ])
    lines.extend([
        "",
        "## Inference rounds",
        "",
        "| Round | Purpose | Prompt tokens | Completion tokens | TTFT | "
        "Prefill tok/s | Decode tok/s | Wire bytes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in trace["rounds"]:
        usage = row["usage"]
        timing = row["timings"]
        wire = row.get("wireEvidence") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["index"], row["purpose"],
            _format_number(usage["promptTokens"]),
            _format_number(usage["completionTokens"]),
            _format_ms(timing["serverTtftMs"]),
            _format_number(timing["serverPrefillTokensPerSecond"]),
            _format_number(timing["serverDecodeTokensPerSecond"]),
            _format_number(wire.get("byteCount")),
        ))
    summary = trace["summary"]
    lines.extend([
        "",
        "## Context accounting",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Current prompt tokens | {} |".format(_format_number(
            summary["currentPromptTokens"])),
        "| Peak prompt tokens | {} |".format(_format_number(
            summary["peakPromptTokens"])),
        "| Total prompt tokens processed | {} |".format(_format_number(
            summary["totalPromptTokensProcessed"])),
        "| Context capacity | {} |".format(_format_number(
            summary["contextCapacityTokens"])),
        "| Compact threshold | {} |".format(_format_number(
            summary["compactAtTokens"])),
        "| Current context utilization | {}% |".format(_format_number(
            summary["contextUtilizationPercent"])),
        "| Reported compact | {} |".format(_format_bool(
            summary["reportedCompactTriggered"])),
        "| Compact should trigger | {} |".format(_format_bool(
            summary["compactShouldTrigger"])),
        "| Compact decision | {} |".format(
            summary["compactDecision"].replace("_", " ").upper()),
        "| First visible token | {} |".format(_format_ms(
            summary["firstVisibleTokenMs"])),
        "| Visible decode | {} tok/s |".format(_format_number(
            summary["visibleDecodeTokensPerSecond"])),
    ])
    if trace["warnings"]:
        lines.extend([
            "",
            "## Evidence warnings",
            "",
        ])
        lines.extend("- `{}`".format(value) for value in trace["warnings"])
    return "\n".join(lines) + "\n"


def persist_agent_trace(value, attempt_dir, store):
    trace = normalize_agent_trace(value)
    if trace is None:
        return None
    for row in trace["rounds"]:
        supplied = row.pop("wireEvidenceRaw")
        relative = os.path.join(
            "agent-trace", "round-{:03d}-engine-wire-body.bin".format(
                row["index"]))
        row["wireEvidence"] = capture_wire_evidence(
            supplied, attempt_dir, store, relative)
        if row["wireEvidence"] is None:
            trace["warnings"].append(
                "round_{}_wire_body_evidence_missing".format(row["index"]))
        elif not row["wireEvidence"].get("valid"):
            trace["warnings"].append(
                "round_{}_wire_body_evidence_invalid".format(row["index"]))
    if not trace["schemaValid"] or not trace["rounds"]:
        trace["status"] = "invalid"
    elif trace["warnings"]:
        trace["status"] = "partial"
    else:
        trace["status"] = "complete"
    json_path = os.path.join(attempt_dir, "agent-trace.json")
    markdown_path = os.path.join(attempt_dir, "agent-trace.md")
    trace["jsonPath"] = store.relative(json_path)
    trace["markdownPath"] = store.relative(markdown_path)
    atomic_write_json(json_path, trace)
    atomic_write_bytes(markdown_path,
                       render_agent_trace_markdown(trace).encode("utf-8"))
    return trace
