"""Hidden HR and AR checks for deterministic boundary probes."""
import hashlib
import json
from pathlib import Path
import sys


def load(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        if not isinstance(decoded, str):
            yield from objects(decoded)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def read_trace(path):
    entries = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("request"), dict)]


def tool_call(entry, name, value):
    calls = []
    for obj in objects(entry.get("response")):
        function = obj.get("function", {})
        if isinstance(function, dict) and function.get("name") == name:
            args = function.get("arguments")
            if isinstance(args, str):
                args = load_args(args)
            calls.append(isinstance(args, dict) and set(args) == {"control", "value"}
                         and args.get("control") == "commit" and type(args.get("value")) is int and args["value"] == value)
    return calls == [True]


def load_args(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def schema_visible(entry, variant):
    for obj in objects(entry["request"]):
        fn = obj.get("function", {})
        if not isinstance(fn, dict) or fn.get("name") != "decide_" + variant:
            continue
        params = fn.get("parameters", {})
        props = list(params.get("properties", {}))
        expected = ["value", "control"] if variant == "last" else ["control", "value"]
        if props == expected and set(params.get("required", [])) == {"control", "value"}:
            return True
    return False


def thinking_cap(request):
    caps = []
    thinking = request.get("thinking", {})
    template = request.get("chat_template_kwargs", {})
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            caps.append(0)
        if type(thinking.get("budget_tokens")) is int and thinking["budget_tokens"] >= 0:
            caps.append(thinking["budget_tokens"])
    if isinstance(template, dict) and template.get("enable_thinking") is False:
        caps.append(0)
    if type(request.get("thinking_budget")) is int and request["thinking_budget"] >= 0:
        caps.append(request["thinking_budget"])
    return min(caps) if caps else None


def constraint_enabled(request):
    fmt = request.get("response_format")
    if fmt is None:
        return False
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_object":
        return True
    spec = fmt.get("json_schema")
    if fmt.get("type") == "json_schema" and isinstance(spec, dict):
        schema = spec.get("schema")
        if isinstance(schema, dict) and schema.get("type") == "object":
            return True
    return None


def verify(kind, workspace, trace):
    ws = Path(workspace)
    initial = Path(__file__).parent / kind / "workspace/input.json"
    config = load(initial, {})
    state = load(ws / ".probe-state.json", {})
    answer = load(ws / "checked.json", None)
    entries = read_trace(trace)
    calls = state.get("calls", [])
    value = sum(config["rows"])
    hr = {"trace_present": bool(entries), "initial_firmware_preserved": load(ws / "input.json", {}) == config}
    ar = {"checked_json_present": isinstance(answer, dict)}
    if kind.startswith(("tk-002", "tk-003")):
        variants = config["variants"]
        expected = {v: {"control": "commit", "value": value} for v in variants}
        ar["exact_decisions"] = answer == expected
        for v in variants:
            matching = [e for e in entries if schema_visible(e, v)]
            if kind.startswith("tk-003"):
                matching = [e for e in matching if constraint_enabled(e["request"]) is (v == "on")]
            hr[v + "_schema_visible"] = bool(matching)
            hr[v + "_tool_recalled_and_valid"] = any(tool_call(e, "decide_" + v, value) for e in matching)
            hr[v + "_committed_once"] = sum(c.get("command") == "commit" and c.get("variant") == v and c.get("valid") is True for c in calls) == 1
        hr["no_invalid_commits"] = not any(c.get("command") == "commit" and c.get("valid") is not True for c in calls)
    elif kind.startswith("tk-004"):
        ar["exact_completed_result"] = answer == {"completed": True, "value": value}
        capped, completed = [], []
        for e in entries:
            cap = thinking_cap(e["request"])
            if cap is not None and cap <= config["thinking_cap"]:
                capped.append(e)
                response = e.get("response", {})
                if isinstance(response, dict):
                    choices = response.get("choices", [])
                    usage = response.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
                    if (choices and all(c.get("finish_reason") in ("stop", "tool_calls") for c in choices)
                            and type(usage) is int and 0 <= usage <= cap):
                        completed.append(e)
        hr["thinking_explicitly_capped"] = bool(capped) and len(capped) == len(entries)
        hr["round_completed_within_cap"] = bool(completed)
        runs = [c for c in calls if c.get("command") == "run"]
        hr["firmware_completed_once"] = len(runs) == 1 and runs[0].get("completed") is True
    else:
        snap = "snap-" + hashlib.sha256(json.dumps(config["rows"]).encode()).hexdigest()[:16]
        digest = hashlib.sha256(json.dumps(config["rows"], separators=(",", ":")).encode()).hexdigest()
        ar["snapshot_matches"] = isinstance(answer, dict) and answer.get("snapshot_id") == snap
        ar["exact_recovered_rows"] = answer == {"snapshot_id": snap, "total": len(config["rows"]), "sum": value, "sha256": digest}
        prefix = json.dumps({"snapshot_id": snap, "rows": config["rows"]}, separators=(",", ":"))[:config["cut_bytes"]]
        hr["export_once"] = sum(c.get("command") == "export" for c in calls) == 1
        hr["no_raw_cut_reinjection"] = bool(entries) and all(prefix not in s for e in entries for s in strings(e["request"]))
        handles = [o for e in entries for o in objects(e["request"]) if o.get("truncated") is True
                   and isinstance(o.get("recovery"), dict) and o["recovery"].get("snapshot_id") == snap
                   and o["recovery"].get("command") == "fetch"]
        hr["structured_recovery_handle"] = bool(handles)
        covered = set()
        fetches = [c for c in calls if c.get("command") == "fetch"]
        valid_fetches = [c for c in fetches if c.get("snapshot") == snap and c.get("offset", -1) >= 0 and 1 <= c.get("limit", 0) <= 50]
        for c in valid_fetches:
            covered.update(range(c["offset"], min(len(config["rows"]), c["offset"] + c["limit"])))
        hr["recovery_pages_complete"] = len(valid_fetches) == len(fetches) and covered == set(range(len(config["rows"])))
    rate = lambda checks: sum(v is True for v in checks.values()) / len(checks)
    return {"task": kind, "hr": hr, "ar": ar, "hr_rate": rate(hr), "ar_rate": rate(ar),
            "score": round(50 * rate(hr) + 50 * rate(ar), 2)}


def main(kind):
    target = Path(sys.argv[1])
    if "--trace" in sys.argv:
        workspace, trace = target, sys.argv[sys.argv.index("--trace") + 1]
    else:
        workspace, trace = target / "workspace", target / "tap.jsonl"
    print(json.dumps(verify(kind, workspace, trace), indent=2))
    return 0
