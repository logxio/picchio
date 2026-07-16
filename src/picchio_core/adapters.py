"""Stable JSON adapters for HTTP engines and arbitrary product backends."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from .artifacts import atomic_write_json
from .host import resolve_executable, stable_process_identity


class AdapterError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def _read_limited(path, limit):
    size = os.path.getsize(path)
    if size > limit:
        raise AdapterError("adapter response exceeded {} bytes".format(limit),
                           {"responseBytes": size, "limitBytes": limit})
    with open(path, "rb") as handle:
        return handle.read()


def command_json(config, payload, directory, prefix="adapter"):
    command = config.get("command")
    if not isinstance(command, list) or not command \
            or not all(isinstance(part, str) and part for part in command):
        raise AdapterError("command adapter requires a non-empty string list")
    timeout = float(config.get("timeoutSeconds", 600))
    limit = int(config.get("maxResponseBytes", 16 * 1024 * 1024))
    request_path = os.path.join(directory, "{}.request.json".format(prefix))
    stdout_path = os.path.join(directory, "{}.stdout.json".format(prefix))
    stderr_path = os.path.join(directory, "{}.stderr.txt".format(prefix))
    atomic_write_json(request_path, payload)
    env = os.environ.copy()
    if isinstance(config.get("env"), dict):
        env.update({str(k): str(v) for k, v in config["env"].items()})
    cwd = config.get("cwd")
    cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else None
    started = time.monotonic()
    with open(request_path, "rb") as stdin, open(stdout_path, "wb") as out, \
            open(stderr_path, "wb") as err:
        try:
            proc = subprocess.Popen(command, stdin=stdin, stdout=out,
                                    stderr=err, cwd=cwd, env=env,
                                    start_new_session=True)
        except OSError as exc:
            raise AdapterError("cannot start adapter: {}".format(exc), {
                "executablePath": resolve_executable(command)})
        identity = stable_process_identity(proc.pid)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise AdapterError("adapter timed out", {
                "timeoutSeconds": timeout, "process": identity})
    elapsed = (time.monotonic() - started) * 1000.0
    if code != 0:
        raise AdapterError("adapter exited with code {}".format(code), {
            "exitCode": code, "process": identity,
            "stderrPath": stderr_path})
    raw = _read_limited(stdout_path, limit)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AdapterError("adapter stdout is not one JSON value: {}"
                           .format(exc), {"process": identity})
    if not isinstance(value, dict):
        raise AdapterError("adapter response must be a JSON object")
    return value, {
        "requestPath": request_path,
        "rawResponsePath": stdout_path,
        "stderrPath": stderr_path,
        "process": identity,
        "endToEndMs": elapsed,
    }


def _deep_merge(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return right
    out = dict(left)
    for key, value in right.items():
        out[key] = _deep_merge(out[key], value) if key in out else value
    return out


def _http_json(config, envelope, directory):
    url = config.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise AdapterError("http adapter requires an http(s) url")
    body = _deep_merge(config.get("requestTemplate", {}),
                       envelope.get("request", {}))
    request_path = os.path.join(directory, "adapter.request.json")
    response_path = os.path.join(directory, "adapter.response.json")
    atomic_write_json(request_path, body)
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {str(k): str(v) for k, v in
               (config.get("headers") or {}).items()}
    for header, env_name in (config.get("headerEnv") or {}).items():
        if env_name not in os.environ:
            raise AdapterError("http adapter environment variable {} is "
                               "not set".format(env_name))
        headers[str(header)] = os.environ[env_name]
    headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=raw_body, headers=headers,
                                     method=config.get("method", "POST"))
    timeout = float(config.get("timeoutSeconds", 600))
    limit = int(config.get("maxResponseBytes", 16 * 1024 * 1024))
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            first_ms = (time.monotonic() - started) * 1000.0
            raw = response.read(limit + 1)
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read(limit + 1)
        with open(response_path, "wb") as handle:
            handle.write(raw)
        raise AdapterError("http adapter returned {}".format(exc.code), {
            "httpStatus": exc.code, "rawResponsePath": response_path})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AdapterError("http adapter failed: {}".format(exc))
    elapsed = (time.monotonic() - started) * 1000.0
    with open(response_path, "wb") as handle:
        handle.write(raw)
    if len(raw) > limit:
        raise AdapterError("http response exceeded {} bytes".format(limit), {
            "rawResponsePath": response_path})
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AdapterError("http response is not JSON: {}".format(exc), {
            "rawResponsePath": response_path})
    return value, {
        "requestPath": request_path,
        "rawResponsePath": response_path,
        "stderrPath": None,
        "process": None,
        "httpStatus": status,
        "adapterFirstChunkMs": first_ms,
        "endToEndMs": elapsed,
    }


def _extract_text(value):
    if isinstance(value.get("text"), str):
        return value["text"]
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(first.get("text"), str):
            return first["text"]
    message = value.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(value.get("response"), str):
        return value["response"]
    return ""


def _normalize_usage(value):
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    prompt = usage.get("promptTokens", usage.get("prompt_tokens",
                                                 value.get("prompt_eval_count")))
    completion = usage.get("completionTokens", usage.get(
        "completion_tokens", value.get("eval_count")))
    return {"promptTokens": prompt, "completionTokens": completion,
            "totalTokens": usage.get("totalTokens", usage.get(
                "total_tokens", (prompt + completion)
                if isinstance(prompt, (int, float)) and
                isinstance(completion, (int, float)) else None))}


def _normalize_timings(value, transport):
    timing = value.get("timings") if isinstance(value.get("timings"), dict) \
        else {}
    prompt_count = value.get("prompt_eval_count")
    eval_count = value.get("eval_count")
    prompt_ns = value.get("prompt_eval_duration")
    eval_ns = value.get("eval_duration")
    prefill = prompt_count * 1e9 / prompt_ns \
        if prompt_count and prompt_ns else None
    decode = eval_count * 1e9 / eval_ns if eval_count and eval_ns else None
    return {
        "routeAdmissionMs": timing.get("routeAdmissionMs"),
        "runtimeLoadMs": timing.get("runtimeLoadMs",
                                     value.get("load_duration", 0) / 1e6
                                     if value.get("load_duration") else None),
        "serverTtftMs": timing.get("serverTtftMs", timing.get("ttftMs")),
        "serverPrefillTokensPerSecond": timing.get(
            "serverPrefillTokensPerSecond", timing.get(
                "prefillTokensPerSecond", prefill)),
        "serverDecodeTokensPerSecond": timing.get(
            "serverDecodeTokensPerSecond", timing.get(
                "decodeTokensPerSecond", decode)),
        "serverTotalMs": timing.get("serverTotalMs",
                                    value.get("total_duration", 0) / 1e6
                                    if value.get("total_duration") else None),
        "adapterFirstChunkMs": timing.get(
            "adapterFirstChunkMs", transport.get("adapterFirstChunkMs")),
        "endToEndMs": transport.get("endToEndMs"),
    }


def call_adapter(config, envelope, directory):
    kind = config.get("type")
    if kind == "command":
        value, transport = command_json(config, envelope, directory)
    elif kind == "http":
        value, transport = _http_json(config, envelope, directory)
    else:
        raise AdapterError("adapter type must be command or http")
    runtime_ok = value.get("ok", True) is not False
    return {
        "runtimeOk": runtime_ok,
        "text": _extract_text(value),
        "usage": _normalize_usage(value),
        "timings": _normalize_timings(value, transport),
        "received": value.get("received")
        if isinstance(value.get("received"), dict) else {},
        "reportedQuality": value.get("quality")
        if isinstance(value.get("quality"), dict) else None,
        "wireEvidence": value.get("wireEvidence")
        if isinstance(value.get("wireEvidence"), dict) else None,
        "agentTrace": value.get("agentTrace")
        if isinstance(value.get("agentTrace"), dict) else None,
        "raw": value,
        "transport": transport,
    }
