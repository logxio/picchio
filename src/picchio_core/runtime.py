"""Launch or adopt a caller-declared runtime without guessing by argv text."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

from .artifacts import ArtifactError, utc_now
from .host import process_identity, same_process, stable_process_identity


class RuntimeError(Exception):
    pass


def _valid_command(command):
    return isinstance(command, list) and command and all(
        isinstance(part, str) and part for part in command)


def _wait_ready(handle, ready):
    if not ready:
        return
    url = ready.get("url")
    if not url:
        raise RuntimeError("runtime ready check requires url")
    timeout = float(ready.get("timeoutSeconds", 60))
    interval = float(ready.get("intervalSeconds", 0.25))
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if not same_process(handle["identity"]):
            raise RuntimeError("runtime exited before it became ready")
        try:
            with urllib.request.urlopen(url, timeout=min(2, timeout)) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(interval)
    raise RuntimeError("runtime did not become ready: {}".format(last))


def ensure_runtime(store, spec, lane="queue"):
    """Return an exact process identity. Picchio never signals it."""
    if not spec:
        return {"identity": None, "stderrPath": None,
                "launchSequence": None, "reused": False}
    previous = store.state.get("runtime", {}).get(lane)
    if previous and same_process(previous.get("identity")):
        return {"identity": previous.get("identity"),
                "stderrPath": previous.get("stderrPath"),
                "stdoutPath": previous.get("stdoutPath"),
                "launchSequence": previous.get("launchSequence"),
                "reused": True}
    pid = spec.get("pid")
    if pid is not None:
        identity = process_identity(pid)
        if not identity:
            raise RuntimeError("declared runtime pid {} is not alive".format(
                pid))
        handle = {"identity": identity, "stderrPath": spec.get("stderrPath"),
                  "stdoutPath": spec.get("stdoutPath"),
                  "launchSequence": spec.get("launchSequence"),
                  "reused": False}
        store.state.setdefault("runtime", {})[lane] = dict(handle)
        store.save_state()
        return handle
    command = spec.get("launchCommand")
    if not _valid_command(command):
        raise RuntimeError("runtime requires pid or launchCommand string list")
    sequence = int(store.state.get("loadCount", 0)) + 1
    runtime_dir = os.path.join(store.root, "runtime", lane)
    os.makedirs(runtime_dir, exist_ok=True)
    stdout_path = os.path.join(
        runtime_dir, "launch-{:04d}.stdout.txt".format(sequence))
    stderr_path = os.path.join(
        runtime_dir, "launch-{:04d}.stderr.txt".format(sequence))
    env = os.environ.copy()
    if isinstance(spec.get("env"), dict):
        env.update({str(k): str(v) for k, v in spec["env"].items()})
    cwd = spec.get("cwd")
    cwd = os.path.abspath(os.path.expanduser(cwd)) if cwd else None
    try:
        with open(stdout_path, "ab", buffering=0) as stdout, \
                open(stderr_path, "ab", buffering=0) as stderr:
            proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, cwd=cwd, env=env, start_new_session=True)
    except OSError as exc:
        raise RuntimeError("cannot launch runtime: {}".format(exc))
    identity = stable_process_identity(proc.pid)
    if not identity:
        raise RuntimeError("runtime exited during launch")
    handle = {
        "identity": identity,
        "stdoutPath": store.relative(stdout_path),
        "stderrPath": store.relative(stderr_path),
        "launchSequence": sequence,
        "launchedAt": utc_now(),
        "reused": False,
    }
    store.state.setdefault("runtime", {})[lane] = dict(handle)
    store.state["loadCount"] = sequence
    store.save_state()
    store.event("runtime_launched", lane=lane, identity=identity,
                launchSequence=sequence)
    try:
        _wait_ready(handle, spec.get("ready"))
    except Exception:
        store.event("runtime_not_ready", lane=lane, identity=identity)
        raise
    return handle


def runtime_alive(handle):
    identity = handle.get("identity") if handle else None
    return True if not identity else same_process(identity)
