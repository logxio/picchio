"""Crash-safe, append-preserving evidence storage."""

import hashlib
import json
import os
import re
import time

from . import CHECKPOINT_SCHEMA


class ArtifactError(Exception):
    pass


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ArtifactError("cannot read JSON {}: {}".format(path, exc))


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path, payload):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temp = "{}.tmp.{}.{}".format(path, os.getpid(), time.time_ns())
    try:
        with open(temp, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(parent)
    except OSError as exc:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise ArtifactError("cannot atomically write {}: {}".format(path,
                                                                     exc))


def atomic_write_json(path, value):
    atomic_write_bytes(path, json.dumps(
        value, indent=1, ensure_ascii=False).encode("utf-8") + b"\n")


def append_jsonl(path, value):
    parent = os.path.dirname(path) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        with open(path, "ab", buffering=0) as handle:
            handle.write(canonical_bytes(value) + b"\n")
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ArtifactError("cannot append evidence {}: {}".format(path,
                                                                    exc))


def safe_component(value):
    raw = str(value)
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-.")[:48]
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return "{}-{}".format(clean or "case", suffix)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


class ArtifactStore:
    """One writer owns a run directory; completed receipts are immutable."""

    def __init__(self, root, manifest, tool):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.manifest = manifest
        self.manifest_sha = sha256_json(manifest)
        self.tool = tool
        self.lock_path = os.path.join(self.root, ".picchio.lock")
        self._closed = False
        try:
            os.makedirs(self.root, exist_ok=True)
        except OSError as exc:
            raise ArtifactError("cannot create artifact directory {}: {}"
                                .format(self.root, exc))
        self._acquire_lock()
        try:
            self._open_manifest()
            self.state = self._load_state()
            self._reconcile_receipts()
        except Exception:
            self.close()
            raise

    def _acquire_lock(self):
        payload = {"pid": os.getpid(), "createdAt": utc_now()}
        for _attempt in range(2):
            try:
                fd = os.open(self.lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                return
            except FileExistsError:
                try:
                    old = load_json(self.lock_path)
                except ArtifactError:
                    old = {}
                if _pid_alive(old.get("pid")):
                    raise ArtifactError(
                        "artifact is already active in pid {}".format(
                            old.get("pid")))
                try:
                    os.unlink(self.lock_path)
                except OSError as exc:
                    raise ArtifactError("cannot clear stale artifact lock: {}"
                                        .format(exc))
            except OSError as exc:
                raise ArtifactError("cannot lock artifact directory: {}"
                                    .format(exc))
        raise ArtifactError("cannot acquire artifact lock")

    def _open_manifest(self):
        path = os.path.join(self.root, "manifest.json")
        if os.path.exists(path):
            previous = load_json(path)
            if sha256_json(previous) != self.manifest_sha:
                raise ArtifactError(
                    "artifact manifest changed; resume requires the exact "
                    "same manifest")
        else:
            atomic_write_json(path, self.manifest)
            atomic_write_json(os.path.join(self.root, "run.json"), {
                "schema": "picchio.run-metadata.v1",
                "manifestSha256": self.manifest_sha,
                "createdAt": utc_now(),
                "tool": self.tool,
            })

    def _load_state(self):
        path = os.path.join(self.root, "checkpoint.json")
        if os.path.exists(path):
            state = load_json(path)
            if state.get("schema") != CHECKPOINT_SCHEMA \
                    or state.get("manifestSha256") != self.manifest_sha:
                raise ArtifactError("checkpoint does not match manifest")
            if not isinstance(state.get("completed"), dict):
                raise ArtifactError("checkpoint completed map is invalid")
            return state
        state = {
            "schema": CHECKPOINT_SCHEMA,
            "manifestSha256": self.manifest_sha,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "completed": {},
            "runtime": {},
            "loadCount": 0,
            "safetyAbort": None,
        }
        atomic_write_json(path, state)
        return state

    def _receipt_paths(self):
        cases = os.path.join(self.root, "cases")
        found = []
        if not os.path.isdir(cases):
            return found
        for case_name in sorted(os.listdir(cases)):
            case_dir = os.path.join(cases, case_name)
            if not os.path.isdir(case_dir):
                continue
            for attempt in sorted(os.listdir(case_dir)):
                path = os.path.join(case_dir, attempt, "result.json")
                if os.path.isfile(path):
                    found.append(path)
        return found

    def _reconcile_receipts(self):
        discovered = {}
        for path in self._receipt_paths():
            receipt = load_json(path)
            if receipt.get("manifestSha256") != self.manifest_sha:
                raise ArtifactError("foreign receipt in artifact: {}"
                                    .format(path))
            key = receipt.get("key")
            if not key:
                raise ArtifactError("receipt has no key: {}".format(path))
            rel = os.path.relpath(path, self.root)
            if key in discovered and discovered[key] != rel:
                raise ArtifactError("duplicate completed receipt for {}"
                                    .format(key))
            discovered[key] = rel
        changed = False
        for key, rel in discovered.items():
            current = self.state["completed"].get(key)
            if current and current != rel:
                raise ArtifactError("checkpoint receipt mismatch for {}"
                                    .format(key))
            if not current:
                self.state["completed"][key] = rel
                changed = True
        if changed:
            self.save_state()

    def save_state(self):
        self.state["updatedAt"] = utc_now()
        atomic_write_json(os.path.join(self.root, "checkpoint.json"),
                          self.state)

    def event(self, kind, **fields):
        row = {"schema": "picchio.event.v1", "at": utc_now(),
               "kind": kind}
        row.update(fields)
        append_jsonl(os.path.join(self.root, "events.jsonl"), row)

    def is_complete(self, key):
        return key in self.state["completed"]

    def completed_results(self):
        rows = []
        for key in sorted(self.state["completed"]):
            path = os.path.join(self.root, self.state["completed"][key])
            rows.append(load_json(path))
        return rows

    def new_attempt(self, key):
        base = os.path.join(self.root, "cases", safe_component(key))
        os.makedirs(base, exist_ok=True)
        index = 1
        while True:
            path = os.path.join(base, "attempt-{:04d}".format(index))
            try:
                os.mkdir(path)
                return path, index
            except FileExistsError:
                index += 1
            except OSError as exc:
                raise ArtifactError("cannot create case attempt: {}"
                                    .format(exc))

    def commit_result(self, key, attempt_dir, receipt):
        if self.is_complete(key):
            raise ArtifactError("case already completed: {}".format(key))
        path = os.path.join(attempt_dir, "result.json")
        atomic_write_json(path, receipt)
        rel = os.path.relpath(path, self.root)
        self.state["completed"][key] = rel
        self.save_state()
        self.event("case_completed", key=key, result=rel,
                   runtimeStatus=receipt.get("runtimeStatus"),
                   qualityStatus=receipt.get("qualityStatus"))

    def relative(self, path):
        return os.path.relpath(path, self.root) if path else None

    def write_summary(self, summary):
        atomic_write_json(os.path.join(self.root, "summary.json"), summary)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            os.unlink(self.lock_path)
            _fsync_dir(self.root)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
