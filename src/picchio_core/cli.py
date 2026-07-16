"""AI-first command surface: one JSON conclusion on stdout."""

import argparse
import json
import os
import sys

from . import AGENT_TRACE_SCHEMA, PARITY_SCHEMA, QUEUE_SCHEMA, RUN_SCHEMA
from .artifacts import ArtifactError
from .parity import run_parity
from .runner import ManifestError, run_queue


def _load_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ManifestError("cannot read manifest {}: {}".format(path, exc))
    if not isinstance(value, dict):
        raise ManifestError("manifest must be a JSON object")
    return value


def _default_artifact(path):
    absolute = os.path.abspath(os.path.expanduser(path))
    stem, extension = os.path.splitext(absolute)
    return stem + ".picchio-run" if extension.lower() == ".json" \
        else absolute + ".picchio-run"


def _progress(message):
    sys.stderr.write("picchio: {}\n".format(message))
    sys.stderr.flush()


def run_cli(argv, hooks):
    parser = argparse.ArgumentParser(
        prog="picchio run",
        description="run a queue or parity manifest, including optional "
                    "multi-round agent traces; stdout is one final JSON "
                    "result and the artifact directory is resumable")
    parser.add_argument("manifest", help="queue/parity JSON manifest")
    parser.add_argument("--artifact", metavar="DIR",
                        help="evidence directory; default MANIFEST.picchio-run")
    args = parser.parse_args(argv)
    artifact = os.path.abspath(os.path.expanduser(
        args.artifact or _default_artifact(args.manifest)))
    try:
        manifest = _load_manifest(args.manifest)
        schema = manifest.get("schema")
        if schema == QUEUE_SCHEMA:
            summary = run_queue(manifest, artifact, hooks, _progress)
        elif schema == PARITY_SCHEMA:
            summary = run_parity(manifest, artifact, hooks, _progress)
        else:
            raise ManifestError("unsupported manifest schema {!r}".format(
                schema))
    except (ManifestError, ArtifactError, OSError, TypeError, ValueError) as exc:
        summary = {
            "schema": RUN_SCHEMA,
            "status": "error",
            "verdict": "COULD NOT RUN",
            "exitCode": 2,
            "artifactDirectory": artifact,
            "error": str(exc),
        }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    raise SystemExit(summary.get("exitCode", 2))


def capabilities_cli(argv, version, protocol):
    if argv not in ([], ["--json"]):
        raise SystemExit("picchio capabilities: unexpected arguments")
    value = {
        "schema": "picchio.capabilities.v1",
        "version": version,
        "protocol": protocol,
        "commands": {
            "diagnose": {"stdout": "JSON with --json",
                         "evidence": "--keep-logs DIR"},
            "watch": {"stdout": "picchio.watch.v1 with --json",
                      "evidence": "watch.samples.jsonl + watch.summary.json"},
            "run": {"stdout": "one final JSON",
                    "manifests": [QUEUE_SCHEMA, PARITY_SCHEMA],
                    "agentTrace": {
                        "adapterResponse": "optional " + AGENT_TRACE_SCHEMA,
                        "artifacts": "agent-trace.json + agent-trace.md + "
                                     "per-round engine wire bodies",
                    },
                    "resume": "same command and artifact directory"},
        },
        "exitCodes": {"0": "pass/completed", "2": "could not run",
                      "3": "runtime failure", "4": "quality failure",
                      "5": "conflicting evidence", "6": "safety stop",
                      "130": "interrupted"},
    }
    print(json.dumps(value, indent=1))
