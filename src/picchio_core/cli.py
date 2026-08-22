"""AI-first command surface: one JSON conclusion on stdout."""

import argparse
import json
import os
import sys

from . import AGENT_TRACE_SCHEMA, PARITY_SCHEMA, QUEUE_SCHEMA, RUN_SCHEMA
from .artifacts import ArtifactError
from .parity import run_parity
from .runner import ManifestError, run_queue


COMMAND_GROUPS = (
    ("measure", "Measure a run",
     ("diagnose", "run")),
    ("inspect", "Inspect the file",
     ("id", "plan")),
    ("watch", "Watch a live setup",
     ("guard", "watch", "monitor")),
    ("share", "Share or check evidence",
     ("share", "verify", "compare", "vet")),
)


# One catalog drives both human help and the machine contract. Adding a
# command in only one of those places made real capabilities invisible to
# agents while people could still find them by scrolling --help.
COMMAND_CAPABILITIES = {
    "diagnose": {
        "usage": "picchio [diagnose] TARGET [--json|--share row]",
        "purpose": "measure one GGUF, Ollama tag or llama-server URL",
        "stdout": "picchio.diagnose.v1 with --json",
        "evidence": "--keep-logs DIR",
        "aliases": ["picchio TARGET"],
        "options": ["--share line|row|post", "--ctx-sweep [LIST]",
                    "--explain TOKS"],
    },
    "run": {
        "usage": "picchio run MANIFEST [--artifact DIR]",
        "purpose": "run or resume a queue, parity job or agent trace",
        "stdout": "one final JSON",
        "manifests": [QUEUE_SCHEMA, PARITY_SCHEMA],
        "agentTrace": {
            "adapterResponse": "optional " + AGENT_TRACE_SCHEMA,
            "artifacts": "agent-trace.json + agent-trace.md + "
                         "per-round engine wire bodies",
        },
        "resume": "same command and artifact directory",
    },
    "id": {
        "usage": "picchio id MODEL",
        "purpose": "hash and inspect the exact GGUF or Ollama model",
    },
    "plan": {
        "usage": "picchio plan [MODEL]",
        "purpose": "estimate fit before loading; label speed estimates",
    },
    "guard": {
        "usage": "picchio guard [--keep-logs DIR] -- COMMAND",
        "purpose": "wrap a command and report placement without killing it",
    },
    "watch": {
        "usage": "picchio watch [PID|ollama] [--for SEC] [--json]",
        "purpose": "sample whole-GPU activity beside a process",
        "stdout": "picchio.watch.v1 with --json",
        "evidence": "watch.samples.jsonl + watch.summary.json",
    },
    "monitor": {
        "usage": "picchio monitor TARGET [--for SEC] [--json]",
        "purpose": "probe a running server and catch intermittent fallback",
        "stdout": "session JSON with --json",
        "evidence": "probe response files with --keep-logs DIR",
    },
    "share": {
        "usage": "picchio share [BLOCK] [--line|--row|--post]",
        "purpose": "reformat one receipt without measuring again",
    },
    "verify": {
        "usage": "picchio verify [FILE]",
        "purpose": "check whether a receipt contradicts itself",
    },
    "compare": {
        "usage": "picchio compare A.txt B.txt",
        "purpose": "name the first config difference, then compare rates",
    },
    "vet": {
        "usage": "picchio vet [FILE]",
        "purpose": "list which settings a benchmark post states or omits",
    },
    "capabilities": {
        "usage": "picchio capabilities --json",
        "purpose": "print this command, schema and exit-code contract",
        "stdout": "picchio.capabilities.v1",
    },
}


EXIT_CODES_BY_COMMAND = {
    "diagnose": {"0": "healthy or no placement evidence",
                 "2": "could not run", "3": "partial offload",
                 "4": "CPU fallback", "5": "conflicting evidence",
                 "7": "no timing evidence"},
    "watch": {"0": "GPU working", "2": "could not run",
              "4": "GPU idle"},
    "monitor": {"0": "engaged throughout", "2": "could not run",
                "4": "CPU fallback seen"},
    "verify": {"0": "self-consistent", "2": "unreadable",
               "5": "contradiction found"},
    "run": {"0": "completed", "2": "could not run or incomplete",
            "3": "runtime failure", "4": "quality failure",
            "5": "causal evidence conflict", "6": "safety stop",
            "130": "interrupted"},
    "guard": {"note": "passes through the wrapped command exit code"},
}


def command_help_epilog():
    return """commands:

  picchio TARGET
    measure a GGUF, Ollama tag or running llama-server
  picchio guard -- COMMAND
    warn when a command leaves model layers on the CPU
  picchio watch [PID|ollama]
    watch GPU activity beside a running model
  picchio monitor TARGET
    catch a server that drops out of its normal performance lane
  picchio compare A.txt B.txt
    show the first changed setting, then compare the rates

prefill is the model reading your prompt; decode is the model writing it."""


def capabilities_value(version, protocol):
    return {
        "schema": "picchio.capabilities.v1",
        "version": version,
        "protocol": protocol,
        "jobs": {
            key: {"title": title, "commands": list(names)}
            for key, title, names in COMMAND_GROUPS
        },
        "commands": COMMAND_CAPABILITIES,
        "exitCodes": {
            "0": "success; exact meaning is command-specific",
            "2": "could not run or unreadable input",
            "3": "partial offload (diagnose) or runtime failure (run)",
            "4": "CPU fallback/idle or run quality failure",
            "5": "conflicting, contradictory or non-causal evidence",
            "6": "run safety stop",
            "7": "diagnose measured no timing evidence",
            "130": "run interrupted",
        },
        "exitCodesByCommand": EXIT_CODES_BY_COMMAND,
    }


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
    print(json.dumps(capabilities_value(version, protocol), indent=1))
