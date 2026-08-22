#!/usr/bin/env python3
"""Build the public Picchio results table."""

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REPO_BLOB = "https://github.com/logxio/picchio/blob/main/"

from picchio import parse_block  # noqa: E402


BENCHMARKS = (
    "healthy-metal.txt",
    "cpu-fallback.txt",
    "ollama-qwen35.txt",
    "server-endpoint.txt",
    "linux-4090.txt",
    "linux-5090-cuda.txt",
    "linux-5090-vulkan-nonce.txt",
    "linux-5090-ollama-nonce.txt",
    "linux-5090-27b.txt",
    "ollama-35b.txt",
)

def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def benchmark_result(filename):
    path = os.path.join(ROOT, "examples", filename)
    block = parse_block(_read(path))
    if block is None:
        raise ValueError("{} has no readable verdict block".format(filename))
    prefill, decode, wallclock = block["rates"]
    return {
        "id": os.path.splitext(filename)[0],
        "model": block["model"],
        "quant": block["quant"],
        "engine": block["engine"],
        "machine": "{}, {} GB".format(block["chip"], block["ram"]),
        "placement": block["place"],
        "placementFraction": block["frac"],
        "ctx": block["ctx"],
        "prefill": prefill,
        "decode": decode,
        "wallclock": wallclock,
        "gpuMemoryGiB": block["os_mem"],
        "watts": block["os_watts"],
        "joulesPerToken": block["os_joules"],
        "verdict": block["verdict"],
        "result": REPO_BLOB + "examples/{}".format(filename),
    }


def registry():
    return {
        "schema": "picchio.results.v1",
        "benchmarks": [benchmark_result(name) for name in BENCHMARKS],
    }


def encoded():
    return json.dumps(registry(), indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        ROOT, "docs", "results.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = encoded()
    if args.check:
        try:
            current = _read(args.output)
        except OSError as exc:
            raise SystemExit("registry missing: {}".format(exc))
        if current != payload:
            raise SystemExit("registry is stale; run scripts/build_registry.py")
        print("results current: {} runs".format(len(BENCHMARKS)))
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(args.output)


if __name__ == "__main__":
    main()
