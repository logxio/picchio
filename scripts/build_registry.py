#!/usr/bin/env python3
"""Build the public Picchio receipt registry from committed evidence."""

import argparse
import json
import os
import re
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

QUANT_CARDS = (
    ("unsloth.txt", "Qwen3.5-9B", "unsloth"),
    ("bartowski.txt", "Qwen3.5-9B", "bartowski"),
    ("lmstudio-community.txt", "Qwen3.5-9B", "lmstudio-community"),
    ("mradermacher.txt", "Qwen3.5-9B", "mradermacher"),
    ("lmstudio-community-4b.txt", "Qwen3.5-4B", "lmstudio-community"),
    ("unsloth-4b.txt", "Qwen3.5-4B", "unsloth"),
    ("bartowski-4b.txt", "Qwen3.5-4B", "bartowski"),
    ("unsloth-27b.txt", "Qwen3.8-27B", "unsloth"),
    ("lmstudio-community-27b.txt", "Qwen3.8-27B", "lmstudio-community"),
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _number(value):
    return int(value.replace(",", ""))


def benchmark_record(filename):
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
        "protocol": block["protocol"],
        "receipt": REPO_BLOB + "examples/{}".format(filename),
    }


def quant_record(filename, family, publisher):
    path = os.path.join(ROOT, "examples", "quantizers", filename)
    text = _read(path)
    flat = " ".join(line.strip() for line in text.splitlines())
    name = re.search(r"picchio id:\s+(\S+)", flat)
    identity = re.search(
        r"file\s+sha256\s+([0-9a-f]{12}),\s+([\d,]+) bytes", flat)
    claimed = re.search(r"claimed\s+(\S+)", flat)
    walked = re.search(r"walked\s+(\d+) tensors,\s+(\d+) types", flat)
    effective = re.search(
        r"effective\s+([\d.]+) bits per weight:\s+([\d,]+) tensor "
        r"bytes\s+over\s+([\d,]+) weights", flat)
    if not all((name, identity, claimed, walked, effective)):
        raise ValueError("{} is missing a registry field".format(filename))
    return {
        "id": os.path.splitext(filename)[0],
        "family": family,
        "publisher": publisher,
        "file": name.group(1),
        "bytes": _number(identity.group(2)),
        "claimed": claimed.group(1),
        "tensors": int(walked.group(1)),
        "types": int(walked.group(2)),
        "bitsPerWeight": float(effective.group(1)),
        "tensorBytes": _number(effective.group(2)),
        "weights": _number(effective.group(3)),
        "card": REPO_BLOB + "examples/quantizers/{}".format(filename),
    }


def registry():
    return {
        "schema": "picchio.receipts.v1",
        "benchmarks": [benchmark_record(name) for name in BENCHMARKS],
        "quantizers": [quant_record(*row) for row in QUANT_CARDS],
    }


def encoded():
    return json.dumps(registry(), indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        ROOT, "docs", "receipts.json"))
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
        print("registry current: {} benchmarks, {} quantizer cards".format(
            len(BENCHMARKS), len(QUANT_CARDS)))
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print(args.output)


if __name__ == "__main__":
    main()
