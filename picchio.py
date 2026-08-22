#!/usr/bin/env python3
"""Picchio local LLM measurement and evidence CLI.

The current human and machine command contracts come from
``picchio --help`` and ``picchio capabilities --json``.
"""

import argparse
import ctypes
import glob
import hashlib
import io
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# Clone checkout: modular sources live under src/. In the zipapp the same
# package sits at archive root and zipimport resolves it without this path.
_SOURCE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_SOURCE_ROOT) and _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from picchio_core import gpu_meters  # noqa: E402  (needs the path above)
# share's output shapes and the whole of vet live in the module layer:
# both are new capability, and this entry point is already long enough
# that new capability landing in it is a rule, not a preference. What
# stays here is the wiring that reaches the gguf walk and the cache.
from picchio_core.share import (  # noqa: E402
    SHARE_COLUMNS, SHARE_URL, render_share, share_line, share_missing,
    share_post, share_row)
from picchio_core.vet import (  # noqa: E402
    VET_NOTES, vet_cli, vet_quant_note, vet_rate_lane, vet_scan)

VERSION = "0.1.0"
# Measurement protocol tag, printed in the block footer. If the prompt
# size, generation length, pass structure or aggregation ever change,
# this bumps, so numbers from different protocols never get compared as
# if they were one series.
PROTOCOL = "mp1"
WIDTH = 66
# Lines in the verdict block, counted and enforced by the renderer, not
# hoped for. It was 15 until the settings line landed; a block carrying
# a WHY line was already spending every one of those, and the verdict
# paragraph cannot shrink below one line, so the disclosure had to be
# paid for in budget rather than out of the paragraph. Still one
# screen, still one forum comment. Adding a line means raising this
# number on purpose, which is the point of it being a number.
HEIGHT = 16
# The exit code contract, in one place because it is a public one: the
# global picchio skill and every script that shells out reads these.
# 0 also covers the states where picchio cannot prove placement but did
# measure rates; 7 is the one where it measured nothing at all. 6 is
# taken by run's safety stop and the two families share one table.
EXIT_CODES = {"HEALTHY": 0, "NO PLACEMENT EVIDENCE": 0,
              "PARTIAL OFFLOAD": 3, "SILENT CPU FALLBACK": 4,
              "CONFLICTING EVIDENCE": 5, "NO TIMING EVIDENCE": 7}
N_PREDICT = 128
CTX = 4096
CACHE_PATH = os.path.expanduser("~/.cache/picchio/last.json")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
KV_TYPES_RE = re.compile(r"llama_kv_cache: size =.*"
                         r"K \((\S+)\):.*V \((\S+)\):")

# A fixed prompt of roughly 730 tokens. Short prompts lie: 7 prompt
# tokens measured 36 tok/s of apparent prefill on the same setup where
# 730 tokens measured about 590, because per call overhead dominates
# below a few hundred tokens. 128 generated tokens because decode
# settles within the first few dozen and 128 gives the median room
# without stretching the run.
_PARA = (
    "A benchmark number without its measurement conditions is a rumor "
    "with digits in it. Tokens per second can describe how fast a model "
    "reads a prompt, how fast it writes an answer, or how long the whole "
    "exchange took including loading the weights from disk. These three "
    "rates differ by an order of magnitude on the same machine in the "
    "same minute, and none of them is wrong. What is wrong is quoting "
    "one of them without saying which one it is. "
)
BENCH_PROMPT = "".join(
    "Consider case number {}: {}".format(i + 1, _PARA) for i in range(8)
)


def bench_prompt(run_id, i):
    """The fixed prompt behind a per-pass nonce.

    An engine that keeps a prefix cache will serve the second pass from
    it and still report the full prompt token count: measured on ollama
    0.32.15, three passes reported 770 prompt tokens each and took
    18.37 s, 36.7 ms and 33.6 ms. The 36 ms readings are a cache lookup
    wearing a prefill number's clothes.

    A prefix nobody has sent before defeats that, and it goes at the
    front because a cache matches on the leading tokens. The body is
    untouched and the nonce is the same shape every pass, so the passes
    stay comparable to each other and the rates stay comparable to
    every mp1 number ever published: five tokens onto seven hundred
    moves a rate by less than the warm span already does."""
    return prompt_nonce(run_id, i) + BENCH_PROMPT


def prompt_nonce(run_id, i):
    """The prefix one pass puts in front of the fixed prompt. Written
    into that pass's kept meta, so the evidence that no two passes
    shared a prefix is in the artifact rather than in a promise."""
    return "Run {} pass {:02d}. ".format(run_id, i + 1)


# ----------------------------------------------------------------- machine

def _cmd_out(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception:
        return ""


def machine_info():
    info = {"os": "", "chip": "", "ram_gb": None}
    sysname = platform.system()
    if sysname == "Darwin":
        info["chip"] = _cmd_out(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _cmd_out(["sysctl", "-n", "hw.memsize"])
        if mem.isdigit():
            info["ram_gb"] = round(int(mem) / (1024 ** 3))
        info["os"] = "macOS " + platform.mac_ver()[0]
    elif sysname == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2))
                        break
        except OSError:
            pass
        info["os"] = "Linux " + platform.release()
        try:
            # the gpu belongs in the machine fingerprint on linux; the
            # nvml name (display form) rides the chip field so every
            # footer and cache entry carries it without a new column
            gpu = gpu_meters.machine_gpu_name()
            if gpu:
                info["chip"] = "{} + {}".format(info["chip"], gpu) \
                    if info["chip"] else gpu
        except Exception:
            pass
    else:
        info["os"] = sysname
    if not info["chip"]:
        info["chip"] = platform.machine() or "unknown cpu"
    return info


def blank_pass():
    return {
        "wall_s": None,
        "load_ms": None,
        "prompt_ms": None, "prompt_tokens": None,
        "eval_ms": None, "eval_tokens": None,
        "offload_n": None, "offload_total": None,
        "gpu_device": None, "gpu_kind": None,
        "model_params": None, "model_size": None, "model_bytes": None,
        "threads": None, "cores": None,
        "vram_frac": None, "n_expert": None,
        "kv_types": None, "kv_source": None, "tensor_types": None,
        "free_mib": None, "fit_seen": False, "init_fail": None,
        "prefill_toks": None, "decode_toks": None, "wallclock_toks": None,
        "sampling": None, "nonce": None,
    }


def finish_rates(d):
    if d["prompt_ms"] and d["prompt_tokens"]:
        d["prefill_toks"] = d["prompt_tokens"] / (d["prompt_ms"] / 1000.0)
    if d["eval_ms"] and d["eval_tokens"]:
        d["decode_toks"] = d["eval_tokens"] / (d["eval_ms"] / 1000.0)
    if d["eval_tokens"] and d["wall_s"]:
        d["wallclock_toks"] = d["eval_tokens"] / d["wall_s"]
    return d


def _num(tok):
    """A number out of engine output, dot or comma decimal. llama.cpp
    reaches setlocale(LC_ALL, "") through console::init before it prints
    its timings, so on a de_DE or fr_FR box every "%.2f" field arrives as
    "28,53" and a dot-only parser reads nothing at all (issue #1: three
    passes, every lane n/a, and a verdict printed anyway). Our own runs
    now pin LC_NUMERIC=C, but logs captured on someone else's machine
    still land here that way. The rightmost separator is the decimal
    point and an earlier one is a thousands group; llama.cpp prints these
    with a plain %f and never groups, so this only ever undoes a locale."""
    tok = (tok or "").strip()
    cut = max(tok.rfind("."), tok.rfind(","))
    if cut >= 0:
        tok = tok[:cut].replace(".", "").replace(",", "") \
            + "." + tok[cut + 1:]
    try:
        return float(tok)
    except ValueError:
        return None


def _vendorless(name):
    """Display form of a GPU name: the vendor word drops so the gpu line
    holds the 66 column budget ("ROCm: AMD Radeon RX 7900 XTX" is one
    column over on its own). The raw name stays in the kept log."""
    return re.sub(r"^(NVIDIA|AMD)\s+", "", (name or "").strip())


def _dotted(s):
    """An engine field rewritten in the decimal style the rest of the
    block uses. Text that is already dot formatted passes through
    untouched, so every committed fixture renders exactly as before."""
    if not s or "," not in s:
        return s
    m = re.match(r"([\d.,]+)(.*)", s, re.S)
    n = _num(m.group(1)) if m else None
    if n is None:
        return s
    return "{:.2f}".format(n).rstrip("0").rstrip(".") + m.group(2)


def size_bytes(s):
    """'5.28 GiB' -> bytes, None when the unit is unfamiliar."""
    m = re.match(r"([\d.,]+)\s*([KMG]i?B|B)", s or "", re.I)
    if not m:
        return None
    n = _num(m.group(1))
    if n is None:
        return None
    mult = {"b": 1, "kib": 1024, "kb": 1000, "mib": 1024 ** 2,
            "mb": 1000 ** 2, "gib": 1024 ** 3, "gb": 1000 ** 3}
    return int(n * mult[m.group(2).lower()])


def keep_log(path, text):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    except OSError as e:
        sys.stderr.write("picchio: could not write {}: {}\n".format(path, e))


# ------------------------------------------------------- engine: llama.cpp

def find_binary(explicit):
    if explicit:
        if shutil.which(explicit) or os.path.isfile(explicit):
            if os.path.basename(explicit).startswith("llama-cli") \
                    and shutil.which("llama-completion"):
                sys.stderr.write(
                    "picchio: note: --bin points at the interactive "
                    "llama-cli; llama-completion is the one-shot binary and "
                    "is on your PATH (the runaway guard is on either way).\n")
            return explicit
        sys.exit("picchio: engine binary not found: {}".format(explicit))
    # llama-completion is the one-shot binary on current llama.cpp builds;
    # older builds did the same job with llama-cli -no-cnv.
    for name in ("llama-completion", "llama-cli"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit(
        "picchio: could not find llama-completion or llama-cli on PATH.\n"
        "Install llama.cpp (e.g. brew install llama.cpp) or pass --bin."
    )


def parse_engine_version(out):
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", out)
    if m:
        return "b" + m.group(1)
    # tarball builds carry no git hash; take a bare version number when
    # --version still prints one. llama.cpp prints version: 0 (unknown)
    # on such builds, and that 0 is a sentinel, not a version: say
    # version unknown instead of dressing the sentinel up as build b0.
    m = re.search(r"version:\s*(\d\S*)", out)
    if m and m.group(1) != "0":
        return "b" + m.group(1).lstrip("b")
    return "(version unknown)"


def engine_version(binpath):
    return parse_engine_version(_cmd_out([binpath, "--version"]))


RUN_CAP = 32 * 1024 * 1024  # hard cap on one engine run's captured output.
# A modern llama-cli told a flag it does not know (-no-cnv) drops into
# conversation mode and, under EOF, prints forever: measured here at tens
# of millions of tokens in 30 s, gigabytes buffered by capture_output. The
# cap kills that in a few seconds; the 30 minute timeout and an unbounded
# capture never would. Well above any real run's output (verbose stderr is
# a few MB at most), so only a runaway ever trips it.


def _run_capped(args, timeout, cap, env=None):
    """subprocess.run(capture_output=True) with a hard byte cap: the child
    is killed the instant its captured output passes `cap`, so a runaway
    interactive binary cannot exhaust memory or hang for the full timeout.
    Returns an object with .returncode, .stdout, .stderr, .capped and
    .timedout. Both pipes are drained by threads so neither can deadlock on
    a full OS buffer. env replaces the child's environment when given."""
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, errors="replace", env=env)
    buf = {"out": [], "err": [], "n": 0, "capped": False}
    lock = threading.Lock()

    def pump(pipe, key):
        try:
            while True:
                chunk = pipe.read(65536)
                if not chunk:
                    return
                with lock:
                    if buf["n"] >= cap:
                        if not buf["capped"]:
                            buf["capped"] = True
                            proc.kill()
                        return
                    buf[key].append(chunk)
                    buf["n"] += len(chunk)
        except (ValueError, OSError):
            pass

    threads = [threading.Thread(target=pump, args=(proc.stdout, "out"),
                                daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, "err"),
                                daemon=True)]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for t in threads:
        t.join(timeout=2)

    class _R:
        pass
    r = _R()
    r.returncode = proc.returncode
    r.stdout = "".join(buf["out"])
    r.stderr = "".join(buf["err"])
    r.capped = buf["capped"]
    r.timedout = timed_out
    return r


def engine_env():
    """The engine's environment with the decimal point pinned. llama.cpp
    calls setlocale(LC_ALL, "") in common/console.cpp before it prints
    the timings picchio measures with, so on a comma-decimal locale the
    whole lane table comes back empty. LC_NUMERIC is the surgical pin: a
    blanket LC_ALL=C would also reset LC_CTYPE and change how the engine
    handles the UTF-8 it generates. LC_ALL has to go with it because it
    outranks every LC_* category."""
    env = dict(os.environ)
    env.pop("LC_ALL", None)
    env["LC_NUMERIC"] = "C"
    return env


def run_llama_pass(binpath, model, extra_args, log_path=None,
                   prompt=BENCH_PROMPT, ctx=CTX):
    base = [
        binpath,
        "-m", model,
        "-p", prompt,
        "-n", str(N_PREDICT),
        "-c", str(ctx),
        "--seed", "7",
        "--ignore-eos",
    ]
    # Newest flags first; older builds reject flags they predate, so on
    # failure retry with a smaller flag set before giving up.
    attempts = [
        base + ["-no-cnv", "--verbose"],
        base + ["-no-cnv"],
        base,
    ]
    last = None
    for args in attempts:
        t0 = time.monotonic()
        r = _run_capped(args + extra_args, 1800, RUN_CAP,
                        env=engine_env())
        wall_s = time.monotonic() - t0
        if r.capped:
            # a flood, not a slow run: the binary went interactive and
            # printed without end. Retrying with fewer flags cannot make it
            # one-shot, so stop now and point at the binary that is.
            hint = ""
            if os.path.basename(binpath).startswith("llama-cli") \
                    and shutil.which("llama-completion"):
                hint = ("\nThis is the interactive llama-cli; rerun with the "
                        "one-shot binary:  --bin llama-completion")
            sys.exit(
                "picchio: {} produced runaway output (over {} MB) and never "
                "returned, so picchio stopped it. A modern llama-cli ignores "
                "-no-cnv and drops into conversation mode under EOF.{}".format(
                    os.path.basename(binpath), RUN_CAP // (1024 * 1024), hint))
        if r.timedout:
            sys.exit("picchio: engine run exceeded 30 minutes, giving up.")
        if r.returncode == 0:
            keep_log(log_path, r.stderr)
            return parse_stderr(r.stderr, wall_s)
        last = r
    tail = "\n".join(last.stderr.strip().splitlines()[-6:]) if last else ""
    sys.exit(
        "picchio: engine exited with code {}.\nLast lines:\n{}".format(
            last.returncode if last else "?", tail
        )
    )


def parse_stderr(text, wall_s):
    d = blank_pass()
    d["wall_s"] = wall_s
    re_load = re.compile(r"load time\s*=\s*([\d.,]+)\s*ms")
    re_pair = re.compile(
        r"=\s*([\d.,]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)")
    re_off = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
    re_metal = re.compile(r"ggml_metal_init: found device:\s*(.+)")
    re_cuda = re.compile(r"Device\s+\d+:\s*([^,]+),")
    # e.g. "using device CUDA0 (NVIDIA GeForce RTX 4090) (0000:61:00.0)
    # - 23818 MiB free": on b9430 CUDA builds this is the only line
    # naming the device (no ggml_cuda_init, no "Device 0:" lines exist
    # there, verified on the 4090 fixtures). llama.cpp puts the backend
    # in the device id it picks, so the same line answers for Vulkan0
    # and SYCL0 too; reading the backend out of it is one rule instead
    # of a per backend banner string to remember, and the Vulkan build
    # measured here prints no ggml_vulkan banner at all.
    re_dev_used = re.compile(r"using device ([A-Za-z]+)\d+ \(([^)]+)\)")
    re_params = re.compile(r"model params\s*=\s*([\d.,]+\s*\S?)")
    re_size = re.compile(r"file size\s*=\s*([\d.,]+\s*\S+)")
    re_threads = re.compile(r"n_threads\s*=\s*(\d+).*?/\s*(\d+)")
    # e.g. "using device MTL0 (Apple M5) (unknown id) - 25558 MiB free":
    # the free figure the engine itself saw, kept for WHY attribution.
    re_free = re.compile(r"-\s*(\d+)\s*MiB free")
    # e.g. "llama_kv_cache: size = 4352.00 MiB (..., 1/1 seqs),
    # K (q8_0): 2176.00 MiB, V (q8_0): 2176.00 MiB": the runtime kv
    # dtype, measured here on b9430 with -ctk q8_0 -ctv q8_0 (the f16
    # default is in every committed fixture). Cached so the id card
    # can cite a dtype this machine has actually run.
    # e.g. "llama_model_loader: - type q4_K:  132 tensors": the
    # loader's own per-type census, the engine side of the id cross
    # check against the gguf table walk
    re_ttype = re.compile(r"- type\s+(\S+):\s+(\d+) tensors")
    # e.g. "sampler seed: 7" and, inside the "sampler params:" block,
    # "top_k = 40, top_p = 0.950, min_p = 0.050, ..., temp = 0.800".
    # The engine already prints the settings it is about to sample
    # with; the block was simply not repeating them. Anchored on the
    # "= value" form so the sampler chain line ("-> temp-ext ->") and
    # the wider names around them (dynatemp_range, typical_p) cannot
    # be mistaken for the four fields read here.
    re_seed = re.compile(r"sampler seed:\s*(\d+)")
    # the value group has to end on a digit: these settings are printed
    # comma separated ("top_p = 0.950, min_p = 0.050"), and a group that
    # may end on a separator swallows the list comma, which _num then
    # reads as this locale's decimal point and returns 950 for 0.95
    re_samp = re.compile(r"\b(top_k|top_p|min_p|temp)\s*=\s*(-?[\d.,]*\d)")

    for line in text.splitlines():
        if "prompt eval time" in line:
            m = re_pair.search(line)
            if m:
                d["prompt_ms"] = _num(m.group(1))
                d["prompt_tokens"] = int(m.group(2))
        elif "eval time" in line:
            m = re_pair.search(line)
            if m:
                d["eval_ms"] = _num(m.group(1))
                d["eval_tokens"] = int(m.group(2))
        elif "load time" in line:
            m = re_load.search(line)
            if m:
                d["load_ms"] = _num(m.group(1))
        m = re_off.search(line)
        if m:
            d["offload_n"] = int(m.group(1))
            d["offload_total"] = int(m.group(2))
        m = re_metal.search(line)
        if m:
            d["gpu_device"] = m.group(1).strip()
            d["gpu_kind"] = "Metal"
        m = re_dev_used.search(line)
        if m and not d["gpu_device"]:
            # never overwrite a kind the log already stated: a HIP build
            # says "ROCm devices" further up and then still calls its
            # device CUDA0, and the log's own word wins
            d["gpu_kind"] = d["gpu_kind"] or m.group(1)
            d["gpu_device"] = _vendorless(m.group(2))
        if "ROCm devices" in line:
            # a HIP build compiles the cuda backend and announces itself
            # as "ggml_cuda_init: found 3 ROCm devices" (GGML_CUDA_NAME,
            # ggml/src/ggml-cuda/ggml-cuda.cu). The log says ROCm, so
            # the block says ROCm; calling an AMD card CUDA would be the
            # one line in it that a reader knows is wrong.
            d["gpu_kind"] = "ROCm"
        elif "ggml_cuda_init" in line or "CUDA devices" in line:
            d["gpu_kind"] = d["gpu_kind"] or "CUDA"
        m = re_cuda.search(line)
        if m and d["gpu_kind"] in ("CUDA", "ROCm") and not d["gpu_device"]:
            d["gpu_device"] = _vendorless(m.group(1))
        if "ggml_vulkan" in line.lower() and not d["gpu_kind"]:
            d["gpu_kind"] = "Vulkan"
        m = re_params.search(line)
        if m:
            d["model_params"] = _dotted(m.group(1).strip())
        m = re_size.search(line)
        if m:
            d["model_size"] = _dotted(m.group(1).strip())
            d["model_bytes"] = size_bytes(d["model_size"])
        if "system_info" in line:
            m = re_threads.search(line)
            if m:
                # llama.cpp defaults to 4 threads on this 10 core test
                # machine; recorded rather than tuned, because CPU rates
                # move a lot with -t and the block should say so.
                d["threads"] = int(m.group(1))
                d["cores"] = int(m.group(2))
        m = re_free.search(line)
        if m:
            d["free_mib"] = int(m.group(1))
        m = KV_TYPES_RE.search(line)
        if m:
            d["kv_types"] = [m.group(1), m.group(2)]
            d["kv_source"] = "llama.cpp stderr"
        m = re_ttype.search(line)
        if m:
            # the loader prints the census once per load and loads
            # twice per pass; identical values, overwrite is idempotent
            d["tensor_types"] = d["tensor_types"] or {}
            d["tensor_types"][m.group(1)] = int(m.group(2))
        m = re_seed.search(line)
        if m:
            d["sampling"] = d["sampling"] or {}
            d["sampling"]["seed"] = int(m.group(1))
        for key, val in re_samp.findall(line):
            n = _num(val)
            if n is not None:
                # the loader prints the block once per load and loads
                # twice per pass; identical values, overwrite is
                # idempotent, same as the tensor census above
                d["sampling"] = d["sampling"] or {}
                d["sampling"][key] = n
        m = re.search(r"n_expert\s+=\s*(\d+)", line)
        if m:
            # 0 on a dense model; the cache keeps this so plan knows a
            # mixture of experts cannot calibrate bandwidth arithmetic
            d["n_expert"] = int(m.group(1))
        if "common_params_fit_impl" in line:
            d["fit_seen"] = True
        low = line.lower()
        if (d["init_fail"] is None
                and ("error" in low or "failed" in low)
                and ("ggml_metal" in low or "ggml_cuda" in low
                     or "ggml_vulkan" in low or "ggml_backend" in low)):
            # first backend init failure line, verbatim minus the
            # "0.00.061.339 I " style log prefix some builds prepend
            d["init_fail"] = re.sub(r"^[\d.]+\s+[A-Z]\s+", "",
                                    line.strip())
    return finish_rates(d)


# ---------------------------------------------------------- engine: ollama

def ollama_api(path, payload=None, timeout=1800):
    url = "http://{}{}".format(OLLAMA_HOST, path)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def ollama_reachable():
    try:
        return ollama_api("/api/version", timeout=3).get("version", "?")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def ollama_has_model(tag):
    try:
        ollama_api("/api/show", {"model": tag}, timeout=15)
        return True
    except urllib.error.HTTPError:
        return False


def ollama_ps_entry(tag):
    try:
        for m in ollama_api("/api/ps", timeout=15).get("models", []):
            if m.get("name") == tag or m.get("model") == tag:
                return m
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return None


def ollama_host_is_local():
    """Runner logs only belong to this machine. Never inspect local logs
    for a remote OLLAMA_HOST and accidentally attribute their dtype to the
    remote server."""
    try:
        host = urllib.parse.urlsplit("//" + OLLAMA_HOST).hostname
    except ValueError:
        return False
    return host in ("127.0.0.1", "0.0.0.0", "localhost", "::1", "::")


def ollama_log_paths():
    """Readable Ollama server logs, including stdout/stderr redirects.

    The packaged app normally writes under ~/.ollama/logs. A terminal or
    service manager may redirect `ollama serve` anywhere, so on local hosts
    its live fd 1/2 targets are the stronger discovery path. Failure to find
    either is an evidence gap, never permission to assume the configured
    default dtype.
    """
    if not ollama_host_is_local():
        return []
    paths = []
    for pat in (os.path.expanduser("~/.ollama/logs/*.log"),
                os.path.expanduser("~/Library/Logs/Ollama/*.log")):
        paths.extend(glob.glob(pat))
    ps = _cmd_out(["ps", "-axo", "pid=,command="])
    pids = []
    for line in ps.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and re.search(
                r"(?:^|/)ollama\s+serve(?:\s|$)", fields[1]):
            pids.append(fields[0])
    if sys.platform.startswith("linux"):
        for pid in pids:
            for fd in ("1", "2"):
                try:
                    paths.append(os.path.realpath("/proc/{}/fd/{}".format(
                        pid, fd)))
                except OSError:
                    pass
    if shutil.which("lsof"):
        for pid in pids:
            for line in _cmd_out(
                    ["lsof", "-a", "-p", pid, "-d", "1,2", "-Fn"]
                    ).splitlines():
                if line.startswith("n/"):
                    paths.append(line[1:])
    out = []
    for path in paths:
        try:
            if path not in out and os.path.isfile(path) \
                    and os.access(path, os.R_OK):
                out.append(path)
        except OSError:
            pass
    return out


def ollama_log_snapshot():
    """(device, inode, byte offset) per current log, keyed by path."""
    snap = {}
    for path in ollama_log_paths():
        try:
            st = os.stat(path)
            snap[path] = (st.st_dev, st.st_ino, st.st_size)
        except OSError:
            pass
    return snap


def parse_kv_types(text):
    """Return the last concrete K/V dtype marker in a runtime log."""
    found = KV_TYPES_RE.findall(text)
    return list(found[-1]) if found else None


def ollama_kv_since(snap):
    """Read only bytes appended after a request began, capped at 4 MiB.

    A rotated file starts at zero. The narrow request window is important:
    Ollama's dtype is global configuration but the evidence cited by `id`
    is the runner line emitted for the model load picchio just caused.
    """
    best = None
    for path in ollama_log_paths():
        try:
            st = os.stat(path)
            old = snap.get(path)
            offset = old[2] if old and old[:2] == (st.st_dev, st.st_ino) \
                and st.st_size >= old[2] else 0
            if st.st_size <= offset:
                continue
            with open(path, "rb") as f:
                f.seek(offset)
                text = f.read(4 * 1024 * 1024).decode(
                    "utf-8", errors="replace")
            kt = parse_kv_types(text)
            if kt:
                best = kt
        except OSError:
            pass
    return best


def map_ollama(resp, wall_s, ps):
    d = blank_pass()
    d["wall_s"] = wall_s
    ns = 1e6  # ns -> ms
    if resp.get("load_duration"):
        d["load_ms"] = resp["load_duration"] / ns
    if resp.get("prompt_eval_duration") and resp.get("prompt_eval_count"):
        d["prompt_ms"] = resp["prompt_eval_duration"] / ns
        d["prompt_tokens"] = resp["prompt_eval_count"]
    if resp.get("eval_duration") and resp.get("eval_count"):
        d["eval_ms"] = resp["eval_duration"] / ns
        d["eval_tokens"] = resp["eval_count"]
    if ps:
        size, vram = ps.get("size"), ps.get("size_vram")
        if size:
            d["model_size"] = "{:.2f} GiB".format(size / (1024 ** 3))
            d["model_bytes"] = size
            d["vram_frac"] = (vram or 0) / size
        det = ps.get("details") or {}
        if det.get("parameter_size"):
            d["model_params"] = det["parameter_size"].rstrip("B") + " B"
        if det.get("quantization_level"):
            d["model_params"] += ", " + det["quantization_level"]
    return finish_rates(d)


def run_ollama_pass(tag, log_path=None, prompt=BENCH_PROMPT, ctx=CTX):
    log_snap = ollama_log_snapshot()
    t0 = time.monotonic()
    resp = ollama_api("/api/generate", {
        "model": tag,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": N_PREDICT, "num_ctx": ctx, "seed": 7},
    })
    wall_s = time.monotonic() - t0
    keep_log(log_path, json.dumps(resp, indent=1))
    ps = ollama_ps_entry(tag)
    d = map_ollama(resp, wall_s, ps)
    d["kv_types"] = ollama_kv_since(log_snap)
    if d["kv_types"]:
        d["kv_source"] = "Ollama runner log"
    return d, ps


def looks_like_tag(s):
    """An ollama tag has no path separator and no .gguf suffix. Anything
    path shaped that does not exist on disk must be reported as a missing
    file, not quietly retried as a tag: a diagnostic that misdiagnoses
    its own arguments has no business diagnosing your GPU."""
    return "/" not in s and not s.lower().endswith(".gguf")


def invocation():
    """The command that actually reached this process, kept pasteable.

    A source checkout needs the interpreter. The downloaded zipapp already
    has a shebang and must not name a picchio.py file that was never
    downloaded."""
    raw = sys.argv[0] or "picchio"
    if raw.endswith(".py"):
        return "python3 " + shlex.quote(raw)
    return shlex.quote(raw)


def hint_no_models():
    command = invocation()
    return (
        "picchio: no model given, and none found in the usual places\n"
        "(no ollama tags, no .gguf in the current folder, the HF cache,\n"
        "or the LM Studio folders).\n\n"
        "Point it at any .gguf file or ollama tag:\n"
        "  {} /path/to/model.gguf\n"
        "  {} some-tag:latest".format(command, command))


def human_size(nbytes):
    """Model-file scale: GiB with one decimal, MiB below one GiB,
    blank when the source offered nothing."""
    if not nbytes:
        return ""
    if nbytes >= 1024 ** 3:
        return "{:.1f} GiB".format(nbytes / float(1024 ** 3))
    return "{:.0f} MiB".format(nbytes / float(1024 ** 2))


def _sourced(note, size):
    """'ollama, 5.3 GiB' for a human; just the source when size is blank."""
    return note + (", " + size if size else "")


def scan_models():
    """Look around this machine (read only, fast) for models it can run:
    ollama tags (the live api, or the manifest folder when ollama is not
    up), then .gguf files in this folder, the HF cache and the LM Studio
    folders. Returns (label, note, arg, size) rows: label, note and size
    name the source for a human (size stays blank when nothing cheap
    reports it), arg is the exact string the pipeline runs."""
    ollama = []
    if ollama_reachable():
        try:
            for m in ollama_api("/api/tags", timeout=5).get("models", []):
                if m.get("name"):
                    ollama.append((m["name"], "ollama", m["name"],
                                   human_size(m.get("size"))))
        except (urllib.error.URLError, OSError, ValueError):
            pass
    else:
        base = os.path.expanduser("~/.ollama/models/manifests")
        for reg in glob.glob(os.path.join(base, "*", "*", "*", "*")):
            parts = reg.split(os.sep)
            full = "{}:{}".format(parts[-2], parts[-1])
            ollama.append((full, "ollama, not running", full, ""))

    patterns = (
        "*.gguf",
        "~/.cache/huggingface/hub/models--*/snapshots/*/*.gguf",
        "~/.cache/lm-studio/models/*/*/*.gguf",
        "~/.lmstudio/models/*/*/*.gguf",
    )
    seen, ggufs = set(), []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.expanduser(pat))):
            real = os.path.realpath(f)
            base = os.path.basename(f).lower()
            if real in seen or "mmproj" in base or f.endswith(".partial"):
                continue
            seen.add(real)
            try:
                size = human_size(os.path.getsize(real))
            except OSError:
                size = ""
            ggufs.append((os.path.basename(f), "gguf", f, size))
    # cap what the menu shows, but count the overflow so the presenters can
    # say "and N more" instead of dropping models silently (a long ollama
    # library used to hide every tag past the eighth with no hint at all)
    shown = ollama[:8] + ggufs[:8]
    dropped = max(0, len(ollama) - 8) + max(0, len(ggufs) - 8)
    return shown, dropped


def print_discovery(cands, dropped=0):
    """No terminal to ask at (a pipe or a redirect): print the commands
    that reproduce a run instead of a menu, each still pasteable as is.
    dropped counts models found past the cap, named so the list never
    hides one without a trace."""
    print("picchio: no model given. Runnable on this machine:\n")
    rows = [('"{}"'.format(arg) if " " in arg else arg,
             _sourced(note, size)) for label, note, arg, size in cands]
    w = min(max(len(q) for q, _ in rows), 48)
    command = invocation()
    for q, note in rows:
        print("  {} {:<{w}} ({})".format(command, q, note, w=w))
    if dropped:
        print("  ... and {} more not shown.".format(dropped))
    print("\nPick one, or point it at any other .gguf path or ollama tag.")


def _ask_line(prompt):
    """Read one line for the single direction question. EOF or ctrl-c
    returns None: declining to answer ends the flow, it does not crash it."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def resolve_direction(cands, interactive, ask, emit, dropped=0):
    """The whole zero-argument entry decision, in one place and pure of
    real IO so every path is testable. cands is what scan_models found;
    interactive says a terminal is on both ends; ask(prompt) returns the
    next typed line (or None at EOF); emit(line) shows one status line.
    dropped is how many models the scan found past the display cap, named
    in the menu so a long library never hides a model without a trace.
    Returns (action, model): ('run', arg) to diagnose arg, ('print', None)
    to fall back to pasteable commands, ('stop', None) for nothing to run.
    The flow asks at most once, and only at a real fork; one model needs no
    menu, and after the answer nothing else is asked."""
    if not interactive:
        return ("print", None) if cands else ("stop", None)
    if not cands:
        emit("No models found.")
        raw = (ask("Model (path or tag): ") or "").strip()
        return ("run", raw) if raw else ("stop", None)
    if len(cands) == 1:
        label, note, arg, size = cands[0]
        emit("1 model found.")
        emit("Selected: {} ({}).".format(label, _sourced(note, size)))
        return ("run", arg)
    emit("{} models found.".format(len(cands) + dropped))
    emit("")
    w = min(max(len(c[0]) for c in cands), 44)
    for i, (label, note, arg, size) in enumerate(cands, 1):
        if len(label) > w:
            label = label[:w - 14] + "..." + label[-11:]
        emit("  {:>2}) {:<{w}}  {:>9}   {}".format(i, label, size, note,
                                                   w=w))
    if dropped:
        emit("  ... and {} more not shown; type its tag or path to "
             "run any.".format(dropped))
    emit("")
    while True:
        line = ask("Model (number, path, or tag): ")
        if line is None or not line.strip():
            return ("stop", None)
        raw = line.strip()
        if raw.isdigit():
            k = int(raw)
            if 1 <= k <= len(cands):
                label, note, arg, size = cands[k - 1]
                emit("Selected: {} ({}).".format(label,
                                                 _sourced(note, size)))
                return ("run", arg)
            emit("No model {} in the list.".format(k))
            continue
        emit("Selected: {}.".format(raw))
        return ("run", raw)


def ollama_unload(tag):
    # Unload first so the cold pass pays the true load cost; ollama
    # keeps models resident for 5 minutes by default, and a cold number
    # measured against a resident model means nothing.
    try:
        ollama_api("/api/generate", {"model": tag, "keep_alive": 0},
                   timeout=60)
    except (urllib.error.URLError, OSError, ValueError):
        pass


# ---------------------------------------------- engine: llama-server (http)
#
# A model url instead of a path or tag means a llama-server someone
# already has running; picchio measures it over its own http api instead
# of launching anything. The api exposes no layer counts, no memory fit
# and no init log (checked against /props on b9430: nothing gpu shaped
# in it), so placement rests on the two witnesses that need no
# confession, the os meter and the prefill/decode signature. And the
# server owns its weights for as long as it lives: there is no unload
# call, so no cold pass exists, and the block says so instead of
# dressing a warm number as one.

def server_api(url, path, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url + path, data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def server_health(url):
    """GET /health: (True, None) when the server is up with its model
    loaded, else (False, reason), a still-loading server (it answers
    503 with an error body) told apart from nothing answering at all."""
    try:
        d = server_api(url, "/health", timeout=5)
        if d.get("status") == "ok":
            return True, None
        return False, ("the server at {} answered /health but is not "
                       "ready: {}".format(url, json.dumps(d)[:120]))
    except urllib.error.HTTPError:
        return False, ("the server at {} is still loading its model; "
                       "try again when /health says ok.".format(url))
    except (urllib.error.URLError, OSError, ValueError):
        return False, ("no llama-server answered at {}.\nStart one "
                       "(llama-server -m model.gguf) or check the "
                       "url.".format(url))


_PROPS_CACHE = {}


def server_props(url):
    """/props, fetched once per url. Fields used here, verified on
    b9430: model_path, model_alias, build_info, and the per request
    context under default_generation_settings.n_ctx."""
    if url not in _PROPS_CACHE:
        try:
            _PROPS_CACHE[url] = server_api(url, "/props", timeout=10)
        except (urllib.error.URLError, OSError, ValueError):
            _PROPS_CACHE[url] = {}
    return _PROPS_CACHE[url]


def server_ctx(url):
    """The context size a request to this server actually gets, or '?'
    when /props does not say; a question mark in the block beats a
    protocol default the server never promised."""
    try:
        return int(server_props(url)["default_generation_settings"]["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return "?"


def url_is_local(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def map_server(resp, wall_s):
    """response.timings from /completion, keys verified on b9430:
    prompt_n / prompt_ms cover the prompt tokens actually evaluated
    this pass, predicted_n / predicted_ms the generated ones, cache_n
    the prompt tokens reused from the kv cache instead of evaluated."""
    d = blank_pass()
    d["wall_s"] = wall_s
    # generation_settings is the server's own echo of what it sampled
    # with, keys verified on b9430. It spells temperature in full where
    # the cli log abbreviates it; one name reaches the block.
    gs = resp.get("generation_settings") or {}
    samp = {}
    for key, src in (("temp", "temperature"), ("top_k", "top_k"),
                     ("top_p", "top_p"), ("min_p", "min_p"),
                     ("seed", "seed")):
        v = gs.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            samp[key] = v
    d["sampling"] = samp or None
    t = resp.get("timings") or {}
    if t.get("prompt_ms") and t.get("prompt_n"):
        d["prompt_ms"] = float(t["prompt_ms"])
        d["prompt_tokens"] = int(t["prompt_n"])
    if t.get("predicted_ms") and t.get("predicted_n"):
        d["eval_ms"] = float(t["predicted_ms"])
        d["eval_tokens"] = int(t["predicted_n"])
    return finish_rates(d)


def run_server_pass(url, log_path=None, prompt=BENCH_PROMPT):
    # cache_prompt false, or the warm passes lie: this build reuses the
    # prompt kv across requests by default, and the second request then
    # evaluates 4 of 457 prompt tokens (measured here), which turns the
    # warm prefill rate into a per call overhead number, the short
    # prompt trap again. Forcing a full prefill every pass is the same
    # discipline as the keep_alive:0 unload in ollama mode, applied per
    # request; the wall clock is picchio's own, wrapped around the call.
    t0 = time.monotonic()
    resp = server_api(url, "/completion", {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "seed": 7,
        "ignore_eos": True,
        "cache_prompt": False,
    })
    wall_s = time.monotonic() - t0
    keep_log(log_path, json.dumps(resp, indent=1))
    t = resp.get("timings") or {}
    if t.get("cache_n"):
        sys.stderr.write("picchio: the server reused {} prompt tokens "
                         "from its cache despite cache_prompt false; "
                         "prefill is not a full read this pass.\n".format(
                             t["cache_n"]))
    if resp.get("truncated"):
        sys.stderr.write("picchio: the server truncated the prompt to "
                         "fit its context; rates are not comparable.\n")
    return map_server(resp, wall_s)


# ----------------------------------------------------------- telemetry (os)
#
# The engine's stderr is a confession; ioreg is the OS's own meter and
# does not care what the engine wrote. While the passes run, a thread
# polls the GPU accelerator entry a few times a second, so the verdict
# can cross check the claimed placement against what the silicon was
# seen doing: utilization over the compute windows, and the memory step
# the weights make when they actually land on the GPU.

TELE_HZ = 4.0  # one ioreg call costs 14-18 ms on the test machine; the
               # measured decode disturbance at 4 Hz is in README limits
TELE_PAD_S = 0.3  # decode ends about this long before the process does

def telemetry_start(disabled=False):
    """A running sampler, or a dict naming why there is none. The os
    line prints that reason, so a run without OS evidence says so
    instead of quietly reading like a fully instrumented one. Which
    meter answers here is gpu_meters' problem, not this file's."""
    meter = gpu_meters.open_meter(disabled)
    if isinstance(meter, dict):
        return meter
    first = meter.sample()
    if first is None:
        return {"off": "no {} data".format(meter.src)}
    return GpuSampler(meter, first)


class GpuSampler:
    def __init__(self, backend, first):
        self.samples = [first]
        self.marks = []
        # Only the sample source varies: the tick, the marks and the
        # window math below are the same physics on every platform, so
        # this class never asks which meter it is holding.
        self._backend = backend
        self._hot = backend.throttled()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        period = 1.0 / TELE_HZ
        while not self._stop.is_set():
            tick = time.monotonic()
            s = self._backend.sample()
            if s:
                self.samples.append(s)
            self._stop.wait(max(0.05, period - (time.monotonic() - tick)))

    def mark_pass(self, p):
        """Called the moment a pass returns: pins the pass to the wall
        clock, with the engine's own phase durations for the windows."""
        self.marks.append({
            "t_end": time.monotonic(), "wall_s": p["wall_s"],
            "load_s": (p["load_ms"] or 0) / 1000.0,
            "prompt_s": (p["prompt_ms"] or 0) / 1000.0,
            "eval_s": (p["eval_ms"] or 0) / 1000.0,
        })

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return telemetry_summary(self.samples, self.marks,
                                 self._hot or self._backend.throttled(),
                                 self._backend.src)


def _med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def telemetry_summary(samples, marks, hot=False, src=None):
    """Distills the timeline into what the verdict and the os line use:
    the idle baseline before pass 1, utilization over the compute
    windows, and the memory step inside the pass windows. Windows are
    tail aligned: decode ends at the pass end minus a small pad and
    prefill sits right before it; checked against the engine's own
    phase durations on the test machine (the gap between a head aligned
    load end and a tail aligned prefill start measured about 0.6 s)."""
    marks = [m for m in marks if m["wall_s"] and m["eval_s"]]
    if not samples or not marks:
        return {"off": "no samples"}
    t_first = marks[0]["t_end"] - marks[0]["wall_s"]
    pre = [s for s in samples if s["t"] < t_first]
    idle = [s["dev"] for s in pre]
    work, work_w, mem_run, dec_w = [], [], [], []
    for m in marks:
        dec1 = m["t_end"] - TELE_PAD_S
        dec0 = dec1 - m["eval_s"]
        pre0 = dec0 - m["prompt_s"]
        t0 = m["t_end"] - m["wall_s"]
        for s in samples:
            if pre0 <= s["t"] <= dec1:
                work.append(s["dev"])
                work_w.append(s.get("gpu_w"))
            if dec0 <= s["t"] <= dec1 and s.get("gpu_w") is not None:
                # decode only, kept apart from the combined compute
                # window on purpose: prefill runs the gpu much harder
                # (22.9 W peak against a 10.6 W decode median, measured
                # here), so energy per generated token priced off the
                # combined median would charge decode for prefill's
                # bursts
                dec_w.append(s["gpu_w"])
            if t0 <= s["t"] <= m["t_end"] and s["mem"] is not None:
                mem_run.append(s["mem"])
    mem_base = _med([s["mem"] for s in pre])
    step = None
    if mem_base is not None and mem_run:
        step = max(0, max(mem_run) - mem_base)
    work = [w for w in work if w is not None]
    return {
        "hz": TELE_HZ, "n": len(samples), "src": src,
        "idle_med": _med(idle),
        "idle_w": _med([s.get("gpu_w") for s in pre]),
        "work_med": _med(work),
        "work_n": len(work), "mem_step": step,
        "work_w": _med(work_w), "throttled": bool(hot),
        "dec_w": _med(dec_w), "dec_n": len(dec_w),
    }


def _watts(w):
    """Watts for a block line: a tenth up to three digits, then whole
    ones, so the field never grows past five characters."""
    return "{:.1f} W".format(w) if w < 100 else "{:.0f} W".format(w)


# Utilization alone cannot tell drawing from computing, and the band it
# confuses them across is an order of magnitude wide. Six pre-run
# windows measured on the Mac here, one median each, four of them in
# examples/raw: 0.00, 0.03, 0.13 W under a still desktop, then 0.38,
# 0.57 and 1.07 W under a busy one reading 30%, 43% and 50%
# utilization. Compute on that same chip, over those same runs' decode
# windows, costs 8.7, 10.5, 11.0, 11.4 and 13.4 W. Nothing has landed
# between 1.07 and 8.69 and the gate sits at the geometric centre of
# that empty band, a factor of three clear on either side.
#
# The discrete cards have no entry, which is a reading and not an
# oversight: of eight NVML pre-run windows the seven quiet ones sat at
# 0% utilization, where power is never consulted, and the one that read
# busy was a genuinely busy card, 31% at 122 W. No high utilization low
# power window has been seen on one, so there is no band to put a number
# in; borrowing the Mac's watts would be inventing a reading, and the
# amdgpu meter has never been sampled at all. Either one keeps the
# utilization answer until somebody measures it.
#
# A meter that records no src cannot be matched to a scale and gets no
# rescue either, which is why replaying an artifact from before this
# field existed returns exactly what it returned then.
IDLE_UTIL_GATE = 25
IDLE_W_GATE = {"ioreg": 3.0}


def pre_run_idle(tele):
    """(idle, note): whether the GPU was quiet before pass 1, and the
    one sentence that explains the answer when it is not the obvious
    one.

    A Mac desktop drives its GPU past half utilization to move windows
    and play video while drawing about a watt. That is pixels, and it
    takes nothing away from the run that follows, so a machine that
    looks busy by percent and idle by watts is idle. Power is the
    second signal because it is the one that separates the two.

    A meter with no calibrated idle draw of its own gets no second
    signal and keeps the utilization answer."""
    util = (tele or {}).get("idle_med")
    if util is None or util <= IDLE_UTIL_GATE:
        return True, None
    watt, gate = tele.get("idle_w"), IDLE_W_GATE.get(tele.get("src"))
    if watt is None or gate is None or watt >= gate:
        return False, None
    # short on purpose. It leads the verdict paragraph, and the
    # paragraph drops sentences from the tail to hold the block's line
    # budget, so every character this spends is taken off the end of
    # the reading advice. Measurement, inference, consequence, done
    return True, ("Pre-run gpu {:.0f}% at {}: not compute, so still "
                  "judged.".format(util, _watts(watt)))


def telemetry_vote(tele, rep, mode):
    """The OS evidence's vote on the engine's placement claim: agree,
    contradict or abstain. Only a full offload claim is judged; this
    tool hunts fake GPU claims, it does not overturn an engine that
    already confessed to CPU. Calibration on the test machine: a full
    Metal offload ran its compute windows at a median 99 device
    utilization with a +6.5 GiB memory step; a forced CPU run stayed
    at a median 0 with single-sample spikes to 53 from the desktop,
    which is why medians are judged and peaks are not."""
    if not tele or tele.get("off"):
        return "off"
    if mode == "ollama":
        full = rep["vram_frac"] is not None and rep["vram_frac"] >= 0.95
    else:
        n, total = rep["offload_n"], rep["offload_total"]
        full = n is not None and total and n >= total
    if not full:
        return "na"
    if tele["idle_med"] is None or tele["work_med"] is None \
            or tele["work_n"] < 6:
        return "abstain"
    if not pre_run_idle(tele)[0]:
        # the meter counts the whole GPU; when something else was
        # already computing on it none of it can be pinned on this one
        # process, so the numbers stop judging
        return "abstain"
    if tele["work_med"] >= 50:
        return "agree"
    if tele["work_med"] < tele["idle_med"] + 15:
        mb = rep.get("model_bytes")
        if mb and tele["mem_step"] is not None \
                and tele["mem_step"] >= 0.5 * mb:
            return "abstain"  # the memory step says the weights landed
        return "contradict"
    return "abstain"


def telemetry_read(tele):
    """The OS meter's own reading when there is no engine claim to
    judge (a server endpoint): 'busy', 'flat', or None when it cannot
    testify. Same gates as the vote: enough samples over the compute
    windows and an idle machine before the run; a middling median stays
    silent rather than guessing."""
    if not tele or tele.get("off"):
        return None
    if tele["idle_med"] is None or tele["work_med"] is None \
            or tele["work_n"] < 6:
        return None
    if not pre_run_idle(tele)[0]:
        return None
    if tele["work_med"] >= 50:
        return "busy"
    if tele["work_med"] < tele["idle_med"] + 15:
        return "flat"
    return None


def energy_per_token(tele, rep):
    """Joules per generated token, or None when the meters cannot say.

    Watts over the decode window divided by decode tokens per second is
    joules per token; the seconds cancel, which is why this needs no
    clock of its own and why anyone can recheck it from the block: take
    the watts off the os line, divide by the decode rate in the lane
    table above it. Same whole-GPU attribution as every other figure on
    that line: this is what the machine drew, not what this process
    drew, and on a shared GPU it is an upper bound.

    Gated on the same six samples the placement vote needs. A power
    median off two ticks is a number with no error bar pretending to
    have one."""
    if not tele or tele.get("off") or not rep:
        return None
    w, rate = tele.get("dec_w"), rep.get("decode_toks")
    if w is None or not rate or (tele.get("dec_n") or 0) < 6:
        return None
    return w / rate


def os_line(tele, rep=None):
    """The one line of OS evidence in the block, None only when the
    render has no telemetry context at all (pre-telemetry replays)."""
    if tele is None:
        return None
    if tele.get("off"):
        # the block's 66 columns are a feature: it gets pasted into
        # forum comments and a wrapped line reads like a broken tool.
        # "ioreg gave no gpu stats" was already two columns over before
        # a second meter existed, so the clamp is mechanical now rather
        # than a rule every new reason has to remember.
        line = "gpu not sampled ({}); evidence: {}".format(
            tele["off"], tele.get("ev", "engine+timing"))
        over = len(line) + 9 - WIDTH  # 9 = the "os" label gutter
        if over > 0:
            reason = tele["off"][:max(2, len(tele["off"]) - over - 2)] + ".."
            line = "gpu not sampled ({}); evidence: {}".format(
                reason, tele.get("ev", "engine+timing"))
        return line
    if not pre_run_idle(tele)[0]:
        # the watts are what disqualified it, so the watts are shown;
        # a meter that reports none says exactly what it said before
        watt = tele.get("idle_w")
        return "gpu {:.0f}%{} before the run, not idle; not judged".format(
            tele["idle_med"],
            " at " + _watts(watt) if watt is not None else " busy")
    parts = []
    if tele["idle_med"] is not None:
        parts.append("idle {:.0f}%".format(tele["idle_med"]))
    if tele["work_med"] is not None:
        parts.append("work {:.0f}%".format(tele["work_med"]))
    if tele["mem_step"] is not None:
        parts.append("mem +{:.1f} GiB".format(tele["mem_step"] / 1024 ** 3))
    w = tele.get("work_w")
    if w is not None:
        parts.append(_watts(w))
    if not parts:
        return "gpu sampled, nothing usable came back"
    # energy sits next to the watts it is derived from, so the two read
    # as one measurement rather than two claims. It is always present on
    # a line that got this far: the power channel is a private framework
    # on macOS and can go missing on a version drift, and a field that
    # quietly disappears when its meter does is indistinguishable from a
    # field nobody thought to print
    j = energy_per_token(tele, rep)
    parts.append("{:.2f} J/tok".format(j) if j is not None
                 else "n/a J/tok")
    if tele.get("throttled"):
        parts.append("throttled")
    line = "gpu " + ", ".join(parts)
    room = WIDTH - 9  # the label gutter, same as every other block line
    if len(line) > room:
        # the reasons branch above has clamped since a second meter
        # existed; this branch grew a field and inherits the same rule
        # rather than a second hand-computed worst case
        line = line[:max(2, room) - 2] + ".."
    return line


# ------------------------------------------------------------- aggregation

def warm_stats(passes, key):
    vals = [p[key] for p in passes[1:] if p.get(key)]
    if not vals:
        return None, None, None
    return statistics.median(vals), min(vals), max(vals)


# Within one run, on one model that stays loaded, prefill is the same
# computation every pass. Committed evidence for what legitimate looks
# like: llama.cpp on Metal spread 1.005x cold to warm, ollama 0.31.1 on
# Metal 1.05x once a nonce stops its prefix cache from answering for
# the warm passes (1.6x before that, and the wide half of that spread
# was the cache, not warmup). The pathology is three
# orders away from either, so the gate sits far above anything real and
# still catches it: ollama 0.32.15 on CUDA read 41.9 tok/s then 22923.
PREFILL_SPREAD_GATE = 10.0


def nonce_witnessed(passes):
    """True when this run's own evidence shows every pass sent a
    different prompt prefix. Read off the passes, and written into each
    kept meta as `prompt_nonce`, so a replayed artifact answers the
    same question its live run did."""
    seen = [p.get("nonce") for p in passes]
    return all(seen) and len(set(seen)) == len(seen)


def prefill_trust(passes, mode=None, gpu_kind=None):
    """(scope, reason) when the prefill lane cannot be quoted whole.

    Two different failures push a pass an order of magnitude off its
    siblings. A prefix cache hands the prompt back in milliseconds
    while still reporting every token of it. A first request that
    carries runner start-up or shader compilation charges seconds of
    one-time work to the prompt. Both mean a number that is not
    prefill, and which one it is decides how much has to go.

    With a nonce on every pass, the cache explanation is already ruled
    out by this run's own evidence: no two passes shared a prefix, so
    there was nothing to serve from. Then an outlying first pass is
    one-time work and only the cold cell abstains. Without that
    evidence, a replayed artifact cannot separate the two from inside
    one run, so the whole lane goes.

    Abstaining more than the evidence demands is its own kind of wrong
    answer: shader compilation and runner start-up are permanent
    features of those engines, and a rule that always eats the warm
    numbers because of them reports nothing forever."""
    rates = [p["prefill_toks"] for p in passes if p.get("prefill_toks")]
    if len(rates) < 2:
        return None
    lo, hi = min(rates), max(rates)
    if hi < PREFILL_SPREAD_GATE * lo:
        return None
    warm = rates[1:] or rates
    mid = statistics.median(warm)
    cold_alone = len(rates) > 2 and rates[0] in (lo, hi) \
        and max(warm) < PREFILL_SPREAD_GATE * min(warm)
    if cold_alone and nonce_witnessed(passes):
        what = "runner start-up" if mode == "ollama" \
            else "shader compilation" if gpu_kind == "Vulkan" \
            else "one-time setup"
        return ("cold", "Cold prefill abstains: {} put it {:.0f}x off "
                        "the warm passes, {:.0f} against {:.0f} tok/s."
                        .format(what, hi / lo, rates[0], mid))
    who = "the cold pass" if cold_alone else "the passes"
    return ("all", "Prefill abstains: {} ran {:.0f}x off the rest, "
                   "{:.0f} against {:.0f} tok/s, so one is not prefill."
                   .format(who, hi / lo, rates[0], mid))


def build_rep(passes):
    """Evidence from the last pass, rates replaced by warm medians."""
    rep = dict(passes[-1])
    for key in ("prefill_toks", "decode_toks", "wallclock_toks"):
        med, _, _ = warm_stats(passes, key)
        rep[key] = med or rep.get(key)
    # Ollama emits its kv line when a runner loads. Pass 1 deliberately
    # causes that load; warm passes reuse the runner and therefore emit no
    # second line. Preserve the newest concrete marker across the series.
    for p in reversed(passes):
        if p.get("kv_types"):
            rep["kv_types"] = p["kv_types"]
            rep["kv_source"] = p.get("kv_source")
            break
    # the sampler block rides the same load, so it keeps the same rule
    for p in reversed(passes):
        if p.get("sampling"):
            rep["sampling"] = p["sampling"]
            break
    return rep


# ------------------------------------------------------- WHY attribution

def placement_flags(argv):
    """Placement flags found on the engine command line, verbatim."""
    names = ("-ngl", "--n-gpu-layers", "--gpu-layers", "--device", "-dev")
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        for n in names:
            if tok == n and i + 1 < len(argv):
                out.append((n, argv[i + 1]))
                i += 1
                break
            if tok.startswith(n + "="):
                out.append((n, tok.split("=", 1)[1]))
                break
        i += 1
    return out


def effective_ctx(extra):
    """The ctx the engine actually got: the protocol default unless the
    passthrough args override it (llama.cpp honors the last -c given;
    picchio's own -c comes first on the command line)."""
    ctx = CTX
    for i, tok in enumerate(extra):
        if tok.startswith(("-c=", "--ctx-size=")):
            tok, val = tok.split("=", 1)
        else:
            val = extra[i + 1] if i + 1 < len(extra) else ""
        if tok in ("-c", "--ctx-size") and val.isdigit():
            ctx = int(val)
    return ctx


def attribute_why(state, rep, mode, engine_argv):
    """One WHY line for a degraded verdict, None otherwise. Climbs a
    fixed evidence ladder and stops at the first rung with real evidence
    behind it: an explicit flag the user passed, the engine's own memory
    fit figures, a backend init failure line. Every rung requires its
    evidence to be present in this run; when none is, the honest answer
    is the word unknown, not a plausible guess."""
    if state not in ("SILENT CPU FALLBACK", "PARTIAL OFFLOAD"):
        # a conflict never takes a WHY line: the ladder attributes a
        # proven degradation, and a conflict is two sources disagreeing
        # about whether one happened at all. The paragraph names the
        # fight; that is the attribution.
        return None
    why = None
    if mode == "ollama":
        # the ollama api exposes no command line, no fit log and no
        # init log, so the ladder has no rungs to climb here
        why = "unknown: not in the ollama api (check the server log)"
    elif mode == "server":
        # same blindness over http: /props carries no placement fields
        # at all (checked on b9430), so the cause lives in the server's
        # own stderr, not in anything picchio can reach
        why = "unknown: not in the server api (check its stderr log)"
    else:
        n, total = rep["offload_n"], rep["offload_total"]
        if n is None:
            # a fallback verdict with no engine claim at all is the
            # silent-engine conviction: the cause this run can prove is
            # the physics itself, and the WHY states it without
            # guessing at the build (a cpu only build can be deliberate)
            why = "no gpu evidence in the log; the gpu meter stayed idle"
            why = "WHY: " + why
            return why
        forced = []
        for name, val in placement_flags(engine_argv):
            # a flag only counts as the cause when its value matches the
            # placement the engine delivered; a flag that asked for more
            # GPU than was given did not cause the shortfall
            if (name in ("--device", "-dev")
                    and val.lower() in ("none", "cpu") and n == 0):
                forced.append("{} {}".format(name, val))
            elif (name in ("-ngl", "--n-gpu-layers", "--gpu-layers")
                    and val.isdigit() and n is not None
                    and int(val) == n and total and n < total):
                forced.append("{} {}".format(name, val))
        if forced:
            why = "forced by flag: " + " ".join(forced)
        elif (rep["fit_seen"] and rep["free_mib"] is not None
                and n is not None and total and n < total):
            why = "memory fit: saw {} MiB free, gave {}/{} layers".format(
                rep["free_mib"], n, total)
        elif rep["init_fail"]:
            why = rep["init_fail"]
        else:
            why = "unknown: the engine log does not say why"
    why = "WHY: " + why
    if len(why) > WIDTH:
        why = why[:WIDTH - 3] + "..."
    return why


# --------------------------------------------------------------- diagnosis

def decode_advice(cold_decode, warm_decode):
    """The decode sentence for a healthy verdict. Normally it quotes the
    warm median. But when that median fell well below the cold pass, which
    on an idle machine it never does (the decode rate excludes the load, so
    cold and warm agree), something shared the machine during the warm
    passes: the number is contaminated, so the sentence names the swing and
    says rerun instead of handing out a figure to quote."""
    if not warm_decode:
        return ""
    if cold_decode and warm_decode < 0.75 * cold_decode:
        return (" Warm decode fell to {:.1f} from the cold pass's {:.1f}; "
                "the machine was busy, rerun idle before quoting.".format(
                    warm_decode, cold_decode))
    return " Quote the warm median decode: {:.1f} tok/s.".format(warm_decode)


def diagnose(cold, rep, mode, tele=None):
    """Returns (state, paragraph). State drives the exit code.

    Three evidence sources vote: the engine's own confession (offload
    lines, ollama ps), the OS meter (ioreg utilization and memory, when
    sampled), and timing physics (the prefill/decode signature ratio).
    A full offload claim earns HEALTHY only while no source actively
    contradicts it; any two sources fighting is CONFLICTING EVIDENCE
    with the fight spelled out. A missing source abstains and the os
    line says what was missing; it never quietly counts as agreement.

    The block must stay inside HEIGHT lines; the renderer drops trailing
    sentences from the paragraph until it fits, so the load bearing
    sentence goes first."""
    decode = rep["decode_toks"] or cold["decode_toks"]
    prefill = rep["prefill_toks"] or cold["prefill_toks"]
    wait_s = 2500.0 / prefill if prefill else None
    vote = telemetry_vote(tele, rep, mode)

    def fallback_para():
        # prefill leads: with an os line and a WHY line in the block the
        # budget leaves this paragraph one line, and the hidden cost is
        # the sentence that must survive (decode's alibi shows in the
        # table right above)
        bits = []
        if prefill:
            bits.append("Prefill: {:.0f} s per 2500 "
                        "tokens.".format(wait_s))
        if decode:
            bits.append("Decode ({:.1f}) looks passable; that is how "
                        "this hides.".format(decode))
        return " ".join(bits) or "The gpu line above is the story."

    if mode == "server":
        # no confession exists over http: the server api exposes neither
        # layer counts nor a memory split, so the two witnesses that
        # need none vote on their own. Cutoffs are the calibrated ones
        # the other modes already use: work median 50 for busy, ratio
        # under 5 cpu shaped, 15 and over gpu shaped (healthy gpu runs
        # measured 20-44x here, cpu runs 2.3-5x).
        votes = []
        osr = telemetry_read(tele)
        if osr == "busy":
            votes.append(("the os meter saw the gpu work at "
                          "{:.0f}%".format(tele["work_med"]), "gpu"))
        elif osr == "flat":
            votes.append(("the os meter saw the gpu stay flat while the "
                          "tokens were made", "cpu"))
        ratio = prefill / decode if prefill and decode else None
        if ratio is not None and ratio >= 15:
            votes.append(("prefill ran {:.0f}x decode, gpu "
                          "shaped".format(ratio), "gpu"))
        elif ratio is not None and ratio < 5:
            votes.append(("prefill at {:.1f}x decode is cpu "
                          "shaped".format(ratio), "cpu"))
        shapes = {s for _, s in votes}
        if len(shapes) == 2:
            para = "{}; {}. No engine claim breaks the tie. Believe " \
                   "neither.".format(votes[0][0], votes[1][0])
            return "CONFLICTING EVIDENCE", para[0].upper() + para[1:]
        if shapes == {"cpu"}:
            para = " and ".join(t for t, _ in votes) \
                + ": the tokens were made on the cpu."
            if wait_s:
                para += " Prefill: {:.0f} s per 2500 tokens.".format(wait_s)
            return "SILENT CPU FALLBACK", para[0].upper() + para[1:]
        if shapes == {"gpu"}:
            para = " and ".join(t for t, _ in votes) \
                + ": the gpu did the work."
            para += decode_advice(cold["decode_toks"], decode)
            return "HEALTHY", para[0].upper() + para[1:]
        return "NO PLACEMENT EVIDENCE", (
            "The server api exposes no placement, and neither the os "
            "meter nor the timing signature was decisive here. Rates "
            "are measured; placement is not."
        )

    if mode == "ollama":
        frac = rep["vram_frac"]
        if frac is None:
            return "NO PLACEMENT EVIDENCE", (
                "Ollama did not report a memory split for this model, so "
                "picchio cannot say where it ran. Rates are measured; "
                "placement is not."
            )
        if frac < 0.05:
            return "SILENT CPU FALLBACK", fallback_para()
        if frac < 0.95:
            return "PARTIAL OFFLOAD", (
                "{:.0f}% of weights sat on CPU; expect rates below a "
                "fully offloaded run.".format(100 - frac * 100)
            )
        # ollama's reported split has been known to disagree with where
        # the kernels actually ran, so a full-GPU claim is cross checked
        # against the OS meter and the speed signature before HEALTHY.
        if vote == "contradict":
            return "CONFLICTING EVIDENCE", (
                "Ollama says 100% GPU; the OS saw the GPU stay flat "
                "while the tokens were made. Believe neither."
            )
        if prefill and decode and prefill < 5 * decode:
            return "CONFLICTING EVIDENCE", (
                "Ollama says 100% GPU; prefill at only {:.1f}x decode "
                "is CPU shaped. Believe neither.".format(prefill / decode)
            )
        para = "Ollama reports 100% of weights in GPU memory."
        para += decode_advice(cold["decode_toks"], decode)
        if prefill and decode and prefill > 3 * decode:
            para += (" {:.0f} tok/s is prefill: reading, not "
                     "writing.".format(prefill))
        return "HEALTHY", para

    n, total = rep["offload_n"], rep["offload_total"]
    if n is None:
        # silent-engine conviction, the linux killer case: a build with
        # no gpu support prints no placement evidence anywhere, yet the
        # machine has an nvidia gpu the os meter can see. Five gates,
        # every one required: no engine evidence at all (no offload
        # line and no device line), the nvml meter present, an idle
        # baseline with a flat compute window (telemetry_read's own
        # gates), and no exculpatory memory step (weights that landed
        # on the gpu veto the conviction, same veto the vote uses).
        # The prefill/decode ratio is deliberately not a gate: the
        # misbuilt 4090 fixture measured prefill at 15.1x decode on 48
        # EPYC threads, so the laptop calibrated 5x line does not
        # transfer to many core machines.
        if tele and tele.get("src") in ("nvml", "amdgpu") \
                and not rep["gpu_kind"] \
                and telemetry_read(tele) == "flat":
            mb = rep.get("model_bytes")
            stepped = mb and tele.get("mem_step") is not None \
                and tele["mem_step"] >= 0.5 * mb
            if not stepped:
                # cost sentence first: the renderer drops sentences from
                # the end under the line budget, the WHY line already
                # carries the evidence, and the 89 char evidence sentence
                # cannot survive a one line squeeze (the 4090 retest cut
                # it mid word as "eviden..")
                para = ("This build printed no gpu evidence and the "
                        "gpu stayed idle while the tokens were made.")
                if wait_s:
                    para = "Prefill: {:.0f} s per 2500 tokens. ".format(
                        wait_s) + para
                return "SILENT CPU FALLBACK", para
        return "NO PLACEMENT EVIDENCE", (
            "This build did not report layer placement, so picchio cannot "
            "prove where the model ran. Rates are measured; placement is "
            "not. A newer llama.cpp build logs it."
        )
    if n == 0:
        return "SILENT CPU FALLBACK", fallback_para()
    if total and n < total:
        return "PARTIAL OFFLOAD", (
            "{} layers sat on CPU; expect rates below a fully "
            "offloaded run.".format(total - n)
        )
    # both cross checks below read rates, so a run that produced none
    # passes them by absence and hands the claim an unearned HEALTHY:
    # issue #1 printed "the gpu did the work" over a table of n/a and a
    # cold breakdown of three zeroes. The rule at the top of this
    # docstring already says a missing source abstains; this is the
    # source the whole lane table rests on, so it abstains loudest.
    if cold.get("wall_s") and not any(
            (cold["load_ms"], cold["prompt_ms"], cold["eval_ms"],
             prefill, decode)):
        return "NO TIMING EVIDENCE", (
            "The engine says {}/{} layers on GPU and then printed no "
            "timings picchio could read: every rate above is unmeasured, "
            "not zero, and no verdict here rests on a number. Rerun with "
            "--keep-logs and open an issue with the log.".format(n, total)
        )
    # a full offload claim from stderr, cross checked the same way the
    # ollama one is: first the OS meter, then the speed signature
    if vote == "contradict":
        return "CONFLICTING EVIDENCE", (
            "The engine says {}/{} layers on GPU; the OS saw the GPU "
            "stay flat while the tokens were made. Believe "
            "neither.".format(n, total)
        )
    if prefill and decode and prefill < 5 * decode:
        return "CONFLICTING EVIDENCE", (
            "The engine says {}/{} layers on GPU; prefill at only "
            "{:.1f}x decode is CPU shaped. Believe neither.".format(
                n, total, prefill / decode)
        )
    para = "The GPU did the work."
    para += decode_advice(cold["decode_toks"], decode)
    if prefill and decode and prefill > 3 * decode:
        para += (" {:.0f} tok/s is prefill: reading speed, not "
                 "writing.".format(prefill))
    return "HEALTHY", para


def classify_number(x, rates):
    """rates: dict lane -> tok/s (may contain None). Returns (verdict, para)."""
    if x <= 0:
        return "NOT A RATE", "tok/s numbers are positive; nothing to check."
    lanes = [(k, v) for k, v in rates.items() if v]
    if not lanes:
        return "NOTHING TO COMPARE AGAINST", "No measured rates available."
    best, best_ratio = None, None
    for k, v in lanes:
        ratio = x / v
        off = max(ratio, 1 / ratio)
        if best_ratio is None or off < best_ratio:
            best, best_ratio = k, off
    lane_desc = {
        "prefill": "prompt reading speed, not generation speed",
        "decode": "generation speed, the number worth comparing",
        "wallclock": "tokens over total wall time, load and all",
    }
    measured = ", ".join("{} {:.1f}".format(k, v) for k, v in lanes)
    # The 1.30 band: wide enough to absorb the drift measured here
    # (same weights across two runtimes differed 12% on decode; warm
    # passes repeat within a few percent), narrow enough that decode
    # and wallclock, 1.4x apart on this machine, cannot both claim the
    # same number.
    if best_ratio <= 1.30:
        para = ("{:.1f} tok/s sits within {:.0f}% of the {} rate measured "
                "here. That reads like {}. (measured: {} tok/s)".format(
                    x, (best_ratio - 1) * 100, best, lane_desc[best],
                    measured))
        return "READS LIKE " + best.upper(), para
    para = ("{:.1f} tok/s is not within 30% of anything measured here "
            "(closest: {}, off by {:.1f}x; measured: {} tok/s). Before "
            "trusting that number, ask which of the three rates it was, "
            "and on what hardware, quant, and context length.".format(
                x, best, best_ratio, measured))
    return "MATCHES NOTHING MEASURED HERE", para


# --------------------------------------------------------------- rendering

def fmt_rate(v):
    return "{:.1f} tok/s".format(v) if v else "n/a"


def fmt_span(lo, hi, big=False):
    if lo is None:
        return "-"
    f = "{:.0f}~{:.0f}" if big else "{:.1f}~{:.1f}"
    return f.format(lo, hi)


def bar_line(label, secs, frac):
    barw = 28
    fill = max(0, min(barw, int(round(frac * barw))))
    return "  {:<13}{:>6.1f} s  {}{}  {:>3.0f}%".format(
        label, secs, "#" * fill, "." * (barw - fill), frac * 100
    )


def wrap_para(text):
    return textwrap.wrap(text, width=WIDTH - 2,
                         initial_indent="  ", subsequent_indent="  ")


def colorize(text, stream=None):
    """ANSI color for terminals only (stream defaults to stdout; guard
    passes stderr). Piped or redirected output stays pure ASCII, so a
    pasted block is identical to what the parser and the selftest see.
    NO_COLOR is respected. The id card is left unpainted on purpose:
    it is the paste totem, and it must look the same everywhere."""
    if os.environ.get("NO_COLOR") or not (stream or sys.stdout).isatty():
        return text
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    GREEN, RED, YELLOW = "\033[32m", "\033[31m", "\033[33m"
    states = (("SILENT CPU FALLBACK", RED), ("CONFLICTING EVIDENCE", YELLOW),
              ("NO TIMING EVIDENCE", YELLOW),
              ("PARTIAL OFFLOAD", YELLOW), ("NO PLACEMENT EVIDENCE", YELLOW),
              ("HEALTHY", GREEN), ("PASS", GREEN), ("FLAG", RED))
    out = []
    for line in text.splitlines():
        if line.startswith("VERDICT: "):
            for state, col in states:
                if state in line:
                    line = line.replace(state, BOLD + col + state + RESET, 1)
                    break
        elif line.startswith("gpu "):
            for word, col in (("NOT ENGAGED", RED), ("EVIDENCE UNKNOWN",
                              YELLOW), ("NO EVIDENCE", YELLOW),
                              ("PARTIAL", YELLOW), ("ENGAGED", GREEN)):
                if word in line:
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("GPU "):
            for word, col in (("GPU BUSY", GREEN), ("GPU IDLE", RED),
                              ("GPU MIXED", YELLOW),
                              ("GPU UNREADABLE", YELLOW)):
                if line.startswith(word):
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("picchio guard: "):
            for word, col in (("NOT ENGAGED", RED),
                              ("SILENT CPU FALLBACK", RED),
                              ("PARTIAL OFFLOAD", YELLOW),
                              ("ENGAGED", GREEN)):
                if word in line:
                    line = line.replace(word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith("picchio monitor: "):
            if line.startswith("picchio monitor: probing "):
                line = DIM + line + RESET      # one-time setup scaffolding
            else:
                # the timestamp, probe index and the "was Nx" baseline aside
                # are scaffolding: dim them so the eye lands on the state
                # word and the live rates, which keep full weight
                line = re.sub(r"\d\d:\d\d:\d\d probe\s+\d+",
                              DIM + r"\g<0>" + RESET, line, count=1)
                line = re.sub(r", was \d+x", DIM + r"\g<0>" + RESET,
                              line, count=1)
                for word, col in (("NOT ENGAGED", RED),
                                  ("SILENT CPU FALLBACK", RED),
                                  ("UNSURE", YELLOW),
                                  ("ENGAGED", GREEN)):
                    if word in line:
                        line = line.replace(word, BOLD + col + word + RESET, 1)
                        break
                # the count roster sits dim behind the summary's verdict
                # word, same idea as the discovery menu's columns
                line = re.sub(r" - \d+ probes.*$", DIM + r"\g<0>" + RESET,
                              line, count=1)
        elif line.startswith("SUSPECT: "):
            line = BOLD + YELLOW + "SUSPECT" + RESET + line[7:]
        elif line.startswith("  verdict"):
            for word, col in (("not judged", None), ("fits", GREEN),
                              ("tight", YELLOW), ("no", RED)):
                if word in line:
                    if col:
                        line = line.replace(
                            word, BOLD + col + word + RESET, 1)
                    break
        elif line.startswith(("WHY: ", "-- picchio")) or (
                line.startswith(("ctx ", "depth"))
                and "prefill" in line and "wallclock" in line
                and "tok/s" not in line):
            line = DIM + line + RESET
        elif line.startswith(("YOUR NUMBER: ", "SLOPE: ")):
            line = BOLD + line + RESET
        out.append(line)
    return "\n".join(out)


def menu_paint(line):
    """Discovery-menu color under colorize's contract: terminals only,
    NO_COLOR respected, and the plain text underneath is exactly what
    the selftest asserts on. Names carry the eye; the size and source
    columns sit dim behind them."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return line
    BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
    if re.match(r"^\d+ models? found\.$", line):
        return BOLD + line + RESET
    m = re.match(r"^(  +\d+\) )(.*?)(  +\S.*)$", line)
    if m:
        return m.group(1) + m.group(2) + DIM + m.group(3) + RESET
    # the overflow trailer is a footer hint, dim like the source columns
    if re.match(r"^  \.\.\. and \d+ more", line):
        return DIM + line + RESET
    return line


def gpu_line(rep, mode):
    if mode == "server":
        return "NO EVIDENCE (the server api exposes no placement)"
    if mode == "ollama":
        frac = rep["vram_frac"]
        if frac is None:
            return "EVIDENCE UNKNOWN (ollama gave no memory split)"
        pct = "{:.0f}% of weights in GPU memory (ollama ps)".format(
            frac * 100)
        if frac < 0.05:
            return "NOT ENGAGED: " + pct
        if frac < 0.95:
            return "PARTIAL: " + pct
        return "ENGAGED: " + pct
    n, total = rep["offload_n"], rep["offload_total"]
    if n is None:
        return "NO EVIDENCE (engine did not report layer placement)"
    if n == 0:
        g = "NOT ENGAGED: 0/{} layers on GPU".format(total)
    elif n < total:
        g = "PARTIAL: {}/{} layers on GPU".format(n, total)
    else:
        g = "ENGAGED: {}/{} layers on GPU".format(n, total)
    if rep["gpu_kind"] and rep["gpu_device"]:
        g += " ({}: {})".format(rep["gpu_kind"], rep["gpu_device"])
    elif rep["gpu_kind"]:
        g += " ({})".format(rep["gpu_kind"])
    return g


# The sampling settings the block reports, in the order the forum
# thread that asked for them listed them: temperature first, seed last.
# Nothing here is judged, only repeated; a sampler setting is not a
# defect and picchio has no opinion about which value is right.
SAMPLING_FIELDS = (("temp", "temp"), ("top_k", "top-k"),
                   ("top_p", "top-p"), ("min_p", "min-p"),
                   ("seed", "seed"))
# Why a given engine had nothing to disclose. Each names the surface
# that was read, so "not recorded" points at a place rather than being
# a shrug: the id card's kv line set this precedent.
NO_SAMPLING = {
    "ollama": "the ollama api returns no sampling settings",
    "server": "this server returned no generation_settings",
    "llama.cpp": "this build's log printed no sampler params",
}


# seed and top-k are counts; temp, top-p and min-p are continuous
# knobs. Formatting both by value rather than by field prints a
# temperature of exactly 1.0 as "1", which reads like a count in a line
# whose whole job is saying precisely which knob sat where. Measured on
# a real pair: Qwen3.6-35B-A3B carries general.sampling.temp = 1.0 in
# its own header and llama.cpp honors it, so this is not hypothetical.
SAMPLING_INTS = ("seed", "top_k")


def _samp_num(v, key):
    """A sampler value written the way a person writes it. The server
    hands back the float that survived a round trip through json
    ("temperature": 0.800000011920929) and the block is not the place
    to show sixteen digits of that."""
    if key in SAMPLING_INTS:
        return str(int(v))
    if float(v).is_integer():
        return "{:.1f}".format(v)
    return "{:.4f}".format(v).rstrip("0")


def settings_line(rep, mode):
    """The settings that were actually in force, quoted from whatever
    the engine itself said about them.

    This is disclosure, not measurement: every value here was already
    printed by the engine into a log picchio was parsing anyway, and
    the only thing that was missing was repeating it where a reader of
    the block can see it. An engine that says nothing gets "not
    recorded" and the name of the surface that stayed quiet, never a
    default filled in on its behalf. llama.cpp's own default temp is
    0.8, and printing 0.8 for an engine that never said so would be
    the one number in the block a reader could not trust."""
    s = rep.get("sampling") or {}
    parts = ["{} {}".format(label, _samp_num(s[key], key))
             for key, label in SAMPLING_FIELDS if s.get(key) is not None]
    line = ", ".join(parts) if parts else "not recorded: {}".format(
        NO_SAMPLING.get(mode, "this engine reported no settings"))
    room = WIDTH - 9  # the label gutter every block line shares
    if len(line) > room:
        # truncated, never dropped: same rule as the gpu line's
        # passthrough args, so a short settings line always means the
        # engine was short, not that the renderer ran out of room
        line = line[:max(2, room) - 2] + ".."
    return line


def render_verdict(mach, engine_str, model_name, passes, state, para, mode,
                   explain_part=None, cold_note=None, why=None,
                   ctx=CTX, extra=(), tele=None):
    """The block stays inside HEIGHT lines, kept narrow so it survives
    pasting into a forum comment (a long model name can push line one
    wider). The budget is a feature; a new line costs a decision."""
    cold = passes[0]
    rep = build_rep(passes)
    out = []
    bits = [model_name]
    if rep.get("model_params"):
        bits.append(rep["model_params"])
    if rep.get("model_size"):
        bits.append(rep["model_size"])
    bits.append(engine_str)
    out.append("model    " + ", ".join(bits))
    gline = "gpu      " + gpu_line(rep, mode)
    if extra:
        # passthrough args (the -ngl asked for, sampling overrides) ride
        # the gpu line: asked-for belongs next to delivered, and a new
        # line would break the budget. Truncated if long, never dropped:
        # on a new-format block, no bracket must mean no extra args.
        astr = " ".join(extra)
        room = WIDTH - len(gline) - 3
        if len(astr) > room:
            astr = astr[:max(2, room) - 2] + ".."
        gline += " [" + astr + "]"
    out.append(gline)
    oline = os_line(tele, rep)
    if oline:
        # the OS's independent reading, right under the engine's claim;
        # absent only on replays of runs that predate the sampler
        out.append("os       " + oline)
    # what the engine was told to sample with, directly above the lanes
    # those settings produced. The single most asked for missing field
    # in the thread this came from was temperature, and every number
    # below this line moves when it does.
    out.append("settings " + settings_line(rep, mode))
    pm, plo, phi = warm_stats(passes, "prefill_toks")
    dm, dlo, dhi = warm_stats(passes, "decode_toks")
    wm, wlo, whi = warm_stats(passes, "wallclock_toks")
    # a prefill lane nobody can stand behind prints the word instead of
    # the number. "abstain" and "n/a" are different answers: n/a is
    # nothing was read, abstain is it was read and it is not prefill.
    trust = prefill_trust(passes, mode, rep.get("gpu_kind"))
    scope, doubt = trust if trust else (None, None)
    warm_out = scope == "all"
    rows = []
    if mode != "server":
        rows.append(("cold", ["abstain" if scope
                              else fmt_rate(cold["prefill_toks"]),
                              fmt_rate(cold["decode_toks"]),
                              fmt_rate(cold["wallclock_toks"])]))
    rows.append(("warm mid", ["abstain" if warm_out else fmt_rate(pm),
                              fmt_rate(dm), fmt_rate(wm)]))
    rows.append(("warm span", ["abstain" if warm_out
                               else fmt_span(plo, phi, big=True),
                               fmt_span(dlo, dhi), fmt_span(wlo, whi)]))
    # The lane column widens when a number needs it. A 5090 reads
    # prefill in five digits, "21951.3 tok/s" fills the column exactly,
    # and the label then runs straight into it with a single space; the
    # committed four digit blocks all land on 13 and do not move. Two
    # spaces is the floor because a reader scanning a column, and the
    # parser reading it back, both need the gap to exist.
    lane = max([13] + [len(c) + 2 - (9 - len(label))
                       for label, cells in rows for c in cells[:1]])
    out.append("{{:<11}}{{:>{w}}}  {{:>{w}}}  {{:>{w}}}".format(w=lane)
               .format("ctx " + str(ctx), "prefill", "decode",
                       "wallclock"))
    for label, cells in rows:
        out.append("  {{:<9}}{{:>{w}}}  {{:>{w}}}  {{:>{w}}}"
                   .format(w=lane).format(label, *cells))

    if mode == "server":
        # the server owned the weights before pass 1, so no cold pass
        # exists: no cold row, no load bar, and one line saying so
        # beats a warm number dressed up as a cold one
        out.append("cold start not measured: the server already owned "
                   "the weights")
    else:
        wall = cold["wall_s"] or 0
        load_s = (cold["load_ms"] or 0) / 1000.0
        prefill_s = (cold["prompt_ms"] or 0) / 1000.0
        decode_s = (cold["eval_ms"] or 0) / 1000.0
        other_s = max(0.0, wall - load_s - prefill_s - decode_s)
        title = "where the cold pass went ({:.1f} s".format(wall)
        if rep.get("threads"):
            title += ", {}/{} threads".format(rep["threads"], rep["cores"])
        if cold_note:
            title += ", weights cached"
        out.append(title + ")")
        if wall > 0:
            out.append(bar_line("load weights", load_s, load_s / wall))
            out.append(bar_line("prefill", prefill_s, prefill_s / wall))
            out.append(bar_line("decode", decode_s, decode_s / wall))
            out.append(bar_line("engine misc", other_s, other_s / wall))
    # the budget is enforced, not hoped for: however many optional
    # lines rode in (the os line, a WHY line), trailing sentences drop
    # from the paragraph until the block fits
    # both leads explain something the reader can already see: an
    # abstain where a rate belongs, and an os line that judged a machine
    # whose own first field reads busy. They lead because the loop below
    # eats from the tail, and a visible oddity with its reason cut off
    # is worse than one line less of prose
    lead = [n for n in (doubt, pre_run_idle(tele)[1]) if n]
    if lead:
        para = " ".join(lead) + " " + para
    fixed = len(out) + (1 if why else 0) + 1  # + WHY + footer
    vlines = textwrap.wrap("VERDICT: {}. {}".format(state, para),
                           width=WIDTH - 2, subsequent_indent="  ")
    while len(vlines) > max(1, HEIGHT - fixed):
        body = para.rstrip()[:-1]
        cut = body.rfind(". ")  # whole sentences drop first,
        if cut >= 0:
            para = para[:cut + 1]
        else:
            cut = body.rfind("; ")  # then a trailing clause
            if cut < 0:
                break
            para = para[:cut] + "."
        vlines = textwrap.wrap("VERDICT: {}. {}".format(state, para),
                               width=WIDTH - 2, subsequent_indent="  ")
    room = max(1, HEIGHT - fixed)
    if len(vlines) > room:
        # a single uncuttable sentence can still overflow; the budget
        # is enforced, not hoped for, so truncate as the last resort
        vlines = vlines[:room]
        vlines[-1] = vlines[-1][:WIDTH - 4].rstrip() + ".."
    out.extend(vlines)
    if why:
        out.append(why)
    if explain_part:
        out.append("YOUR NUMBER: " + explain_part[0])
        out.extend(wrap_para(explain_part[1]))
    out.append("-- picchio v{} {} on {}, {} GB, {}".format(
        VERSION, PROTOCOL, mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    return "\n".join(out)


# ------------------------------------------------------------------- guard

RE_GUARD_OFF = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
# lines worth pinning for the exit summary even after the tail window
# has rolled past them: placement, fit, device and init evidence
RE_GUARD_PIN = re.compile(
    r"offloaded\s+\d+/\d+\s+layers|model buffer size|MiB free|"
    r"common_params_fit_impl|ggml_metal|ggml_cuda|ggml_vulkan|"
    r"ggml_backend|system_info")
# a line that means the engine moved past loading; placement evidence
# seen by now is final even on builds that print no buffer lines
RE_GUARD_PAST = re.compile(
    r"system_info|prompt eval time|listening|server is listening|"
    r"main: server", re.I)


def guard_why(rep, cmd):
    """WHY attribution for a degraded placement seen by guard; None when
    the placement is full (a healthy load needs no cause assigned)."""
    n, total = rep["offload_n"], rep["offload_total"]
    if n is None or not total or n >= total:
        return None
    state = "SILENT CPU FALLBACK" if n == 0 else "PARTIAL OFFLOAD"
    return attribute_why(state, rep, "llama.cpp", cmd)


def guard_state_line(rep, why):
    line = "picchio guard: " + gpu_line(rep, "llama.cpp")
    if why:
        line += "; " + why
    return line


def guard(cmd, keep_dir=None):
    """Wraps the user's own llama.cpp command (llama-server, llama-cli,
    anything that logs to stderr), tees its stderr through untouched,
    and speaks exactly twice on top of it: one placement line the moment
    the evidence is complete, and a short summary when the child exits.
    It never kills or signals the child: the requirement this mode comes
    from is a tool that warns but refuses to get in the way."""
    # the command being judged, echoed once at the top, quoted so it can
    # be pasted back into a shell and rerun. Everything guard says after
    # this is about these arguments, and a reader who only has the
    # output cannot otherwise tell which -ngl or which model produced
    # it. Never truncated: a shortened disclosure discloses nothing.
    sys.stderr.write("picchio guard: command: {}\n".format(
        " ".join(shlex.quote(a) for a in cmd)))
    try:
        child = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True,
                                 errors="replace")
    except OSError as e:
        sys.exit("picchio guard: could not start {}: {}".format(cmd[0], e))
    t0 = time.monotonic()
    log = open(os.path.join(keep_dir, "guard.stderr.txt"), "w") \
        if keep_dir else None
    # pinned keeps the load time placement evidence forever; tail keeps
    # the recent perf lines. A guarded server can log for hours, so the
    # full stream is never held in memory (the placement evidence sits
    # well under the caps: 238 lines at -lv 4, 1.7k at -lv 5 here).
    pinned, tail = [], []
    announced = False
    pending = False  # an "offloaded n/total" line arrived, unconfirmed
    try:
        for line in child.stderr:
            sys.stderr.write(line)
            if log:
                log.write(line)
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if len(tail) > 4000:
                del tail[:2000]
            if len(pinned) < 800 and RE_GUARD_PIN.search(stripped):
                pinned.append(stripped)
            if announced:
                continue
            if RE_GUARD_OFF.search(stripped):
                pending = True
                continue
            # the fit planning pass also prints an "offloaded" line, but
            # only the real load allocates buffers (its planning twin
            # reports 0.00 MiB and no _Mapped suffix), so an offloaded
            # line is confirmed by the next _Mapped buffer line, or by
            # any line that shows the engine already running
            if pending and ("_Mapped model buffer size" in stripped
                            or RE_GUARD_PAST.search(stripped)):
                rep = parse_stderr("\n".join(pinned), None)
                sys.stderr.write(colorize(
                    guard_state_line(rep, guard_why(rep, cmd)),
                    sys.stderr) + "\n")
                announced = True
    except KeyboardInterrupt:
        pass  # ctrl-c went to the child too; fall through to its exit
    try:
        code = child.wait()
    except KeyboardInterrupt:
        sys.exit(130)  # second ctrl-c: leave, still without killing it
    wall = time.monotonic() - t0
    if log:
        log.close()
    rep = parse_stderr("\n".join(pinned + tail[-1200:]), wall)
    out = ["picchio guard: {} exited {} after {:.1f} s".format(
        os.path.basename(cmd[0]), code, wall)]
    if rep["offload_n"] is None:
        out.append("picchio guard: no placement evidence appeared on "
                   "stderr; on llama.cpp builds where the default "
                   "verbosity hides it, add --verbose or -lv 4")
    else:
        out.append(guard_state_line(rep, guard_why(rep, cmd)))
    if rep["prefill_toks"] or rep["decode_toks"]:
        out.append("picchio guard: last rates seen: prefill {}, "
                   "decode {}".format(fmt_rate(rep["prefill_toks"]),
                                      fmt_rate(rep["decode_toks"])))
    sys.stderr.write(colorize("\n".join(out), sys.stderr) + "\n")
    # exit code: the child's own, passed through (128+N for a signal,
    # the shell convention). Measure mode owns its subprocess, so there
    # picchio's 0/2/3/4/5 codes are the product; here the subprocess is
    # the user's product, and scripts wrapping their server must keep
    # seeing the exit semantics they already depend on. The warning
    # lives on stderr, not in the code.
    sys.exit(code if code >= 0 else 128 - code)


def guard_cli(argv):
    keep = None
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio guard [--keep-logs DIR] -- <command...>\n"
              "wrap a llama.cpp command; warn on stderr the moment its\n"
              "own log shows layers landing off the GPU, never kill it,\n"
              "and print a placement summary when it exits.")
        sys.exit(0)
    if argv[:1] == ["--keep-logs"] and len(argv) > 1:
        keep = argv[1]
        os.makedirs(keep, exist_ok=True)
        argv = argv[2:]
    if argv[:1] != ["--"] or len(argv) < 2:
        sys.exit("picchio guard: usage: picchio guard "
                 "[--keep-logs DIR] -- <command...>")
    guard(argv[1:], keep)


# ----------------------------------------------------------------- compare

RE_QUANT = re.compile(r"\b(I?Q\d+(?:_[A-Z0-9]+)+|F16|BF16|F32)\b", re.I)


def parse_block(text):
    """Reads a pasted verdict block back into its variables. The input
    is a forum comment, so junk around the block is ignored. Fields the
    block does not carry stay None and print as unknown, never guessed;
    blocks from before the fingerprint fields have no ctx line, which
    is also how the two formats are told apart."""
    b = {k: None for k in ("model", "model_gib", "quant", "engine",
                           "place", "frac", "args", "ctx", "threads",
                           "chip", "ram", "os", "place_word", "verdict",
                           "os_raw", "os_work", "os_idle", "os_mem",
                           "os_watts", "os_note", "os_joules", "settings",
                           "protocol", "version")}
    rates = {}
    for line in text.splitlines():
        line = line.rstrip()
        m = re.match(r"model\s{4}(\S.*)", line)
        if m and b["model"] is None:
            b["model"] = m.group(1).split(",")[0].strip()
            sm = re.search(r"([\d.]+) GiB", m.group(1))
            b["model_gib"] = float(sm.group(1)) if sm else None
            em = re.search(r"((?:llama\.cpp|ollama)\s+\S+)$", m.group(1))
            b["engine"] = em.group(1) if em else None
            qm = RE_QUANT.search(m.group(1))
            b["quant"] = qm.group(1).upper() if qm else None
        m = re.match(r"gpu\s{6}(\S.*)", line)
        if m and b["place"] is None:
            g = m.group(1)
            am = re.search(r" \[(.+)\]$", g)
            if am:
                b["args"], g = am.group(1), g[:am.start()]
            for w in ("NOT ENGAGED", "PARTIAL", "ENGAGED",
                      "NO EVIDENCE", "EVIDENCE UNKNOWN"):
                if g.startswith(w):  # NOT ENGAGED before ENGAGED (substring)
                    b["place_word"] = w
                    break
            lm = re.search(r"(\d+)/(\d+) layers", g)
            pm = re.search(r"(\d+)% of weights", g)
            if lm and int(lm.group(2)):
                b["frac"] = int(lm.group(1)) / int(lm.group(2))
                b["place"] = "{}/{} layers on GPU".format(*lm.groups())
            elif pm:
                b["frac"] = int(pm.group(1)) / 100.0
                b["place"] = "{}% of weights on GPU".format(pm.group(1))
            else:
                b["place"] = g.split("(")[0].strip()
        m = re.match(r"ctx (\d+)\s+prefill", line)
        if m:
            b["ctx"] = int(m.group(1))
        # one space, not two: a five digit prefill fills the column and
        # the label runs straight into it (measured on an RTX 5090,
        # 21951.3 tok/s). The two fixed labels still anchor the row, so
        # loosening the gap cannot match anything else.
        m = re.match(r"\s{2}(cold|warm mid)\s+(\S.*)", line)
        if m and m.group(1) not in rates:
            # three cell vocabularies, not two: a rate, `n/a` for
            # nothing was read, and `abstain` for it was read and it is
            # not prefill. The renderer grew the third one and this did
            # not, so picchio could print a block it could no longer
            # read back and `verify` answered "no verdict block found",
            # which blames the paste for the tool's own gap. Both words
            # carry no number and both park as None; what they mean is
            # already said in the block's own verdict line.
            cells = re.findall(r"([\d.]+) tok/s|n/a|abstain", m.group(2))
            if len(cells) == 3:
                rates[m.group(1)] = [float(c) if c else None for c in cells]
        if line.startswith("where the cold pass went"):
            tm = re.search(r"(\d+/\d+) threads", line)
            if tm:
                b["threads"] = tm.group(1)
        m = re.match(r"os\s{2,}(gpu .*)", line)
        if m and b["os_raw"] is None:
            g = b["os_raw"] = m.group(1)
            if "not judged" in g:
                b["os_note"] = "not judged"
            elif "not sampled" in g:
                b["os_note"] = "not sampled"
            elif "nothing usable" in g:
                b["os_note"] = "unusable"
            for key, rx in (("os_idle", r"idle (\d+)%"),
                            ("os_work", r"work (\d+)%")):
                mm = re.search(rx, g)
                if mm:
                    b[key] = int(mm.group(1))
            mm = re.search(r"mem \+([\d.]+) GiB", g)
            if mm:
                b["os_mem"] = float(mm.group(1))
            mm = re.search(r"([\d.]+) W\b", g)
            if mm:
                b["os_watts"] = float(mm.group(1))
            mm = re.search(r"([\d.]+) J/tok\b", g)
            if mm:
                b["os_joules"] = float(mm.group(1))
        m = re.match(r"settings (\S.*)", line)
        if m and b["settings"] is None:
            # kept as written. Whether the engine disclosed anything is
            # part of what a reader of the block is judging, so a
            # "not recorded: ..." line reads back as itself, not as None
            b["settings"] = m.group(1)
        m = re.match(r"VERDICT: (\S.*)", line)
        if m and b["verdict"] is None:
            for st in ("SILENT CPU FALLBACK", "PARTIAL OFFLOAD",
                       "NO PLACEMENT EVIDENCE", "NO TIMING EVIDENCE",
                       "CONFLICTING EVIDENCE", "HEALTHY"):
                if m.group(1).startswith(st):
                    b["verdict"] = st
                    break
        m = re.match(r"-- picchio v(\S+) (\S+) on (.+), (\d+|\?) GB, (.+)",
                     line)
        if m:
            (b["version"], b["protocol"], b["chip"], b["ram"],
             b["os"]) = m.groups()
    b["row"] = "warm mid" if "warm mid" in rates else \
        ("cold" if "cold" in rates else None)
    b["rates"] = rates.get(b["row"]) or [None] * 3
    return b if b["model"] and b["row"] else None


def base_model(b):
    """Model name normalized for identity: quant token, .gguf suffix and
    separators dropped, so Qwen3.5-9B-Q4_K_M.gguf and qwen3.5:9b read as
    the same weights. Registry tags also drop suffixes like -Instruct,
    so containment counts as a match; that rule is mechanical, not fuzzy."""
    s = re.sub(r"\.gguf$", "", b["model"], flags=re.I)
    return re.sub(r"[^a-z0-9]", "", RE_QUANT.sub("", s).lower())


def suspect_para(a, b):
    """The attribution ladder, mechanical and in fixed order: placement,
    then quantization, then a ctx an order of magnitude apart, then
    hardware. The first rung whose evidence differs takes the blame and
    the climb stops; a rung missing its evidence on either side is
    skipped and named, never guessed across. Returns (text, skipped)."""
    skipped = []

    def known(key):
        if a[key] is not None and b[key] is not None:
            return True
        skipped.append({"frac": "placement"}.get(key, key))
        return False

    ma, mb = base_model(a), base_model(b)
    if not (ma == mb or ma in mb or mb in ma):
        text = ("NOT COMPARABLE: different models ({} vs {}). The ladder "
                "ranks configuration, not models.".format(a["model"],
                                                          b["model"]))
    elif known("frac") and abs(a["frac"] - b["frac"]) > 0.02:
        text = ("SUSPECT: placement. A ran {}, B ran {}. Fix that first; "
                "nothing else gets blamed while the first rung "
                "differs.".format(a["place"], b["place"]))
    elif known("quant") and a["quant"] != b["quant"]:
        text = ("SUSPECT: quantization. Placement agrees, the weights do "
                "not ({} vs {}): different bytes per token, so the rates "
                "are not one series.".format(a["quant"], b["quant"]))
    elif known("ctx") and max(a["ctx"], b["ctx"]) >= 10 * min(a["ctx"],
                                                              b["ctx"]):
        text = ("SUSPECT: context size. Placement and quant agree; ctx "
                "{} against {} is an order of magnitude, and the KV "
                "cache scales with it.".format(a["ctx"], b["ctx"]))
    elif a["chip"] and b["chip"] and (a["chip"] != b["chip"]
                                      or a["ram"] != b["ram"]):
        text = ("SUSPECT: hardware. Every config variable both blocks "
                "carry agrees; the machines differ ({}, {} GB vs {}, {} "
                "GB). What is left is silicon, mostly memory bandwidth; "
                "a block cannot rank that.".format(
                    a["chip"], a["ram"], b["chip"], b["ram"]))
    else:
        if not (a["chip"] and b["chip"]):
            skipped.append("machine")
        text = ("NO SUSPECT: every variable both blocks carry agrees. "
                "What remains (background load, thermals, power mode, "
                "disk cache) does not print in a block; picchio will "
                "not guess.")
    minor = [k for k in ("engine", "threads", "os")
             if a[k] and b[k] and a[k] != b[k]]
    if minor and text.startswith(("SUSPECT: hardware", "NO SUSPECT")):
        text += (" Outside the ladder these differ too: "
                 + ", ".join(minor) + ".")
    return text, skipped


def render_compare(names, a, b):
    def cell(v):
        s = "unknown" if v is None else str(v)
        return s if len(s) <= 24 else s[:22] + ".."

    a, b = dict(a), dict(b)
    for x in (a, b):
        # a new-format block (it has a ctx line) with no bracket really
        # ran without extra args; an old block just cannot say
        x["args"] = x["args"] or ("none" if x["ctx"] else None)
        x["machine"] = "{}, {} GB".format(x["chip"], x["ram"]) \
            if x["chip"] else None
    rows = [(k, a[k], b[k]) for k in
            ("model", "quant", "engine", "place", "args", "ctx",
             "threads", "machine", "os")]
    if all(va == vb for _, va, vb in rows) and a["rates"] == b["rates"]:
        return ("picchio compare: A and B carry the same fingerprint "
                "and the same rates. Nothing to compare.")
    out = ["picchio compare", "A: " + names[0], "B: " + names[1], "",
           "{:<11}{:<26}{}".format("", "A", "B")]
    for label, va, vb in rows:
        same = va == vb and va is not None
        out.append("{:<11}{:<26}{}".format(
            label, cell(va), "same" if same else cell(vb)))
    note = a["row"] if a["row"] == b["row"] else \
        "A {}, B {}".format(a["row"], b["row"])
    out += ["", "rates ({}), tok/s:".format(note)]
    for i, lane in enumerate(("prefill", "decode", "wallclock")):
        va, vb = a["rates"][i], b["rates"][i]
        gap = "-"
        if va and vb:
            gap = "A {:.1f}x faster".format(va / vb) if va >= vb else \
                "B {:.1f}x faster".format(vb / va)
        out.append("  {:<11}{:>10}  {:>10}   {}".format(
            lane, "{:.1f}".format(va) if va else "n/a",
            "{:.1f}".format(vb) if vb else "n/a", gap))
    text, skipped = suspect_para(a, b)
    out += [""] + textwrap.wrap(text, width=WIDTH, subsequent_indent="  ")
    if skipped:
        out += textwrap.wrap("not judged, missing from one block: "
                             + ", ".join(skipped), width=WIDTH,
                             subsequent_indent="  ")
    return "\n".join(out)


def compare_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio compare A.txt B.txt\n"
              "each file holds one pasted verdict block (surrounding "
              "forum text is fine)")
        sys.exit(0)
    if len(argv) != 2:
        sys.exit("picchio compare: usage: picchio compare A.txt B.txt\n"
                 "each file holds one pasted verdict block (surrounding "
                 "forum text is fine)")
    blocks = []
    for path in argv:
        try:
            with open(path, errors="replace") as f:
                blk = parse_block(f.read())
        except OSError as e:
            sys.exit("picchio compare: {}".format(e))
        if blk is None:
            sys.exit("picchio compare: no verdict block in {} (need at "
                     "least the model line and a rates row)".format(path))
        blocks.append(blk)
    print(colorize(render_compare(argv, blocks[0], blocks[1])))


# ------------------------------------------------------------------ verify

def claim_shape(b):
    """Where a parsed block claims the work ran, from placement alone:
    'gpu', 'cpu', 'partial', or None when it reports no evidence."""
    if b["place_word"] in ("NO EVIDENCE", "EVIDENCE UNKNOWN"):
        return None
    frac = b["frac"]
    if b["place_word"] == "ENGAGED" or (frac is not None and frac >= 0.95):
        return "gpu"
    if b["place_word"] == "NOT ENGAGED" or (frac is not None and frac < 0.05):
        return "cpu"
    if b["place_word"] == "PARTIAL" or frac is not None:
        return "partial"
    return None


def os_residency_witness(b):
    """True when the pasted os line saw a model-sized step in gpu memory.

    Live measurement already treats a step of at least half the model as
    proof the weights landed, and abstains rather than contradict a
    bursty utilization median (telemetry_vote). Static verification has
    to make the same vote or it rejects blocks the live judge correctly
    accepted: an RTX 5090 running Vulkan read a 0% median over 99
    samples of which 23 were non-zero and one peaked at 92%."""
    return b["os_mem"] is not None and b["model_gib"] is not None \
        and b["os_mem"] >= 0.5 * b["model_gib"]


def verify_block(b):
    """Recomputes the physics a verdict block claims and checks the block
    agrees with itself. Every number in it is a shadow of one run:
    placement, the prefill/decode signature, the os meter and the
    headline each answer 'did the gpu do the work', and an honest block
    has all of them describing the same run. Returns (verdict, findings):
    PASS with no findings, or FLAG naming each physical contradiction.

    It cannot prove a block is real, since numbers can be faked so they
    agree; it proves only that a block contradicts itself, which is what
    fabrication and casual tampering almost always leave behind."""
    pf, dc, wc = b["rates"]
    claim = claim_shape(b)
    ratio = pf / dc if pf and dc else None
    f = []
    # 1. lane ordering is pure physics, hardware independent: prefill
    #    reads the whole prompt in one batched pass, decode writes one
    #    token at a time reading every weight each time, and wallclock
    #    spreads the generated tokens over load and prefill as well. On a
    #    single run prefill > decode > wallclock always holds; an
    #    inversion is a number that was typed, not measured.
    if pf and dc and dc >= pf:
        f.append("decode {:.1f} >= prefill {:.1f} tok/s: generation cannot "
                 "outrun prompt reading on one run".format(dc, pf))
    if dc and wc and wc >= dc:
        f.append("wallclock {:.1f} >= decode {:.1f} tok/s: wall time "
                 "includes load and prefill, it cannot be faster".format(
                     wc, dc))
    # 2. the prefill/decode ratio is a scale free signature of placement:
    #    a full-gpu run measures 20-44x on the calibrated machines, a cpu
    #    run 2-5x. A ratio that fights the placement claim is the
    #    ollama-ps-lies case (#7323 family), now caught in a static paste.
    if ratio is not None and claim == "gpu" and ratio < 5:
        f.append("claims full gpu but prefill is only {:.1f}x decode, a cpu "
                 "shaped ratio (a real gpu run is 20x+)".format(ratio))
    if ratio is not None and claim == "cpu" and ratio >= 15:
        f.append("claims no gpu but prefill is {:.1f}x decode, a gpu shaped "
                 "ratio a cpu run never reaches".format(ratio))
    # 3. the os meter is an independent witness, held against the claim
    #    only when it was sampled and the machine was idle enough to read;
    #    a block whose own os line already abstained is not judged on it
    if b["os_work"] is not None and b["os_note"] is None:
        if claim == "gpu" and b["os_work"] < 15 \
                and not os_residency_witness(b):
            f.append("claims full gpu but its own os line saw the gpu at "
                     "{}% while the tokens were made".format(b["os_work"]))
        if claim == "cpu" and b["os_work"] >= 50:
            f.append("claims no gpu but its own os line saw the gpu busy at "
                     "{}% while the tokens were made".format(b["os_work"]))
    # 4. the headline must match the block's own placement line; a
    #    consistent body under a lying VERDICT word is the cheapest forgery
    if b["verdict"] == "HEALTHY" and claim in ("cpu", "partial"):
        f.append("headline says HEALTHY but the placement line says "
                 "{}".format(b["place_word"] or "not full gpu"))
    if b["verdict"] == "SILENT CPU FALLBACK" and claim == "gpu":
        f.append("headline says CPU FALLBACK but the placement line claims "
                 "the full gpu")
    return ("FLAG" if f else "PASS"), f


def render_verify(src, b, verdict, flags):
    pf, dc, wc = b["rates"]
    claim = claim_shape(b)
    shape = {"gpu": "full gpu", "cpu": "no gpu", "partial": "partial",
             None: "no placement evidence"}[claim]
    out = ["picchio verify: " + src,
           "  model     " + (b["model"] or "unknown"),
           "  claim     {} ({}), headline {}".format(
               b["place_word"] or "?", shape, b["verdict"] or "none")]
    if pf and dc:
        out.append("  signature prefill {:.1f} = {:.1f}x decode {:.1f}, "
                   "wallclock {}".format(
                       pf, pf / dc, dc,
                       "{:.1f}".format(wc) if wc else "n/a"))
    if b["os_raw"]:
        out.append("  os        " + b["os_raw"])
    if verdict == "PASS":
        witnessed = b["os_note"] is None and (b["os_work"] is not None
                                              or os_residency_witness(b))
        out.append("VERDICT: PASS. placement, the timing signature"
                   + (" and the os meter" if witnessed else "")
                   + " all describe the same run.")
    else:
        out.append("VERDICT: FLAG. {} physical contradiction{} in this "
                   "block:".format(len(flags),
                                   "" if len(flags) == 1 else "s"))
        for fl in flags:
            out.extend(textwrap.wrap(fl, width=WIDTH, initial_indent="  - ",
                                     subsequent_indent="    "))
        out.append("This block contradicts itself; do not trust its numbers "
                   "as one run.")
    return "\n".join(out)


def verify_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio verify [FILE]\n"
              "re-derive the physics a pasted verdict block claims, and\n"
              "flag it when placement, the prefill/decode signature, the\n"
              "os meter and the headline do not describe the same run.\n"
              "reads the block from FILE, or from stdin when none is given.")
        sys.exit(0)
    src = argv[0] if argv and argv[0] != "-" else None
    if src:
        try:
            text = open(src, errors="replace").read()
        except OSError as e:
            sys.exit("picchio verify: {}".format(e))
    else:
        text = sys.stdin.read()
        src = "pasted block"
    b = parse_block(text)
    if b is None:
        sys.stderr.write("picchio verify: no verdict block found in {} (need "
                         "the model line and a rates row).\n".format(src))
        sys.exit(2)
    verdict, flags = verify_block(b)
    print(colorize(render_verify(src, b, verdict, flags)))
    # reuse the measure exit map: a self-consistent block is 0, a block
    # whose sources fight is CONFLICTING EVIDENCE (5), the same code a
    # live run gets when two sources disagree
    sys.exit(0 if verdict == "PASS" else 5)


# ------------------------------------------------------------------- watch
#
# watch reads placement the engine-free way: it does not parse anyone's
# stderr, it points the OS meter at a running process or the whole GPU
# and reports what the silicon is doing. That makes it engine agnostic:
# MLX, LM Studio, vLLM, a raw torch script, anything that generates can
# be watched. ioreg meters the whole GPU, not one process, so watch
# never claims per-process precision: it reports machine level truth and
# says so, exactly the abstain discipline the measure-mode vote already
# uses on a busy desktop.

def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user
    # Signal 0 also succeeds for a dead child waiting to be reaped. Treat
    # that zombie as exited or watch PID can wait an hour after work ended.
    state = _cmd_out(["ps", "-p", str(pid), "-o", "stat="]).strip()
    return not state.startswith("Z") if state else True


def proc_name(pid):
    out = _cmd_out(["ps", "-p", str(pid), "-o", "comm="]).splitlines()
    return os.path.basename(out[0]) if out and out[0] else "?"


def ollama_loaded():
    """The first model ollama currently has resident, or None with a
    reason string. watch uses it only as a label for what is running."""
    if not ollama_reachable():
        return None, "no ollama is answering at {}".format(OLLAMA_HOST)
    try:
        models = ollama_api("/api/ps", timeout=5).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None, "ollama did not answer /api/ps"
    if not models:
        return None, "ollama is running but no model is loaded"
    return models[0].get("name") or models[0].get("model") or "?", None


def watch_summary(samples):
    """Machine-level aggregates, kept derivable from the raw JSONL."""
    dev = [s["dev"] for s in samples if s.get("dev") is not None]
    mem = [s["mem"] for s in samples if s.get("mem") is not None]
    watts = [s["gpu_w"] for s in samples if s.get("gpu_w") is not None]
    return {
        "n": len(samples),
        "secs": samples[-1]["t"] - samples[0]["t"] if len(samples) >= 2
        else 0.0,
        "work_med": _med(dev), "work_peak": max(dev) if dev else None,
        "work_min": min(dev) if dev else None,
        "mem_bytes": max(mem) if mem else None,
        "watts": _med(watts), "watts_peak": max(watts) if watts else None,
        "fell_idle": (min(dev) < 15 and max(dev) >= 50) if dev else None,
        "available": {"utilization": len(dev), "power": len(watts),
                      "memory": len(mem)},
        "throttled": False,
    }


def watch_verdict(summ, ctx):
    """Machine level placement read: is the GPU doing the work. ctx is a
    label for what is being watched (a process, an ollama model) or None
    for the whole machine; when set, the whole-GPU caveat is spelled out
    rather than pretending the number belongs to that one job."""
    wm = summ["work_med"]
    if wm is None:
        return "GPU UNREADABLE", "the gpu meter returned no usable samples."
    w = ", {:.1f} W".format(summ["watts"]) if summ["watts"] is not None else ""
    if wm >= 50:
        para = ("something is running kernels on the gpu (work {:.0f}% "
                "median, peak {:.0f}%{}).".format(wm, summ["work_peak"], w))
        if ctx:
            para += (" ioreg meters the whole gpu, so this is machine level, "
                     "not pinned to {}.".format(ctx))
            if summ["work_min"] is not None and summ["work_min"] < 15:
                para += " It fell idle between bursts, consistent with one job."
        return "GPU BUSY", para
    if wm < 15:
        para = "the gpu ran at {:.0f}% median over the window.".format(wm)
        para += (" If {} is generating tokens now, it is doing it on the cpu, "
                 "not the gpu.".format(ctx) if ctx
                 else " Nothing is driving the gpu right now.")
        return "GPU IDLE", para
    para = ("the gpu is lightly used (work {:.0f}% median, peak {:.0f}%{}): "
            "partial offload, or another job sharing it.".format(
                wm, summ["work_peak"], w))
    if ctx:
        para += " ioreg is whole-gpu; machine level only."
    return "GPU MIXED", para


def render_watch(ctx, summ, state, para):
    out = ["picchio watch" + (": " + ctx if ctx else "")]
    out.append("  window   {:.1f} s, {} samples at {:.0f} Hz  (whole "
               "gpu)".format(summ["secs"], summ["n"], TELE_HZ))
    parts = []
    if summ["work_med"] is not None:
        parts.append("work {:.0f}% median".format(summ["work_med"]))
    if summ["work_peak"] is not None:
        parts.append("peak {:.0f}%".format(summ["work_peak"]))
    if summ["watts"] is not None:
        parts.append("{:.1f} W".format(summ["watts"]))
    if summ["throttled"]:
        parts.append("throttled")
    if parts:
        out.append("  gpu      " + ", ".join(parts))
    if summ["mem_bytes"] is not None:
        out.append("  memory   {:.1f} GiB in use by the gpu".format(
            summ["mem_bytes"] / 1024 ** 3))
    out += textwrap.wrap("{}: {}".format(state, para), width=WIDTH,
                         subsequent_indent="  ")
    return "\n".join(out)


def watch_sample_json(sample, t0):
    return {
        "monotonicSeconds": round(sample["t"], 6),
        "elapsedSeconds": round(sample["t"] - t0, 6),
        "gpuUtilizationPercent": sample.get("dev"),
        "gpuPowerWatts": sample.get("gpu_w"),
        "gpuMemoryBytes": sample.get("mem"),
    }


def watch_json(target, summ, state, exit_code, started, ended, stop_reason):
    """Stable watch artifact. Field names carry units; null means absent."""
    warnings = ["GPU metrics are whole-GPU, not per-process attribution."]
    for key, label in (("utilization", "GPU utilization"),
                       ("power", "GPU power"), ("memory", "GPU memory")):
        missing = summ["n"] - summ["available"][key]
        if missing:
            warnings.append("{} was unavailable for {}/{} samples; raw "
                            "fields are null.".format(label, missing,
                                                       summ["n"]))
    return {
        "schema": "picchio.watch.v1", "tool": "picchio",
        "version": VERSION, "protocol": PROTOCOL, "target": target,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "endedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended)),
        "windowSeconds": round(summ["secs"], 3),
        "sampling": {"frequencyHz": TELE_HZ, "sampleCount": summ["n"],
                     "availableSamples": summ["available"]},
        "gpu": {
            "utilizationPercent": {"median": summ["work_med"],
                                   "peak": summ["work_peak"]},
            "powerWatts": {"median": summ["watts"],
                           "peak": summ["watts_peak"]},
            "memoryBytes": {"peak": summ["mem_bytes"]},
            "fellIdleBetweenBursts": summ["fell_idle"],
            "throttled": bool(summ["throttled"]),
        },
        "verdict": state, "exitCode": exit_code,
        "stopReason": stop_reason, "attribution": "whole_gpu",
        "warnings": warnings,
    }


def write_watch_logs(directory, samples, payload):
    os.makedirs(directory, exist_ok=True)
    sample_path = os.path.join(directory, "watch.samples.jsonl")
    summary_path = os.path.join(directory, "watch.summary.json")
    t0 = samples[0]["t"] if samples else 0.0
    with open(sample_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(watch_sample_json(sample, t0)) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")


def watch(pid=None, engine=None, duration=None, keep_dir=None, as_json=False):
    ctx, name = None, None
    if pid is not None and engine is not None:
        sys.exit("picchio watch: give a pid or an engine, not both.")
    if engine is not None:
        if engine != "ollama":
            sys.exit("picchio watch: only --engine ollama is supported "
                     "(any other engine: give its pid, or just watch the "
                     "whole gpu with no argument).")
        name, why = ollama_loaded()
        if name is None:
            sys.exit("picchio watch: {}. Load a model and generate, then "
                     "watch.".format(why))
        ctx = "ollama model " + name
    if pid is not None:
        if not pid_alive(pid):
            sys.exit("picchio watch: no process with pid {}.".format(pid))
        name = proc_name(pid)
        ctx = "{} (pid {})".format(name, pid)
    target = {"pid": pid, "name": name, "engine": engine}
    if keep_dir:
        try:
            os.makedirs(keep_dir, exist_ok=True)
        except OSError as e:
            sys.exit("picchio watch: could not create {}: {}".format(
                keep_dir, e))
    started = time.time()
    sampler = telemetry_start()
    if not isinstance(sampler, GpuSampler):
        sys.exit("picchio watch: no gpu meter here ({}). watch needs the "
                 "macos ioreg meter; on other platforms there is no engine "
                 "free placement signal yet.".format(sampler.get("off", "?")))
    if sampler._backend.src != "ioreg":
        # the linux meters feed measure mode only for now; watch has its
        # own calibration and is a separate milestone there
        sampler.stop()
        sys.exit("picchio watch: watch is ioreg only for now; the {} "
                 "meter runs inside measure mode.".format(
                     sampler._backend.src))
    # window: an explicit --for wins; otherwise watch until the pid exits
    # (capped), or a short fixed window for the whole-gpu snapshot
    if duration is None:
        duration = 3600.0 if pid is not None else 6.0
    sys.stderr.write("picchio watch: sampling the gpu{}{} ...\n".format(
        " while " + ctx if ctx else "",
        "" if pid is not None and duration >= 3600 else
        " for {:.0f} s".format(duration)))
    deadline, stop_reason = time.monotonic() + duration, "duration_elapsed"
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if pid is not None and not pid_alive(pid):
                stop_reason = "target_exited"
                break
            time.sleep(min(0.25, deadline - now))
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        sys.stderr.write("\n")
    sampler.stop()
    summ = watch_summary(sampler.samples)
    summ["throttled"] = sampler._hot or sampler._backend.throttled()
    state, para = watch_verdict(summ, ctx)
    code = 4 if state == "GPU IDLE" else 0
    payload = watch_json(target, summ, state, code, started, time.time(),
                         stop_reason)
    write_error = None
    if keep_dir:
        try:
            write_watch_logs(keep_dir, sampler.samples, payload)
        except OSError as e:
            write_error, code = e, 2
            payload["exitCode"] = code
            payload["warnings"].append("Evidence write failed: {}".format(e))
    human = render_watch(ctx, summ, state, para)
    if as_json:
        sys.stderr.write(colorize(human, sys.stderr) + "\n")
        print(json.dumps(payload, indent=1))
    else:
        print(colorize(human))
    if write_error:
        sys.stderr.write("picchio watch: could not write evidence in {}: {}\n"
                         .format(keep_dir, write_error))
    sys.exit(code)


def watch_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio watch [PID|ollama] [--for SEC] [--json] "
              "[--keep-logs DIR]\n"
              "point the os gpu meter at a running inference process (or\n"
              "the whole gpu) and report whether the gpu is doing the work,\n"
              "without parsing any engine's output. engine agnostic: works\n"
              "for mlx, lm studio, anything. macOS only (needs ioreg).\n"
              "--json keeps this human conclusion on stderr and writes a\n"
              "picchio.watch.v1 summary to stdout. --keep-logs writes\n"
              "watch.samples.jsonl and watch.summary.json. --engine ollama\n"
              "remains accepted as an alias for positional ollama.")
        sys.exit(0)
    pid = engine = dur = keep = None
    as_json, i = False, 0
    while i < len(argv):
        a = argv[i]
        if a == "--for" and i + 1 < len(argv):
            try:
                dur = float(argv[i + 1])
            except ValueError:
                sys.exit("picchio watch: --for wants a number of seconds.")
            if not dur > 0:
                sys.exit("picchio watch: --for wants a positive number.")
            i += 2
        elif a == "--engine" and i + 1 < len(argv):
            engine, i = argv[i + 1], i + 2
        elif a == "--json":
            as_json, i = True, i + 1
        elif a == "--keep-logs" and i + 1 < len(argv):
            keep, i = argv[i + 1], i + 2
        elif a.isdigit() and pid is None and engine is None:
            pid, i = int(a), i + 1
        elif not a.startswith("-") and pid is None and engine is None:
            engine, i = a, i + 1
        else:
            sys.exit("picchio watch: unexpected argument {!r}.\nusage: "
                     "picchio watch [PID|ollama] [--for SEC] [--json] "
                     "[--keep-logs DIR]".format(a))
    watch(pid, engine, dur, keep, as_json)


# ------------------------------------------------------------------ monitor
#
# measure and server mode each take one snapshot; a setup that runs on the
# GPU now can fall to the CPU an hour later, on a reload the http api never
# announces, and the next snapshot you happen to take is the only place you
# would ever see it. monitor closes that window: it sends one controlled
# probe on a fixed interval to a running llama-server, reads the per
# request prefill and decode timings the server already returns, and
# classifies each probe two ways, worse wins: the fixed ratio signature the
# block votes on (prefill under 5x decode is cpu shaped, 15x and over gpu
# shaped), and, once this server's own healthy baseline has locked from its
# first few probes, a collapse relative to that baseline. The relative
# signal is what makes the call self-calibrate to the machine under it
# instead of a laptop-tuned constant, so a fallback on a many-core box
# whose cpu keeps prefill above the fixed line still gets caught. Each
# probe is one line; a probe that flips the running
# placement prints a louder line, because a GPU that comes and goes between
# requests is exactly the failure a single snapshot cannot catch. It
# launches nothing and kills nothing: the server is the user's, monitor
# only knocks on it on a timer. The probe reuses measure's fixed prompt, so
# a short prompt can never make prefill look slow (the trap the whole tool
# exists to warn about); that is why it reads its own probes and not the
# user's variable length traffic.

MON_CPU_RATIO = 5.0     # prefill under this many x decode reads cpu shaped
MON_GPU_RATIO = 15.0    # this and over reads gpu shaped; between is unsure
MON_WARMUP = 3          # gpu-shaped probes that lock this server's healthy
                        # baseline; after it locks, a collapse relative to
                        # it flags even when the absolute ratio still passes
MON_DROP_FLAG = 0.5     # a ratio under this fraction of baseline is cpu
MON_DROP_WATCH = 0.7    # under this fraction is a dip worth a WATCH
MON_EVERY_S = 30.0      # default seconds between probes; each probe is one
                        # full BENCH_PROMPT completion, so a shorter gap
                        # puts more real load on the server being watched
MON_TAG = {"OK": "ENGAGED", "FLAG": "NOT ENGAGED",
           "WATCH": "UNSURE", "NODATA": "NO DATA"}


# ------------------------------------------------------------- residency
#
# The other half of the same question monitor already asks. Placement is
# "the engine says GPU, did the GPU do the work"; residency is "the
# engine says N GB, is N GB what this machine needs or just what it
# happens to be holding right now". A streaming MoE runtime keeps the
# routed experts in memory and leaves the rest on disk, so a single
# reading taken early is a working set, not a capacity.
#
# One physical fact shapes every judgment below: a resident set smaller
# than the model file is normal for anything mmapped, because untouched
# pages are never counted. The ratio alone proves nothing. What
# separates the cases is drift across rounds, which is why this rides
# monitor (already the across-rounds machine) and not a snapshot.

# Distinct passages, not one paragraph repeated. A MoE routes on content:
# probe the same text ten times and the same experts answer ten times, the
# resident set stops moving after round one, and a streaming runtime looks
# identical to a fully loaded one. Length is matched so the rate lanes stay
# comparable probe to probe; only the content varies.
_RES_SEEDS = (
    "A harbour master logs every vessel by draught and tide, refusing "
    "berths that would ground a hull at low water.",
    "The vineyard keeps its malolactic fermentation cool, trading a "
    "rounder mouthfeel for acidity the cellar can age.",
    "Sediment cores from the lakebed record ash layers, each one a "
    "eruption dated against the varves above and below it.",
    "The compiler hoists loop invariants only after proving no aliasing "
    "between the pointers the loop body dereferences.",
    "Baroque counterpoint forbids parallel fifths because the two voices "
    "stop sounding independent the moment they move together.",
    "Antitrust remedies split conduct from structure: behavioural decrees "
    "police a firm, divestiture changes what the firm is.",
    "Mycorrhizal networks trade phosphorus for photosynthate, and the "
    "exchange rate shifts with how shaded the seedling is.",
    "A rope soloist backs up the ascender with a clove hitch, because the "
    "device alone will strip the sheath on a shock load.",
    "Kiln atmosphere decides the glaze: the same copper reads green in "
    "oxidation and ox blood red under reduction.",
    "Actuarial reserving discounts future claims at a rate the regulator "
    "sets, so a rate change rewrites the balance sheet.",
)


def residency_prompt(i, words=None):
    """Probe i's prompt: one seed, padded with itself to the length the
    default bench prompt runs, so every probe costs the same prefill."""
    words = words or len(BENCH_PROMPT.split())
    seed = _RES_SEEDS[i % len(_RES_SEEDS)].split()
    out = []
    while len(out) < words:
        out.extend(seed)
    # trimmed to the exact count: a few percent of prefill drift between
    # probes would show up in the rate lanes as noise this run invented
    return " ".join(out[:words])


def ollama_model_path(tag):
    """The blob a tag's weights live in, read from the manifest rather
    than guessed: /api/show reports sizes, not paths."""
    base = os.path.expanduser("~/.ollama/models")
    name, _, ver = tag.partition(":")
    ver = ver or "latest"
    reg = "registry.ollama.ai"
    if "/" not in name:
        name = "library/" + name
    path = os.path.join(base, "manifests", reg, name, ver)
    try:
        with open(path) as f:
            layers = json.load(f).get("layers") or []
    except (OSError, ValueError):
        return None
    weights = [l for l in layers
               if str(l.get("mediaType", "")).endswith(".model")]
    if not weights:
        return None
    blob = max(weights, key=lambda l: l.get("size") or 0)
    digest = str(blob.get("digest", "")).replace(":", "-")
    blob_path = os.path.join(base, "blobs", digest)
    return blob_path if os.path.exists(blob_path) else None


def engine_pid_for(model_path):
    """The pid actually serving this model file, or None.

    Matched on the model path in the process command line, not on a
    process name: ollama renames and relocates its runner between
    releases, and the path is the one thing that identifies which loaded
    model this is when several are resident. lsof is the fallback, but it
    only answers when the weights are mmapped, and ollama runs its runner
    with --no-mmap, so it cannot be the primary."""
    if not model_path:
        return None
    hits = []
    for line in _cmd_out(["ps", "-axo", "pid=,command="]).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and model_path in fields[1] \
                and fields[0].isdigit():
            hits.append(int(fields[0]))
    if not hits and shutil.which("lsof"):
        for tok in _cmd_out(["lsof", "-t", model_path]).split():
            if tok.isdigit():
                hits.append(int(tok))
    if len(hits) > 1:
        # two processes hold the same weights: an old runner on its way
        # out, or a second engine. A reading nobody can attribute is
        # worse than no reading, so abstain, but say which case it was.
        sys.stderr.write("picchio: {} processes are serving {}; the "
                         "residency reading cannot be attributed to "
                         "one\n".format(len(hits), os.path.basename(
                             model_path)))
        return None
    return hits[0] if hits else None


def ttft_ms(p, mode):
    """Time to the first token, from figures the engine already returned.
    The third lane the memory number must never be quoted without: every
    byte a runtime declines to keep resident is paid for here, on the
    fault that pulls it back in."""
    prompt = p.get("prompt_ms")
    if prompt is None:
        return None
    return (p.get("load_ms") or 0) + prompt if mode == "ollama" else prompt


def residency_probe(pid, tag, mode):
    """One residency reading: what the OS says is resident, and what the
    engine says it is holding. Missing fields stay None, never zero."""
    from picchio_core.host import memory_snapshot
    rss = memory_snapshot(pid).get("rssBytes") if pid else None
    reported = vram = None
    if mode == "ollama":
        entry = ollama_ps_entry(tag) or {}
        reported, vram = entry.get("size"), entry.get("size_vram")
    return {"rss": rss, "reported": reported, "vram": vram}


# Calibrated on one machine, 2026-08-20, Apple M5 32 GB, the same
# qwen3.6:35b-a3b (20.61 GiB on disk) run two ways. Raw sessions in
# .ai/evidence-residency/:
#
#   ollama, Metal, fully loaded  20.8 GiB held, decode 26.6-33.3 tok/s
#                                (spread 1.25), first token 0.8-1.3 s
#   llama.cpp, cpu, mmap          5.3-14.9 GiB resident (median 9.7,
#                                ratio 0.47), decode 6.6-16.8 tok/s
#                                (spread 2.5), first token 16.5-31.7 s
#
# Two runs is two points. These are observation lines, not gates: the
# residency lane never sets the exit code, so a line drawn between two
# measurements can only mislabel a paragraph, never withhold a result.
RES_CAPACITY_RATIO = 0.9   # resident/file at or above this is the model
RES_MIN_PROBES = 5


# Shape needs three thirds of at least three rounds each to be read at
# all, and a run that has not reached that says so.
RES_SHAPE_MIN = 9
RES_FLAT_DISPERSION = 0.05  # median absolute deviation over the level


def residency_shape(series):
    """(shape, plateau bytes or None, one clause of detail).

    Level cannot separate a runtime that manages residency from one that
    simply has not touched the rest of the file yet. mmap hands out a
    small resident set for free on round one and a slightly larger one
    every round after, so a reading taken early reads exactly like a
    bounded cache. The curve is what tells them apart, and the axis is
    rounds, not seconds inside one run.

    Four shapes, none of them a threshold on gigabytes:

      RAMP        still climbing at the end. Whatever it holds now is
                  not what it will hold later, so no figure taken from
                  this run is quotable as a working set.
      PLATEAU     climbed, then stopped climbing. The plateau is the
                  working set, and it is the number worth comparing.
      FLAT        never moved. Everything was resident from round one.
      OSCILLATING moves with no direction. Measured on a machine under
                  memory pressure, where the kernel reclaims clean
                  pages as fast as the run touches them, so the level
                  is an equilibrium between two processes and not a
                  property of the engine at all.
    """
    if len(series) < RES_SHAPE_MIN:
        return ("UNDECIDED", None,
                "{} rounds is too few to read a shape; {} is the "
                "minimum".format(len(series), RES_SHAPE_MIN))
    cut = len(series) // 3
    lo = statistics.median(series[:cut])
    mid = statistics.median(series[cut:2 * cut])
    hi = statistics.median(series[2 * cut:])
    level = statistics.median(series) or 1
    noise = statistics.median([abs(x - level) for x in series])
    rise, late = hi - lo, hi - mid
    if abs(rise) <= noise:
        if noise / float(level) <= RES_FLAT_DISPERSION:
            return ("FLAT", level, "never moved across the run")
        return ("OSCILLATING", None,
                "swings {:.0%} of its own level with no direction, which "
                "is what reclaim looks like, not a "
                "cache".format(noise / float(level)))
    if rise < 0:
        return ("FALLING", None,
                "shrank across the run; the kernel is reclaiming faster "
                "than the run touches new pages")
    if late > 0.3 * rise:
        return ("RAMP", None,
                "still climbing at the last round, so nothing measured "
                "here is the number it settles at")
    return ("PLATEAU", hi, "climbed and then stopped climbing")


def residency_verdict(rows, file_bytes, mode):
    """(state, resident bytes, para) for the residency lane, or None when
    the lane never got a reading.

    Shape decides, not level. An earlier cut of this judged on the
    resident/file ratio plus how much decode wobbled, and it was wrong in
    the one way that matters: a run that is uniformly slow has a small
    wobble, so a 4 tok/s crawl with fifty second first tokens passed the
    steadiness test and got told it "really did need less than the file".
    Uniformly bad is not steady. The ratio is also exactly what mmap
    hands out for free, so no ratio can carry this verdict on its own.

    The other trap is instrument, not statistics: on unified memory a
    fully offloaded model lives where a process resident set cannot see
    it, so rss reads a couple of GiB for a twenty GiB model. Placement
    decides which instrument applies, and placement is already measured
    one lane over."""
    rows = [r for r in rows if r.get("rss")]
    if len(rows) < RES_MIN_PROBES:
        return None
    series = [r["rss"] for r in rows]
    decodes = [r["decode"] for r in rows if r.get("decode")]
    ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms")]
    resident = statistics.median(series)
    speed = statistics.median(decodes) if decodes else None
    first = statistics.median(ttfts) if ttfts else None
    # the triple always travels together: held bytes quoted without the
    # speed they bought is the reading this lane exists to take apart
    cost = ""
    if speed:
        cost = " It bought {:.0f} tok/s decode".format(speed)
        cost += " at {:.1f} s to first token.".format(first / 1000.0) \
            if first else "."
    on_gpu = sum(1 for r in rows if r.get("state") == "OK")
    if mode == "ollama":
        vram = [r["vram"] for r in rows if r.get("vram")]
        device = bool(vram) and statistics.median(vram) >= 0.5 * (
            statistics.median([r["reported"] for r in rows
                               if r.get("reported")] or [1]))
    else:
        device = on_gpu > len(rows) / 2
    if device:
        held = statistics.median([r["reported"] for r in rows
                                  if r.get("reported")] or [0]) or None
        return ("WEIGHTS ON DEVICE", held,
                "The weights are in device memory, which a process "
                "resident set does not count, so rss is not the residency "
                "figure here.{}".format(cost))
    if not file_bytes:
        return ("NO RESIDENCY EVIDENCE", resident,
                "The weights file could not be sized, so there is nothing "
                "to read the resident set against.")
    shape, plateau, detail = residency_shape(series)
    frac = "{:.0%} of the weights file".format(resident / float(file_bytes))
    if shape in ("UNDECIDED", "RAMP", "OSCILLATING", "FALLING"):
        # every one of these means there is no settled figure to quote,
        # which is the answer, not a failure to produce one
        return ("NO SETTLED WORKING SET", resident,
                "Resident set sat at {} but {}. Nothing measured here is "
                "a working set anyone can compare.{}".format(
                    frac, detail, cost))
    held = plateau or resident
    ratio = held / float(file_bytes)
    if ratio >= RES_CAPACITY_RATIO:
        return ("CAPACITY", held,
                "{:.0%} of the weights file stayed resident and {}: this "
                "number is the model, not a working set.{}".format(
                    ratio, detail, cost))
    return ("BOUNDED WORKING SET", held,
            "Held {:.0%} of the weights file and {}: this run kept less "
            "than the file and stayed there. What the rest cost is in the "
            "next sentence.{}".format(ratio, detail, cost))


def monitor_classify(prefill, decode, baseline=None):
    """One probe's verdict from its two rates. Returns (state, ratio), and
    the worse of two signals wins. The absolute signal is the fixed one the
    server block uses (prefill under 5x decode is cpu shaped, 15x and over
    gpu shaped); it is the only signal during warmup and always a floor.
    The relative signal, live once this server's own healthy baseline has
    locked, flags a ratio that collapsed to under half the baseline even
    when its absolute value would pass, which is how a fallback is caught
    on a many-core box whose cpu keeps prefill above the fixed line. NODATA
    when a rate is missing (nothing to convict on)."""
    if not prefill or not decode:
        return "NODATA", None
    ratio = prefill / decode
    if ratio < MON_CPU_RATIO:
        abs_state = "FLAG"
    elif ratio >= MON_GPU_RATIO:
        abs_state = "OK"
    else:
        abs_state = "WATCH"
    if not baseline:
        return abs_state, ratio
    if ratio < MON_DROP_FLAG * baseline:
        rel_state = "FLAG"
    elif ratio < MON_DROP_WATCH * baseline:
        rel_state = "WATCH"
    else:
        rel_state = "OK"
    rank = {"FLAG": 2, "WATCH": 1, "OK": 0}
    return (abs_state if rank[abs_state] >= rank[rel_state]
            else rel_state), ratio


def monitor_summarize(events):
    """The exit summary from the probe log. events is a list of (state,
    ratio) in order. Counts the two decisive states, the transitions
    between them (OK<->FLAG, the intermittent signal a snapshot misses),
    and the worst prefill/decode ratio seen."""
    ok = sum(1 for s, _ in events if s == "OK")
    flag = sum(1 for s, _ in events if s == "FLAG")
    decisive = [s for s, _ in events if s in ("OK", "FLAG")]
    trans = sum(1 for a, b in zip(decisive, decisive[1:]) if a != b)
    ratios = [r for _, r in events if r is not None]
    return {"n": len(events), "ok": ok, "flag": flag, "transitions": trans,
            "worst_ratio": min(ratios) if ratios else None}


def monitor_line(stamp, i, state, ratio, prefill, decode, baseline=None):
    """One compact status line per probe, pasteable, colorized by the
    monitor branch in colorize(). A degraded probe carries the baseline it
    fell from, so the line shows the call was made against this server and
    not a fixed number."""
    head = "picchio monitor: {} probe {:<3} {}".format(
        stamp, i, MON_TAG[state])
    if state == "NODATA":
        return head + "  the server returned no usable timings"
    tail = "prefill {}, decode {} ({:.1f}x".format(
        fmt_rate(prefill), fmt_rate(decode), ratio)
    if baseline and state in ("FLAG", "WATCH"):
        tail += ", was {:.0f}x".format(baseline)
    return "{}  {})".format(head, tail)


def monitor_summary_line(summ):
    """The one line printed when monitor stops: what it saw over the whole
    session, and the verdict that sets the exit code."""
    if summ["n"] == 0:
        return "picchio monitor: no probes completed."
    parts = ["{} probes".format(summ["n"]),
             "{} engaged".format(summ["ok"]),
             "{} on cpu".format(summ["flag"])]
    if summ["transitions"]:
        parts.append("{} placement change(s)".format(summ["transitions"]))
    if summ["worst_ratio"] is not None:
        parts.append("worst prefill {:.1f}x decode".format(
            summ["worst_ratio"]))
    verdict = "SILENT CPU FALLBACK seen" if summ["flag"] \
        else "ENGAGED throughout"
    return "picchio monitor: {} - {}".format(verdict, ", ".join(parts))


def _mon_secs(flag, val):
    try:
        s = float(val)
    except ValueError:
        sys.exit("picchio monitor: {} wants a number of seconds.".format(
            flag))
    if s <= 0:
        sys.exit("picchio monitor: {} wants a positive number.".format(flag))
    return s


def _monitor_wait(t0, every, deadline):
    """Sleep out the rest of one interval after a probe, but never past
    the --for deadline. Returns False once the deadline has arrived so the
    loop stops on time instead of one probe late."""
    target = t0 + every
    if deadline is not None:
        target = min(target, deadline)
    time.sleep(max(0.0, target - time.monotonic()))
    return deadline is None or time.monotonic() < deadline


def monitor_target_mode(arg):
    """What kind of running engine a monitor target names: 'server' for an
    http(s) llama-server url, 'ollama' for a bare model tag, None for a
    file path (monitor watches a running server, never a file on disk)."""
    if arg.startswith(("http://", "https://")):
        return "server"
    if "/" in arg or arg.lower().endswith(".gguf"):
        return None
    return "ollama"


def monitor_json(target, mode, engine, machine, baseline, timeline, summ,
                 residency=None, file_bytes=None):
    """The pasteable session artifact for an issue: what was watched, the
    baseline it calibrated to, every probe in order, and the verdict that
    set the exit code. Pure, so the selftest checks its shape with no live
    server."""
    return {
        "tool": "picchio", "version": VERSION, "protocol": PROTOCOL,
        "mode": mode, "target": target, "engine": engine,
        "machine": machine, "baseline_ratio": baseline, "probes": timeline,
        "summary": {"probes": summ["n"], "engaged": summ["ok"],
                    "cpu": summ["flag"], "transitions": summ["transitions"],
                    "worst_ratio": round(summ["worst_ratio"], 1)
                    if summ["worst_ratio"] is not None else None},
        "verdict": ("SILENT CPU FALLBACK seen" if summ["flag"]
                    else "ENGAGED throughout"),
        # a second axis, deliberately not wired to the exit code: the
        # placement lane owns that contract and one code cannot carry
        # two meanings
        "residency": {
            "weights_file_bytes": file_bytes,
            "state": residency[0], "resident_bytes": residency[1],
            "note": residency[2],
        } if residency else None,
        "exit_code": 4 if summ["flag"] else 0,
    }


def monitor(target, mode, every=MON_EVERY_S, duration=None, keep_dir=None,
            as_json=False, residency=False):
    """Probe a running engine on a timer and flag any probe whose
    prefill/decode signature collapses. mode is 'server' (a llama-server
    url, read over /completion) or 'ollama' (a model tag, over
    /api/generate); both read the per request timings the engine already
    returns and run the same ratio-plus-baseline call. One line per probe,
    a louder line when the placement flips, a pasteable json session on
    --json; never launches or signals the engine, and exits 4 the moment a
    probe caught the work off the gpu (0 if it held the whole session)."""
    if mode == "ollama":
        ver = ollama_reachable()
        if not ver:
            sys.exit("picchio monitor: no ollama is answering at {}. Start "
                     "it, or check OLLAMA_HOST.".format(OLLAMA_HOST))
        if not ollama_has_model(target):
            sys.exit("picchio monitor: ollama has no model tagged {}; check "
                     "`ollama list`.".format(target))
        engine, ctx = "ollama " + ver, CTX
    else:
        ok, why = server_health(target)
        if not ok:
            sys.exit("picchio monitor: " + why)
        build = server_props(target).get("build_info")
        engine = "llama-server" + (" " + str(build) if build else "")
        ctx = server_ctx(target)
    res_pid = res_path = res_bytes = None
    if residency:
        if mode == "ollama":
            res_path = ollama_model_path(target)
        else:
            res_path = server_props(target).get("model_path")
        try:
            res_bytes = os.path.getsize(res_path) if res_path else None
        except OSError:
            res_bytes = None
        if res_path is None:
            sys.stderr.write(
                "picchio monitor: cannot find the weights file for {}; the "
                "residency lane abstains and the placement lane runs as "
                "usual\n".format(target))
        else:
            sys.stderr.write(
                "picchio monitor: residency on {:.2f} GiB of weights\n"
                .format((res_bytes or 0) / 1024 ** 3))
    sys.stderr.write(
        "picchio monitor: probing {} every {:.0f} s (ctx {}); "
        "ctrl-c to stop\n".format(target, every, ctx))
    events, timeline, last_decisive, i = [], [], None, 0
    baseline, healthy = None, []
    deadline = (time.monotonic() + duration) if duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            i += 1
            t0 = time.monotonic()
            lp = os.path.join(keep_dir, "probe{}.response.json".format(i)) \
                if keep_dir else None
            try:
                pr = residency_prompt(i - 1) if residency else BENCH_PROMPT
                p = run_ollama_pass(target, lp, prompt=pr)[0] \
                    if mode == "ollama" \
                    else run_server_pass(target, lp, prompt=pr)
            except (urllib.error.URLError, OSError, ValueError) as e:
                # an engine that stopped answering is an event worth a line,
                # but not a cpu conviction; keep the timer running so a
                # restart is picked up on the next tick
                sys.stderr.write("picchio monitor: probe {} could not reach "
                                 "the engine: {}\n".format(i, e))
                if not _monitor_wait(t0, every, deadline):
                    break
                continue
            prefill, decode = p["prefill_toks"], p["decode_toks"]
            state, ratio = monitor_classify(prefill, decode, baseline)
            events.append((state, ratio))
            row = {
                "i": i, "state": state,
                "ratio": round(ratio, 1) if ratio else None,
                "prefill": round(prefill, 1) if prefill else None,
                "decode": round(decode, 1) if decode else None}
            if residency:
                # ollama spawns its runner on the first request, so the
                # pid does not exist before probe 1; a runner that dies
                # and comes back gets picked up the same way
                if res_pid is None or not pid_alive(res_pid):
                    res_pid = engine_pid_for(res_path)
                sample = residency_probe(res_pid, target, mode)
                ttft = ttft_ms(p, mode)
                # the three lanes ride together on purpose: a resident
                # figure quoted without the speed it bought is the number
                # this whole lane exists to take apart
                row.update({
                    "rss": sample["rss"], "reported": sample["reported"],
                    "vram": sample["vram"],
                    "ttft_ms": round(ttft, 1) if ttft else None,
                    "prompt_tokens": p.get("prompt_tokens")})
                sys.stderr.write(
                    "          resident {:>8}  reported {:>8}  "
                    "ttft {:>7}\n".format(
                        human_size(sample["rss"]) if sample["rss"] else "n/a",
                        human_size(sample["reported"])
                        if sample["reported"] else "n/a",
                        "{:.0f} ms".format(ttft) if ttft else "n/a"))
            timeline.append(row)
            sys.stderr.write(colorize(monitor_line(
                time.strftime("%H:%M:%S"), i, state, ratio, prefill, decode,
                baseline), sys.stderr) + "\n")
            # lock this server's own healthy baseline from its first few
            # gpu-shaped probes; after it locks a collapse from it flags
            # even when the absolute ratio would still clear the fixed line
            if baseline is None and ratio is not None \
                    and ratio >= MON_GPU_RATIO:
                healthy.append(ratio)
                if len(healthy) >= MON_WARMUP:
                    baseline = statistics.median(healthy)
                    sys.stderr.write(
                        "picchio monitor: baseline locked at {:.0f}x "
                        "prefill/decode for this engine; a collapse from "
                        "it now flags too\n".format(baseline))
            if state in ("OK", "FLAG"):
                if last_decisive and state != last_decisive:
                    sys.stderr.write(colorize(
                        "picchio monitor: placement changed {} -> {} at "
                        "probe {}".format(MON_TAG[last_decisive],
                                          MON_TAG[state], i),
                        sys.stderr) + "\n")
                last_decisive = state
            if not _monitor_wait(t0, every, deadline):
                break
    except KeyboardInterrupt:
        sys.stderr.write("\n")
    summ = monitor_summarize(events)
    sys.stderr.write(colorize(monitor_summary_line(summ), sys.stderr) + "\n")
    res = residency_verdict(timeline, res_bytes, mode) if residency else None
    if res:
        sys.stderr.write(colorize("picchio monitor: {} - {} of {} on disk. "
                                  "{}".format(
                                      res[0], human_size(res[1]) if res[1]
                                      else "n/a",
                                      human_size(res_bytes) if res_bytes
                                      else "n/a", res[2]),
                                  sys.stderr) + "\n")
    elif residency:
        sys.stderr.write("picchio monitor: residency lane got fewer than {} "
                         "readings; nothing to say.\n".format(RES_MIN_PROBES))
    if as_json:
        print(json.dumps(monitor_json(
            target, mode, engine, machine_info(),
            round(baseline, 1) if baseline else None, timeline, summ,
            res, res_bytes), indent=1))
    sys.exit(4 if summ["flag"] else 0)


def monitor_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio monitor TARGET [--every SEC] [--for SEC] "
              "[--residency] [--json] [--keep-logs DIR]\n"
              "probe a running engine on an interval and flag any probe\n"
              "whose prefill/decode ratio collapses from that engine's own\n"
              "healthy baseline: the intermittent fallback a single snapshot\n"
              "cannot see. TARGET is a llama-server url or an ollama tag.\n"
              "--residency also reads what the engine process is holding\n"
              "against the weights on disk, with the decode rate and first\n"
              "token time those bytes bought.\n"
              "--json prints a pasteable session summary. Never launches or\n"
              "kills the engine.")
        sys.exit(0)
    target = keep = None
    every, dur, as_json, i = MON_EVERY_S, None, False, 0
    residency = False
    while i < len(argv):
        a = argv[i]
        if a == "--every" and i + 1 < len(argv):
            every, i = _mon_secs("--every", argv[i + 1]), i + 2
        elif a == "--for" and i + 1 < len(argv):
            dur, i = _mon_secs("--for", argv[i + 1]), i + 2
        elif a == "--json":
            as_json, i = True, i + 1
        elif a == "--residency":
            residency, i = True, i + 1
        elif a == "--keep-logs" and i + 1 < len(argv):
            keep = argv[i + 1]
            os.makedirs(keep, exist_ok=True)
            i += 2
        elif not a.startswith("-") and target is None:
            target, i = a, i + 1
        else:
            sys.exit("picchio monitor: unexpected argument {!r}.\nusage: "
                     "picchio monitor TARGET [--every SEC] [--for SEC] "
                     "[--json] [--keep-logs DIR]".format(a))
    if target is None:
        sys.exit("picchio monitor: give a llama-server url or an ollama tag, "
                 "e.g. picchio monitor http://127.0.0.1:8080  or  "
                 "picchio monitor qwen3.5:9b")
    mode = monitor_target_mode(target)
    if mode is None:
        sys.exit("picchio monitor: {!r} looks like a file; monitor watches a "
                 "running server. Give a url (http://host:port) or an ollama "
                 "tag.".format(target))
    monitor(target, mode, every, dur, keep, as_json, residency)


# --------------------------------------------------------------- ctx sweep
#
# One tok/s number is measured at one context depth, almost always a
# short one, and quoted as if it held at any length. It does not: every
# token decode generates attends to the whole kv cache, so decode slows
# as the context fills. The sweep re-measures the three lanes at a few
# ctx depths, each with a prompt long enough to actually fill that depth
# (a short prompt at -c 32768 fills nothing and would just measure the 4k
# number again), and reports the decay slope. It answers a question the
# forums do not: what does your decode rate do at 32k that it did not at
# 4k.

def resolve_engine(model, bin_):
    """llama.cpp-vs-ollama-vs-server resolution shared by measure and
    sweep: an existing file is llama.cpp, an http(s) url is a running
    llama-server, a bare tag is ollama, a missing path is an error
    (never quietly retried as a tag). Returns (mode, binpath,
    engine_str, model_name); for a server the binpath slot carries the
    url, since there is no binary to find."""
    if model.startswith(("http://", "https://")):
        url = model.rstrip("/")
        ok, why = server_health(url)
        if not ok:
            sys.exit("picchio: " + why)
        props = server_props(url)
        name = os.path.basename(props.get("model_path") or "") \
            or props.get("model_alias") or url
        build = props.get("build_info") or "?"
        return "server", url, "llama-server " + str(build), name
    if os.path.isfile(model):
        binpath = find_binary(bin_)
        name = os.path.basename(model)
        try:
            # the header's own name beats an uninformative file name
            # (a 4090 fixture block read "model.gguf"); when the header
            # name lacks the quant token the file name carries, the
            # token rides along so the compare fingerprint stays whole
            gname = gguf_meta(model).get("general.name")
            if gname:
                qm = RE_QUANT.search(name)
                if qm and not RE_QUANT.search(gname):
                    gname += " " + qm.group(1).upper()
                name = str(gname)
        except (ValueError, struct.error, KeyError, OSError):
            pass
        return ("llama.cpp", binpath,
                "llama.cpp " + engine_version(binpath), name)
    if not looks_like_tag(model):
        sys.exit("picchio: no such file: {}\nRun picchio with no arguments "
                 "to see the models on this machine.".format(model))
    ver = ollama_reachable()
    if not ver:
        sys.exit("picchio: {!r} looks like an ollama tag, but no ollama "
                 "answered at {}.\nStart ollama, or give a .gguf path.".format(
                     model, OLLAMA_HOST))
    if not ollama_has_model(model):
        sys.exit("picchio: ollama at {} does not know the model {!r}.".format(
            OLLAMA_HOST, model))
    return "ollama", None, "ollama " + ver, model


def sweep_prompt(target_tokens):
    """A prompt long enough to fill about target_tokens of context, so
    decode is measured at real kv depth. English runs a little over one
    token per word here, so the paragraph is repeated until the word
    count crosses the target; the block reports the depth the engine
    actually reached, not this estimate."""
    words = len(_PARA.split())
    reps = max(1, int(target_tokens / (words * 1.25)))
    return "".join("Passage {}. {}".format(i + 1, _PARA) for i in range(reps))


def parse_tiers(spec):
    tiers = sorted({int(t) for t in spec.split(",")
                    if t.strip().isdigit() and int(t) > 0})
    if len(tiers) < 2:
        sys.exit("picchio: --ctx-sweep needs at least two ctx sizes, "
                 "e.g. --ctx-sweep 4096,32768")
    return tiers


def ctx_sweep(model, mode, binpath, engine_str, model_name, tiers, passes, lp):
    """Re-measures the three lanes at each ctx tier, each tier fed a
    prompt sized to fill it, so decode is read at real kv depth. Returns
    one row per tier: the depth actually reached and the warm median
    lanes. The first pass of each tier is the cold one and is dropped
    from the warm median, exactly as measure mode does it."""
    rows = []
    sys.stderr.write(
        "picchio: ctx sweep = {} full runs ({} tiers x {} passes), each a "
        "fresh prompt sized to fill its context; this takes minutes, not "
        "the ~1 min of a single run.\n".format(
            len(tiers) * passes, len(tiers), passes))
    for ctx in tiers:
        prompt = sweep_prompt(int(ctx * 0.7))  # leave headroom for 128 gen
        # one meter per tier, not one for the sweep: the question is how
        # much memory each depth costs, and a single figure spanning
        # every tier would answer only the deepest one
        sampler = telemetry_start()
        ps = []
        for i in range(passes):
            sys.stderr.write("picchio: ctx {} pass {}/{}{} ...\n".format(
                ctx, i + 1, passes, " (includes cold load)" if i == 0 else ""))
            if mode == "llama.cpp":
                p = run_llama_pass(
                    binpath, model, [],
                    lp("ctx{}.pass{}.stderr.txt".format(ctx, i + 1)),
                    prompt=prompt, ctx=ctx)
            else:
                p, _ = run_ollama_pass(
                    model, lp("ctx{}.pass{}.response.json".format(ctx, i + 1)),
                    prompt=prompt, ctx=ctx)
            # wall_s is measured by picchio, not in the engine log; persist
            # it per pass so the sweep table can be replayed like a verdict
            keep_log(lp("ctx{}.pass{}.meta.json".format(ctx, i + 1)),
                     json.dumps({"wall_s": p["wall_s"]}, indent=1))
            if isinstance(sampler, GpuSampler):
                sampler.mark_pass(p)
            ps.append(p)
        tele = sampler.stop() if isinstance(sampler, GpuSampler) \
            else dict(sampler)
        rep = build_rep(ps)
        rows.append({"ctx": ctx, "depth": rep.get("prompt_tokens"),
                     "prefill": rep["prefill_toks"],
                     "decode": rep["decode_toks"],
                     "wallclock": rep["wallclock_toks"],
                     # absent stays absent: a tier the meter could not
                     # read prints blank, never a zero anyone could
                     # average or plot as "no memory used"
                     "mem": tele.get("mem_step"),
                     "mem_src": tele.get("src") or tele.get("off")})
    keep_log(lp("sweep.meta.json"), json.dumps(
        {"engine": engine_str, "model_name": model_name, "mode": mode,
         "tiers": tiers, "passes": passes,
         "mem": {str(r["ctx"]): r.get("mem") for r in rows}}, indent=1))
    return rows


def sweep_slope(rows):
    """The decay sentence: decode from the shallowest to the deepest tier
    reached, the lane long context actually taxes. None when either end
    is unmeasured."""
    lo, hi = rows[0], rows[-1]
    if not (lo["decode"] and hi["decode"] and lo["depth"] and hi["depth"]):
        return None
    drop = (1 - hi["decode"] / lo["decode"]) * 100
    span = hi["depth"] / lo["depth"]
    ends = "{} to {} tokens ({:.0f}x deeper): {:.1f} -> {:.1f} tok/s".format(
        lo["depth"], hi["depth"], span, lo["decode"], hi["decode"])
    if drop >= 5:
        return ("decode fell {:.0f}% from {}. Long context is not free; the "
                "kv cache taxes every token you generate.".format(drop, ends))
    if drop >= -5:
        return ("decode held within {:.0f}% from {}. Here weight bandwidth "
                "dominates and the kv cache barely shows.".format(
                    abs(drop), ends))
    return ("decode read {:.0f}% faster from {}: that is measurement noise, "
            "not a real speedup at depth.".format(-drop, ends))


def render_sweep(mach, engine_str, model_name, rows):
    # The memory column appears only when a meter answered for at least
    # one tier. A replay of a run recorded before the sweep sampled
    # memory renders exactly as it did then, the same way the verdict
    # block's os line is absent rather than empty on old fixtures.
    mem = any(r.get("mem") is not None for r in rows)
    head = "{:<15}{:>12}  {:>12}  {:>12}".format(
        "depth   ctx", "prefill", "decode", "wallclock")
    out = ["ctx sweep  " + ", ".join(x for x in (model_name, engine_str) if x),
           (head + "  {:>9}".format("mem")) if mem else head]
    for r in rows:
        line = "{:<15}{:>12}  {:>12}  {:>12}".format(
            "{:>6}  {}".format(r["depth"] if r["depth"] else "?", r["ctx"]),
            fmt_rate(r["prefill"]), fmt_rate(r["decode"]),
            fmt_rate(r["wallclock"]))
        if mem:
            step = r.get("mem")
            line += "  {:>9}".format(
                "+{:.1f} GiB".format(step / 1024 ** 3) if step else "")
        out.append(line.rstrip())
    slope = sweep_slope(rows)
    if slope:
        out += textwrap.wrap("SLOPE: " + slope, width=WIDTH,
                             subsequent_indent="  ")
    out.append("-- picchio v{} ctx-sweep on {}, {} GB, {}".format(
        VERSION, mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    return "\n".join(out)


# ------------------------------------------------------- plan (capacity)
#
# picchio plan answers the question people ask before the download
# finishes: will this model fit this machine, and roughly how fast will
# it decode. The fit half is static and always available: the GGUF
# header carries the geometry, and the account it feeds matched the
# engine's own allocations exactly on both local models. The speed half
# is only ever a projection of this machine's own last measured run;
# with no run cached there is no number at all, because an estimate
# with no measurement behind it is a guess wearing digits. Nothing plan
# prints is a verdict block: no fixed line budget, no mp1 footer, so a
# projection can never be pasted somewhere a measurement belongs.

GGUF_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
              6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_meta_stream(f):
    """The GGUF v2/v3 header key value table, scalars and strings only
    (arrays are read past and dropped): magic, version, tensor count,
    kv count, then typed pairs. Layout and every key name used
    downstream were checked against the two local model files and
    against ollama's /api/show mirror of the same table."""
    if f.read(4) != b"GGUF":
        raise ValueError("not a gguf file (magic mismatch)")
    version = struct.unpack("<I", f.read(4))[0]
    if version < 2:
        raise ValueError("gguf v{} predates the v2 layout".format(version))
    # tensor count: no part of the kv account, but the id walk resumes
    # right after this function, so it rides along under a private key
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    if n_kv > 65536:
        raise ValueError("gguf header claims {} keys".format(n_kv))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 1 << 24:
            raise ValueError("gguf string of {} bytes".format(n))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            it = struct.unpack("<I", f.read(4))[0]
            cnt = struct.unpack("<Q", f.read(8))[0]
            if it == 8:
                for _ in range(cnt):
                    rstr()
            elif it == 9:
                raise ValueError("nested gguf array")
            else:
                f.seek(struct.calcsize(GGUF_TYPES[it]) * cnt, 1)
            return None  # array values feed nothing in the account
        fmt = GGUF_TYPES[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    out = {"__tensor_count": n_tensors}
    for _ in range(n_kv):
        key = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if v is not None:
            out[key] = v
    return out


def gguf_meta(path):
    with open(path, "rb") as f:
        return gguf_meta_stream(f)


def _arch_get(meta, key):
    arch = meta.get("general.architecture")
    return meta.get("{}.{}".format(arch, key)) if arch else None


def plan_is_moe(meta):
    return bool(_arch_get(meta, "expert_count"))


def kv_account(meta, ctx=CTX):
    """(kv bytes at ctx, note). Formula: ctx x attention layers x kv
    heads x (key length + value length) x 2 bytes of f16. Hybrid
    attention models mark every Nth layer as full attention
    (full_attention_interval) and the rest hold constant state, so the
    interval divides the layer count; honoring it lands this exactly
    on the engine's own llama_kv_cache allocation for both local
    models (128.00 MiB on the 9B, 80.00 MiB on the 35B MoE, ctx 4096,
    the 9B line committed in examples/raw/healthy-metal). Experts
    change the ffn, not the kv, so MoE needs no special case here.
    When key/value length are absent the classic head_dim fallback is
    embedding_length over head_count."""
    if not meta.get("general.architecture"):
        return None, "header lacks general.architecture"
    blocks = _arch_get(meta, "block_count")
    heads = _arch_get(meta, "attention.head_count")
    heads_kv = _arch_get(meta, "attention.head_count_kv") or heads
    klen = _arch_get(meta, "attention.key_length")
    vlen = _arch_get(meta, "attention.value_length")
    if (not klen or not vlen) and _arch_get(meta, "embedding_length") \
            and heads:
        klen = vlen = int(_arch_get(meta, "embedding_length")) // int(heads)
    if not (blocks and heads_kv and klen and vlen):
        return None, "header lacks the kv geometry keys"
    interval = int(_arch_get(meta, "full_attention_interval") or 1)
    att = max(1, int(blocks) // max(1, interval))
    note = "at ctx {}".format(ctx)
    if interval > 1:
        note += ", {} of {} layers attend".format(att, blocks)
    return int(ctx) * att * int(heads_kv) * (int(klen) + int(vlen)) * 2, note


PLAN_COMPUTE = 512 * 1024 ** 2  # the graph buffer: sched_reserve
# measured 505.02 MiB on the 9B and 493.00 MiB on the 35B here, so a
# flat half GiB stands in for what the header cannot predict


def plan_budget(mach):
    """(budget bytes, label). On macOS the wall is the metal working
    set, about 0.78 of ram: the engine itself reported 25558 MiB free
    on the idle 32 GB test machine (the MiB-free figure in
    examples/raw/healthy-metal). Elsewhere no fraction has been
    calibrated yet, so whole ram is the bar and the label says the
    check is ram only."""
    ram = (mach["ram_gb"] or 0) * 1024 ** 3
    if not ram:
        return None, "ram size unknown"
    if platform.system() == "Darwin":
        return int(ram * 0.78), "metal working set, 0.78 of {} GB ram" \
            .format(mach["ram_gb"])
    return ram, "system ram only, gpu memory not judged"


def plan_state(need, budget):
    """fits / tight / no. The 35B MoE measured HEALTHY fully offloaded
    at 85% of this budget (22.1 GB of weights on the 32 GB machine),
    so fits runs to 0.95; past 1.05 even an idle machine has no room
    left to find."""
    r = need / budget
    if r <= 0.95:
        return "fits"
    if r <= 1.05:
        return "tight"
    return "no"


def plan_speed_source(cache):
    """(bytes/s bandwidth, provenance) or (None, refusal). The one
    legal source for a speed figure here is this machine's own last
    measured run: warm decode times file size, the same arithmetic the
    README derives effective bandwidth with. No cached run means no
    number at all, and a mixture of experts cannot calibrate it: its
    decode reads only the active experts, so decode times file size
    overstates the bandwidth several fold."""
    if not cache or not cache.get("model_bytes") \
            or not (cache.get("rates") or {}).get("decode"):
        return None, ("speed: not calibrated, no measured run cached "
                      "on this machine. Run a diagnosis once (python3 "
                      "picchio.py MODEL) and plan gains an estimated "
                      "decode column from that run's bandwidth.")
    if cache.get("moe"):
        return None, ("speed: the cached run ({}) is a mixture of "
                      "experts, and its bandwidth arithmetic does not "
                      "transfer. Diagnose a dense model once for the "
                      "estimate.".format(cache.get("model_name", "?")))
    bw = cache["rates"]["decode"] * cache["model_bytes"]
    return bw, "calibrated by {} at {:.1f} tok/s decode".format(
        cache.get("model_name", "?"), cache["rates"]["decode"])


def plan_est_decode(bw, file_bytes, moe):
    """Estimated decode for one target, or None: a MoE target is never
    priced (the file is not what each token reads)."""
    if bw is None or not file_bytes or moe:
        return None
    return bw / file_bytes


def _gib(n):
    return "{:.1f} GiB".format(n / 1024 ** 3)


def plan_target(arg):
    """Resolve one plan argument into (name, file_bytes, meta, note):
    a .gguf path is read directly, an ollama tag through /api/show
    (model_info mirrors the same header keys) plus /api/tags for the
    blob size. meta is None when unreadable, and note says why."""
    if os.path.isfile(arg):
        try:
            return (os.path.basename(arg), os.path.getsize(arg),
                    gguf_meta(arg), None)
        except (ValueError, struct.error, KeyError, OSError) as e:
            return (os.path.basename(arg), os.path.getsize(arg),
                    None, str(e))
    if not looks_like_tag(arg):
        sys.exit("picchio plan: no such file: {}".format(arg))
    if not ollama_reachable():
        sys.exit("picchio plan: {!r} looks like an ollama tag, but no "
                 "ollama answered at {}.".format(arg, OLLAMA_HOST))
    try:
        show = ollama_api("/api/show", {"model": arg}, timeout=15)
    except (urllib.error.URLError, OSError, ValueError):
        sys.exit("picchio plan: ollama at {} does not know the model "
                 "{!r}.".format(OLLAMA_HOST, arg))
    size = None
    try:
        for m in ollama_api("/api/tags", timeout=5).get("models", []):
            if m.get("name") == arg or m.get("model") == arg:
                size = m.get("size")
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return arg, size, show.get("model_info") or {}, None


def plan_row(name, file_bytes, meta, note, budget, bw):
    """One accounted row: need, state, estimate; honest holes where
    the evidence is missing."""
    if file_bytes is None:
        return {"name": name, "need": None, "state": "not judged",
                "est": None, "moe": False,
                "note": note or "no size available"}
    kv, kv_note = kv_account(meta) if meta else (None, note or "?")
    need = file_bytes + (kv or 0) + PLAN_COMPUTE
    moe = plan_is_moe(meta) if meta else False
    state = plan_state(need, budget) if budget else "not judged"
    return {"name": name, "need": need, "state": state, "moe": moe,
            "est": plan_est_decode(bw, file_bytes, moe),
            "kv": kv, "kv_note": kv_note, "file": file_bytes,
            "note": None if meta else (note or "header unreadable")}


def render_plan_one(row, budget, blabel, bw, speed_note):
    out = ["picchio plan: " + row["name"]]
    if row["need"] is None:
        out.append("  " + row["note"])
        return "\n".join(out)
    out.append("  weights   {:>10}   the file itself".format(
        _gib(row["file"])))
    if row["kv"] is not None:
        out.append("  kv cache  {:>10}   {}".format(_gib(row["kv"]),
                                                    row["kv_note"]))
    else:
        out.append("  kv cache  {:>10}   not counted: {}".format(
            "?", row.get("kv_note") or row.get("note") or "?"))
    out.append("  compute   {:>10}   graph buffer, measured constant"
               .format(_gib(PLAN_COMPUTE)))
    out.append("  need      {:>10}".format(_gib(row["need"])))
    if budget:
        out.append("  budget    {:>10}   {}".format(_gib(budget), blabel))
        out.append("  verdict   {:>10}   {:.0f}% of budget".format(
            row["state"], 100.0 * row["need"] / budget))
    else:
        out.append("  verdict   not judged   " + blabel)
    if row["note"]:
        out.append("  note: " + row["note"])
    if row["moe"]:
        out.append("  speed: no estimate for a mixture of experts; each")
        out.append("  token reads only the active experts, so file size")
        out.append("  arithmetic would lie about it.")
    elif row["est"] is not None:
        out.append("  est decode  ~{:.1f} tok/s   estimate, not a "
                   "measurement".format(row["est"]))
        out += textwrap.wrap(speed_note, width=WIDTH - 4,
                             initial_indent="    ",
                             subsequent_indent="    ")
    if row["est"] is None and not row["moe"]:
        out += textwrap.wrap(speed_note, width=WIDTH,
                             initial_indent="  ", subsequent_indent="  ")
    return "\n".join(out)


def render_plan_scan(rows, budget, blabel, bw, speed_note):
    out = ["picchio plan: {} model{} on this machine".format(
        len(rows), "" if len(rows) == 1 else "s")]
    if budget:
        out.append("budget {} ({})".format(_gib(budget), blabel))
        out.append("kv counted at ctx {}".format(CTX))
    else:
        out.append("budget not judged: " + blabel)
    out.append("")
    calibrated = bw is not None
    head = "  {:<30}{:>9}   {:<5}".format("model", "need", "fit")
    if calibrated:
        head += "  est decode"
    out.append(head.rstrip())
    for r in rows:
        name = r["name"] if len(r["name"]) <= 30 else r["name"][:28] + ".."
        line = "  {:<30}{:>9}   {:<5}".format(
            name, _gib(r["need"]) if r["need"] else "?", r["state"])
        if calibrated:
            if r["est"] is not None:
                line += "  ~{:.1f} tok/s".format(r["est"])
            elif r["moe"]:
                line += "  n/a (moe)"
            else:
                line += "  n/a"
        out.append(line.rstrip())
    out.append("")
    if calibrated:
        out += textwrap.wrap("every est decode figure is an estimate "
                             "projected from one measured run ({}), not "
                             "a measurement".format(speed_note),
                             width=WIDTH, subsequent_indent="  ")
    else:
        out += textwrap.wrap(speed_note, width=WIDTH,
                             subsequent_indent="  ")
    return "\n".join(out)


def plan_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio plan [MODEL]\n"
              "the capacity account before you download or load: will it\n"
              "fit (gguf header geometry against this machine's memory\n"
              "budget), and, once one real diagnosis has been run here,\n"
              "an estimated decode rate. With no MODEL, accounts every\n"
              "model found on this machine. Estimates are labeled and\n"
              "never appear in a verdict block.")
        sys.exit(0)
    if len(argv) > 1:
        sys.exit("picchio plan: usage: picchio plan [MODEL]")
    mach = machine_info()
    budget, blabel = plan_budget(mach)
    bw, speed_note = plan_speed_source(load_cache())
    if argv:
        name, fb, meta, note = plan_target(argv[0])
        row = plan_row(name, fb, meta, note, budget, bw)
        print(colorize(render_plan_one(row, budget, blabel, bw,
                                       speed_note)))
        sys.exit(0)
    sizes = {}
    if ollama_reachable():
        try:
            for m in ollama_api("/api/tags", timeout=5).get("models", []):
                if m.get("name"):
                    sizes[m["name"]] = m.get("size")
        except (urllib.error.URLError, OSError, ValueError):
            pass
    rows = []
    for label, note, arg, _size in scan_models()[0]:
        if note == "gguf":
            n, fb, meta, why = plan_target(arg)
            rows.append(plan_row(n, fb, meta, why, budget, bw))
        elif note == "ollama":
            n, fb, meta, why = plan_target(arg)
            fb = fb or sizes.get(arg)
            rows.append(plan_row(n, fb, meta, why, budget, bw))
        else:
            rows.append({"name": label, "need": None, "est": None,
                         "moe": False, "state": "not judged",
                         "note": note})
    if not rows:
        sys.exit("picchio plan: no models found on this machine; give "
                 "it a .gguf path or an ollama tag.")
    print(colorize(render_plan_scan(rows, budget, blabel, bw, speed_note)))
    sys.exit(0)


# --------------------------------------------------- id (effective identity)
#
# picchio id splits the one word people trade ("4bit") back into the
# three axes it actually is. The weight recipe: general.file_type is a
# recipe name, and every recipe mixes per-tensor types, so the card
# walks the tensor table and prices each tensor by its ggml type into
# one effective bits-per-weight figure. The kv cache: a runtime flag,
# never in the file, so the card cites only a dtype this machine has
# measured. The experts: on a mixture of experts most weights sit
# parked, and expert_used_count over expert_count is how many wake per
# token. Same contract as plan: read only, no verdict block, exit 0.

# ggml type number -> (name, bytes per block, elements per block),
# probed from this machine's own libggml 0.13.1 (the library the
# b9430 binaries link) through ggml_type_name / ggml_type_size /
# ggml_blck_size. Removed and deprecated slots are absent on purpose:
# an unknown number refuses loudly instead of guessing a size.
GGML_TENSOR_TYPES = {
    0: ("f32", 4, 1), 1: ("f16", 2, 1), 2: ("q4_0", 18, 32),
    3: ("q4_1", 20, 32), 6: ("q5_0", 22, 32), 7: ("q5_1", 24, 32),
    8: ("q8_0", 34, 32), 9: ("q8_1", 36, 32), 10: ("q2_K", 84, 256),
    11: ("q3_K", 110, 256), 12: ("q4_K", 144, 256),
    13: ("q5_K", 176, 256), 14: ("q6_K", 210, 256),
    15: ("q8_K", 292, 256), 16: ("iq2_xxs", 66, 256),
    17: ("iq2_xs", 74, 256), 18: ("iq3_xxs", 98, 256),
    19: ("iq1_s", 50, 256), 20: ("iq4_nl", 18, 32),
    21: ("iq3_s", 110, 256), 22: ("iq2_s", 82, 256),
    23: ("iq4_xs", 136, 256), 24: ("i8", 1, 1), 25: ("i16", 2, 1),
    26: ("i32", 4, 1), 27: ("i64", 8, 1), 28: ("f64", 8, 1),
    29: ("iq1_m", 56, 256), 30: ("bf16", 2, 1), 34: ("tq1_0", 54, 256),
    35: ("tq2_0", 66, 256), 39: ("mxfp4", 17, 32),
    40: ("nvfp4", 36, 64), 41: ("q1_0", 18, 128),
}

# general.file_type number -> recipe name, from the llama_ftype enum
# in this machine's b9430 llama.h (removed slots absent, same rule)
LLAMA_FTYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS",
    31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0",
}


# The metadata keys a gguf can carry that say where the file came from
# and who produced it. A fixed list, matched exactly, because a looser
# rule ("any key with url in it") drags in the license link and the
# tokenizer homepage, neither of which identifies the quantizer. The
# indexed base_model rows are the upstream weights this file was made
# from; row 0 is the only one every published file fills in.
ID_SOURCE_KEYS = (
    "general.quantized_by",
    "general.repo_url",
    "general.source.url",
    "general.source.repo_url",
    "general.source.huggingface.repository",
    "general.organization",
    "general.base_model.0.repo_url",
    "general.base_model.0.organization",
)


def file_identity(path, chunk=1 << 22):
    """(sha256 hex, bytes read) over the whole file.

    The whole file, not a sampled fingerprint. Two reasons, and the
    second is the load bearing one. First, a head-and-tail fingerprint
    cannot notice an edit in the middle, and the point of this line is
    that any byte difference shows up. Second, the full sha256 is
    already the identity anchor everyone else uses: it is the oid
    HuggingFace stores for an LFS object and it is the name ollama gives
    the blob on disk, so this number can be checked against a registry
    nobody had to teach picchio about. A private fingerprint would be a
    number only picchio can read, which is the opposite of the job.

    Measured here at 2.28 GiB/s, so a 5 GiB file costs about 2 s and a
    22 GiB one about 10 s. The caller says so before it starts."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def file_fingerprint(path, emit=None):
    """(sha256 hex, byte count) for a file, or (None, reason).

    Announces itself first: several gigabytes of reading is a few
    seconds of nothing happening, and a card that returns instantly
    every other time looks hung rather than busy."""
    if not path:
        return None, "no file to read"
    if emit:
        emit("picchio: reading {} to fingerprint it\n".format(
            os.path.basename(path)))
    try:
        return file_identity(path)
    except OSError as e:
        return None, "{} could not be read in full ({})".format(
            os.path.basename(path), e)


def id_file_note(path, emit=None):
    """The identity line's text for a file on disk, or a note saying why
    there is none. Two people running this on the same bytes get the
    same string, character for character."""
    if not path:
        return None
    digest, nbytes = file_fingerprint(path, emit)
    if digest is None:
        return "not recorded: {}".format(nbytes)
    return "sha256 {}, {:,} bytes".format(digest[:12], nbytes)


def gguf_source_note(meta):
    """What the file says about its own origin, quoted as written.

    Every value here is the file talking about itself. Nothing in a gguf
    header is signed and picchio does not go to the network, so this is
    a claim to be repeated, never a provenance check that passed."""
    parts = []
    for key in ID_SOURCE_KEYS:
        v = (meta or {}).get(key)
        if isinstance(v, str) and v.strip():
            parts.append("{} {}".format(key[len("general."):], v.strip()))
    if not parts:
        return ("not recorded: this file carries no origin keys, so who "
                "produced it cannot be read off the file")
    return "; ".join(parts) + " (the file's own claim, verified " \
        "against nothing)"


def gguf_tensor_table(f, n_tensors):
    """The descriptor table between the kv section and the tensor
    data, layout verified against the local 9B file byte by byte: per
    tensor a u64-length name, u32 dimension count, u64 dims fastest
    first, u32 ggml type, u64 offset relative to the aligned start of
    the data section. Returns (descriptors, header end position)."""
    if n_tensors > 65536:
        raise ValueError("gguf header claims {} tensors".format(n_tensors))
    out = []
    for _ in range(n_tensors):
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 1 << 16:
            raise ValueError("gguf tensor name of {} bytes".format(n))
        name = f.read(n).decode("utf-8", "replace")
        nd = struct.unpack("<I", f.read(4))[0]
        if nd > 8:
            raise ValueError("{} claims {} dimensions".format(name, nd))
        dims = struct.unpack("<{}Q".format(nd), f.read(8 * nd))
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        out.append((name, dims, ttype, off))
    return out, f.tell()


def ollama_tensor_table(ts):
    """/api/show tensors (name, type string, shape) mapped onto the
    same descriptor tuples the file walk yields. The api mirrors the
    table without offsets (measured on 0.31.1), so only the type
    arithmetic can price it; the offset audit is file-only."""
    byname = {v[0].lower(): k for k, v in GGML_TENSOR_TYPES.items()}
    out = []
    for t in ts:
        tt = byname.get(str(t.get("type", "")).lower())
        if tt is None:
            raise ValueError("unknown tensor type {!r} on {}".format(
                t.get("type"), t.get("name", "?")))
        out.append((t.get("name", "?"),
                    tuple(int(d) for d in t.get("shape", [])), tt, None))
    return out


def id_account(tensors, data_bytes=None, align=32):
    """({type name: [tensors, elements, bytes]}, elements, bytes),
    priced two ways when the file is at hand. Method one is type
    arithmetic: elements over the block size times the block bytes.
    Method two is the header's own offsets: each tensor must end
    within one alignment unit of the next offset, the last within one
    unit of the data section end. Both landed on the same byte total
    on both local files (zero padding); a mismatch raises, because a
    wrong triple or a misread table must never print a number."""
    hist, elems, total, priced = {}, 0, 0, []
    for name, dims, tt, off in tensors:
        if tt not in GGML_TENSOR_TYPES:
            raise ValueError("unknown ggml type {} on {}".format(tt, name))
        tname, tsize, blck = GGML_TENSOR_TYPES[tt]
        n = 1
        for d in dims:
            n *= int(d)
        if not n or n % blck:
            raise ValueError("{} elements do not fill {} blocks"
                             .format(name, tname))
        b = n // blck * tsize
        h = hist.setdefault(tname, [0, 0, 0])
        h[0] += 1
        h[1] += n
        h[2] += b
        elems += n
        total += b
        priced.append((name, off, b))
    if not elems:
        raise ValueError("the tensor table is empty")
    if data_bytes is not None:
        priced.sort(key=lambda t: t[1])
        for i, (name, off, b) in enumerate(priced):
            nxt = priced[i + 1][1] if i + 1 < len(priced) else data_bytes
            if not (off + b <= nxt < off + b + align):
                raise ValueError("offset audit failed at {}: the typed "
                                 "size does not meet the next offset"
                                 .format(name))
    return hist, elems, total


def id_experts(meta, tensors, elems):
    """(used, count, active elements) or None on a dense model. An
    expert bank is any tensor whose slowest dimension equals
    expert_count: on the local 35B that selects exactly the
    ffn_{down,gate,up}_exps banks, and the api mirror reports the
    same dimension order, so one rule serves both sources."""
    count = _arch_get(meta, "expert_count")
    if not count:
        return None
    used = int(_arch_get(meta, "expert_used_count") or 0)
    bank = 0
    for name, dims, tt, off in tensors:
        if len(dims) >= 3 and int(dims[-1]) == int(count):
            n = 1
            for d in dims:
                n *= int(d)
            bank += n
    active = elems - bank + bank * used // int(count)
    return used, int(count), active


def id_walk_file(path):
    """(metadata, tensor descriptors, priced account) for a gguf on
    disk. The one place a file gets walked and priced, so the identity
    card and the share formats can never disagree about a model's
    effective bits per weight. Raises the same errors id_account does
    rather than returning a partial answer."""
    with open(path, "rb") as f:
        meta = gguf_meta_stream(f)
        tensors, hdr_end = gguf_tensor_table(f, meta.get("__tensor_count", 0))
    align = int(meta.get("general.alignment") or 32)
    data_start = (hdr_end + align - 1) // align * align
    acct = id_account(tensors, os.path.getsize(path) - data_start, align)
    return meta, tensors, acct


def id_claim(recipe, name):
    """What the model says it is before any walking: the declared
    recipe name against the quant token the file or tag name carries.
    Both are claims; the table walk is what checks them."""
    m = re.findall(r"(?i)\b(?:[it]?q\d[0-9a-z_]*|bf16|f16|f32|mxfp4|"
                   r"nvfp4)\b", name)
    token = max(m, key=len) if m else None
    if recipe and token:
        if token.upper() == recipe.upper():
            return "{} (general.file_type; the name agrees)".format(recipe)
        return "{} in general.file_type, but the name says {}".format(
            recipe, token)
    if recipe:
        return "{} (general.file_type; no quant token in the name)" \
            .format(recipe)
    if token:
        return "{} from the name only; no general.file_type".format(token)
    return "none: no general.file_type, no quant token in the name"


def id_kv_note(cache):
    """The kv axis only ever cites a dtype this machine has measured:
    a run's concrete K/V marker lands in the per-model cache, and with
    none on file the card says not measured instead of assuming f16."""
    kt = (cache or {}).get("kv_types")
    if kt:
        source = cache.get("kv_source") or "runtime log"
        control = "OLLAMA_KV_CACHE_TYPE changes it for Ollama" \
            if source == "Ollama runner log" else \
            "-ctk / -ctv move it per llama.cpp run"
        return ("a runtime choice, not in the model. K {}, V {} on the "
                "last measured run for this model and engine ({}, {}, "
                "via {}); {}".format(
                    kt[0], kt[1], cache.get("model_name", "?"),
                    str(cache.get("kv_stamp") or
                        cache.get("stamp", "?"))[:10], source, control))
    return ("a runtime choice, not in the model, and no measured run "
            "for this model and engine has recorded it yet. Measure once "
            "(python3 picchio.py MODEL) and the card cites that run; "
            "a local Ollama run needs readable runner logs")


def _id_wrap(label, text):
    return textwrap.wrap(text, width=WIDTH,
                         initial_indent="  " + label.ljust(11),
                         subsequent_indent=" " * 13)


def render_id(name, claim, acct, moe, kv_note, audit_note,
              file_note=None, source_note=None):
    """The identity card: which bytes, who says they made them, what
    they claim to be, then the walked mixture, effective bits per
    weight, and the axes the file cannot carry. Information card
    contract, same as plan: no fixed line budget, no mp1 footer.

    The file and source lines come first because they answer the
    question the rest of the card assumes: two people comparing decode
    rates for "the same Q4_K_M" have to establish that it is the same
    file before any of the numbers below mean anything."""
    out = ["picchio id: " + name]
    if file_note:
        out += _id_wrap("file", file_note)
    if source_note:
        out += _id_wrap("source", source_note)
    out += _id_wrap("claimed", claim)
    if not acct:
        out += _id_wrap("walked", "nothing: {}. The per tensor mix "
                        "lives in the table itself; point id at the "
                        ".gguf file for the walk.".format(audit_note))
        out += _id_wrap("kv cache", kv_note)
        return "\n".join(out)
    hist, elems, total = acct
    out += _id_wrap("walked", "{} tensors, {} types, priced one by "
                    "one:".format(sum(h[0] for h in hist.values()),
                                  len(hist)))
    for tname in sorted(hist, key=lambda k: -hist[k][2]):
        c, n, b = hist[tname]
        out.append("    {:<8}{:>5} tensors {:>6.2f} bits {:>6.1f}% of "
                   "weight bytes".format(tname, c, b * 8.0 / n,
                                         100.0 * b / total))
    out += _id_wrap("effective", "{:.2f} bits per weight: {:,} tensor "
                    "bytes over {:,} weights; {}".format(
                        total * 8.0 / elems, total, elems, audit_note))
    out += _id_wrap("kv cache", kv_note)
    if moe:
        used, count, active = moe
        out += _id_wrap("experts", "{} of {} wake per token: about "
                        "{:.1f}B of the {:.1f}B weights are read for "
                        "any one token".format(used, count,
                                               active / 1e9,
                                               elems / 1e9))
    return "\n".join(out)


def id_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio id MODEL\n"
              "split the quant label into the three axes it hides: the\n"
              "per tensor type mix priced into one effective bits per\n"
              "weight figure (walked from the gguf tensor table, offsets\n"
              "audited), the kv cache dtype (a runtime choice, cited\n"
              "only from a run measured here), and how many experts\n"
              "wake per token on a mixture of experts. A .gguf path is\n"
              "walked directly, an ollama tag through the api's mirror\n"
              "of the same table. Read only, never a verdict.")
        sys.exit(0)
    if len(argv) != 1:
        sys.exit("picchio id: usage: picchio id MODEL (a .gguf path "
                 "or an ollama tag)")
    arg = argv[0]
    if os.path.isfile(arg):
        kv_note = id_kv_note(cache_for_measurement(
            load_cache(), measurement_key("llama.cpp", arg), arg))
        name = os.path.basename(arg)
        try:
            meta, tensors, acct = id_walk_file(arg)
        except (ValueError, struct.error, OSError) as e:
            sys.exit("picchio id: {}: {}".format(name, e))
        claim = id_claim(
            LLAMA_FTYPES.get(meta.get("general.file_type")), name)
        # after the header parsed: no point reading 22 GiB of something
        # that turned out not to be a gguf at all
        print(render_id(name, claim, acct,
                        id_experts(meta, tensors, acct[1]), kv_note,
                        "the header's own offsets audit to the same "
                        "byte total",
                        id_file_note(arg, sys.stderr.write),
                        gguf_source_note(meta)))
        sys.exit(0)
    if not looks_like_tag(arg):
        sys.exit("picchio id: no such file: {}".format(arg))
    if not ollama_reachable():
        sys.exit("picchio id: {!r} looks like an ollama tag, but no "
                 "ollama answered at {}.".format(arg, OLLAMA_HOST))
    try:
        show = ollama_api("/api/show", {"model": arg}, timeout=15)
    except (urllib.error.URLError, OSError, ValueError):
        sys.exit("picchio id: ollama at {} does not know the model "
                 "{!r}.".format(OLLAMA_HOST, arg))
    kv_note = id_kv_note(cache_for_measurement(
        load_cache(), measurement_key("ollama", arg), arg))
    mi = show.get("model_info") or {}
    recipe = LLAMA_FTYPES.get(mi.get("general.file_type")) \
        or (show.get("details") or {}).get("quantization_level")
    claim = id_claim(recipe, arg)
    # the api reports sizes, not paths, so the blob behind the tag comes
    # from the manifest on disk; a remote ollama has none here to read
    local = ollama_host_is_local()
    blob = ollama_model_path(arg) if local else None
    file_note = id_file_note(blob, sys.stderr.write)
    if not file_note:
        file_note = "not recorded: " + (
            "no local blob for this tag under ~/.ollama/models"
            if local else
            "this ollama serves from {}, whose blobs are not on this "
            "disk".format(OLLAMA_HOST))
    source_note = gguf_source_note(mi)
    try:
        if not show.get("tensors"):
            raise ValueError("this ollama api answered without a "
                             "tensors field")
        tensors = ollama_tensor_table(show["tensors"])
        acct = id_account(tensors)
    except ValueError as e:
        print(render_id(arg, claim, None, None, kv_note, str(e),
                        file_note, source_note))
        sys.exit(0)
    print(render_id(arg, claim, acct, id_experts(mi, tensors, acct[1]),
                    kv_note, "typed shapes from the api, which mirrors "
                    "the table without offsets, so no offset audit",
                    file_note, source_note))
    sys.exit(0)


# ------------------------------------------------------------------ share
#
# picchio share reformats a block that already exists. It measures
# nothing and it decides nothing; every value it prints was read out of
# the block it was handed, or walked out of the model file that block
# names. It exists because the full block is right for a bug
# report and heavy for a comment, and the argument about that on
# r/LocalLLaMA ended in a standoff: one side hand pasting llama-bench
# tables, the other saying that if posting needs a paper they will not
# post. Nobody offered a third option. These are the three options.
#
# The skeleton fills in what can be measured and leaves the opinion
# blank. A tool that writes your conclusion for you is a tool that has
# started deciding what your numbers mean.
#
# What lives here is the gathering: the gguf walk, the cache lookup and
# the argument parsing. How any of it reads is picchio_core.share.


def share_identity(model, emit=None):
    """quant, effective bits per weight and file identity for a model,
    from the same walk the id card does. Read only, and not a
    measurement: nothing here starts an engine. Fields a source cannot
    answer come back absent rather than guessed."""
    if not model:
        return {}
    if os.path.isfile(model):
        try:
            meta, tensors, acct = id_walk_file(model)
        except (ValueError, struct.error, OSError) as e:
            if emit:
                emit("picchio share: {} could not be walked: {}\n".format(
                    os.path.basename(model), e))
            return {}
        hist, elems, total = acct
        digest, nbytes = file_fingerprint(model, emit)
        return {"quant": LLAMA_FTYPES.get(meta.get("general.file_type")),
                "bpw": total * 8.0 / elems, "mode": "llama.cpp",
                "sha256": digest,
                "bytes": nbytes if digest else None}
    if not looks_like_tag(model) or not ollama_reachable():
        return {}
    try:
        show = ollama_api("/api/show", {"model": model}, timeout=15)
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    mi = show.get("model_info") or {}
    out = {"quant": LLAMA_FTYPES.get(mi.get("general.file_type"))
           or (show.get("details") or {}).get("quantization_level"),
           "mode": "ollama"}
    if ollama_host_is_local():
        digest, nbytes = file_fingerprint(ollama_model_path(model), emit)
        if digest:
            out["sha256"], out["bytes"] = digest, nbytes
    try:
        _h, elems, total = id_account(ollama_tensor_table(
            show.get("tensors") or []))
        out["bpw"] = total * 8.0 / elems
    except ValueError:
        pass
    return out


def share_facts(b, model=None, emit=None):
    """Every field the three formats print, gathered once so they
    cannot disagree with each other. The block supplies what a run
    measured; the model file supplies what a file is. Anything neither
    can answer is n/a, which is a reading, not a blank."""
    ident = share_identity(model, emit)
    kv = None
    if model and ident.get("mode"):
        rec = cache_for_measurement(
            load_cache(), measurement_key(ident["mode"], model), model)
        if rec and rec.get("kv_types"):
            kv = "/".join(rec["kv_types"])
    pp, tg, wall = (b.get("rates") or [None] * 3)[:3]
    return {
        "model": b.get("model"),
        # the file's own recipe name beats the quant token guessed out
        # of a filename, and disagreement between them is exactly what
        # the id card exists to surface
        "quant": ident.get("quant") or b.get("quant"),
        "bpw": ident.get("bpw"),
        "sha256": ident.get("sha256"),
        "bytes": ident.get("bytes"),
        "engine": b.get("engine"),
        "machine": "{}, {} GB".format(b["chip"], b["ram"])
        if b.get("chip") else None,
        "ctx": b.get("ctx"),
        "kv": kv,
        "settings": b.get("settings"),
        "pp": pp, "tg": tg, "wallclock": wall,
        "verdict": b.get("verdict"),
        "place": b.get("place"),
        "os_raw": b.get("os_raw"),
        "protocol": b.get("protocol"),
        "row": b.get("row"),
    }


def warn_share_missing(facts, model):
    miss = share_missing(facts)
    if miss:
        # stdout stays exactly the artifact; the coaching goes where the
        # rest of picchio's human output goes
        sys.stderr.write(
            "picchio share: {} came back n/a{}\n".format(
                ", ".join(miss),
                "; run with --model MODEL to fill the file side"
                if not model else ""))


def share_cli(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("usage: picchio share [BLOCK] [--model MODEL]\n"
              "                        [--line | --row | --post]\n"
              "reformat a verdict block you already have into something\n"
              "postable. --line is one line for a comment, --row is a\n"
              "markdown table row, --post is a full post skeleton with\n"
              "the specs filled in and the opinion left blank. BLOCK is\n"
              "a saved block or stdin. --model points at the .gguf or\n"
              "ollama tag the block measured, which adds the effective\n"
              "bits per weight, the file identity and the kv dtype; the\n"
              "file is walked and hashed, never run. Nothing here is\n"
              "measured and no field is filled in by guessing.")
        sys.exit(0)
    mode, model, path = "line", None, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--line", "--row", "--post"):
            mode = a[2:]
        elif a == "--model" and i + 1 < len(argv):
            i += 1
            model = argv[i]
        elif a.startswith("-") and a != "-":
            sys.exit("picchio share: unknown option {}".format(a))
        elif path is None:
            path = a
        else:
            sys.exit("picchio share: one block at a time")
        i += 1
    try:
        text = sys.stdin.read() if path in (None, "-") \
            else open(path).read()
    except OSError as e:
        sys.exit("picchio share: {}".format(e))
    b = parse_block(text)
    if not b:
        sys.exit("picchio share: no picchio block found in {}. Save one "
                 "with: {} MODEL > block.txt".format(
                     path or "stdin", invocation()))
    f = share_facts(b, model, sys.stderr.write)
    print(render_share(f, text, mode))
    warn_share_missing(f, model)
    sys.exit(0)


# ---------------------------------------------------------------- selftest

def selftest():
    """Run Picchio's bundled parser and behavior checks."""
    here = os.path.dirname(os.path.abspath(__file__))
    rawroot = os.path.join(here, "examples", "raw")
    if not os.path.isdir(rawroot):
        # The public download carries pure logic checks that need no model,
        # GPU, network connection or fixture directory.
        flat = [{"t": i * 0.25, "dev": d, "mem": 0, "gpu_w": 0.0}
                for i, d in enumerate([1, 0, 2, 1])]
        cpu = dict(blank_pass(), offload_n=0, offload_total=33,
                   prefill_toks=27.0, decode_toks=12.0)
        # a five digit prefill packs the lane column down to one space
        # after the label, and a model sized memory step is residency
        # evidence the offline judge has to honor. Both came off a real
        # RTX 5090 box; both ship in the single file build because a
        # curl'd picchio has no fixtures to replay.
        tight_warm = parse_block(
            "model    test, 1.0 B, Q4_K_M, 1.0 GiB, llama.cpp b1\n"
            "gpu      ENGAGED: 1/1 layers on GPU\n"
            "os       gpu idle 0%, work 99%, mem +1.0 GiB\n"
            "ctx 4096         prefill         decode      wallclock\n"
            "  cold         2.0 tok/s      1.0 tok/s      0.5 tok/s\n"
            "  warm mid 100.0 tok/s      5.0 tok/s      4.0 tok/s\n"
            "VERDICT: HEALTHY. synthetic\n")
        resident = parse_block(
            "model    test, 1.0 B, Q4_K_M, 5.0 GiB, llama.cpp b1\n"
            "gpu      ENGAGED: 1/1 layers on GPU\n"
            "os       gpu idle 0%, work 0%, mem +5.0 GiB\n"
            "ctx 4096         prefill         decode      wallclock\n"
            "  warm mid  100.0 tok/s      5.0 tok/s      4.0 tok/s\n"
            "VERDICT: HEALTHY. synthetic\n")
        checks = [
            monitor_classify(588.0, 21.1)[0] == "OK",
            monitor_classify(26.8, 12.2)[0] == "FLAG",
            monitor_classify(226.0, 15.0, baseline=60.0)[0] == "FLAG",
            monitor_target_mode("qwen3.5:9b") == "ollama",
            monitor_target_mode("http://x:8080") == "server",
            "rerun idle" in decode_advice(19.6, 10.3),
            decode_advice(20.0, 21.1).startswith(" Quote"),
            watch_verdict(watch_summary(flat), None)[0] == "GPU IDLE",
            parse_engine_version("version: 9430 (d48a56ef)") == "b9430",
            diagnose(cpu, cpu, "llama.cpp")[0] == "SILENT CPU FALLBACK",
            tight_warm["row"] == "warm mid"
            and tight_warm["rates"] == [100.0, 5.0, 4.0],
            verify_block(resident)[0] == "PASS",
            # the runaway guard: a flood is capped and killed, not buffered
            _run_capped([sys.executable, "-c", "print('x' * 2000000)"],
                        10, 500000).capped,
        ]
        n_ok = sum(1 for c in checks if c)
        from picchio_core.selftest import run_selftests
        core_ok, core_all, core_failures = run_selftests(
            [sys.executable, os.path.abspath(sys.argv[0])])
        print("Picchio selftest: logic {}/{}, extended {}/{}".format(
            n_ok, len(checks), core_ok, core_all))
        if core_failures:
            print("failures: " + ", ".join(core_failures))
        sys.exit(0 if n_ok == len(checks) and core_ok == core_all else 1)
    fx_ok = fx_all = rp_ok = rp_all = 0
    for name in sorted(os.listdir(rawroot)):
        d = os.path.join(rawroot, name)
        if not os.path.isdir(d):
            continue
        passes, metas = [], []
        for i in range(1, 32):
            stderr_p = os.path.join(d, "pass{}.stderr.txt".format(i))
            resp_p = os.path.join(d, "pass{}.response.json".format(i))
            meta_p = os.path.join(d, "pass{}.meta.json".format(i))
            if not os.path.exists(meta_p):
                break
            meta = json.load(open(meta_p))
            metas.append(meta)
            fx_all += 1
            if os.path.exists(stderr_p):
                p = parse_stderr(open(stderr_p).read(), meta["wall_s"])
                mode = "llama.cpp"
            elif os.path.exists(resp_p) and meta.get("mode") == "server":
                p = map_server(json.load(open(resp_p)), meta["wall_s"])
                mode = "server"
            elif os.path.exists(resp_p):
                p = map_ollama(json.load(open(resp_p)), meta["wall_s"],
                               meta.get("ps"))
                mode = "ollama"
            else:
                break
            # the artifact answers the same question its live run did
            p["nonce"] = meta.get("prompt_nonce")
            if p["prefill_toks"] and p["decode_toks"] and p["wallclock_toks"]:
                fx_ok += 1
            passes.append(p)
        if not passes:
            continue
        rp_all += 1
        txt_p = os.path.join(here, "examples", name + ".txt")
        want = open(txt_p).read().rstrip().splitlines()
        l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
        cold_note = (l1 is not None and l2 is not None
                     and l1 < 2 * l2 + 500)
        tele = None  # raw dirs that predate the sampler have no curve
        tj = os.path.join(d, "telemetry.json")
        if os.path.exists(tj):
            raw = json.load(open(tj))
            tele = raw.get("summary")
            if raw.get("samples") and raw.get("marks"):
                # the committed curve is the evidence and the summary
                # beside it is one derivation of that curve, so the
                # replay redoes the derivation instead of trusting a
                # cached one. Every stored field reproduces exactly;
                # what this buys is that a new figure computed from the
                # same samples appears in the replay too, rather than
                # needing the fixtures recaptured on a machine with a
                # gpu meter. throttled and src are sampler state, not
                # derivations, so they are read, not recomputed.
                tele = telemetry_summary(
                    raw["samples"], raw["marks"],
                    (tele or {}).get("throttled", False),
                    (tele or {}).get("src"))
        rep = build_rep(passes)
        state, para = diagnose(passes[0], rep, mode, tele)
        extra = metas[0].get("extra_args", [])
        why = attribute_why(state, rep, mode, extra)
        got = render_verdict(
            machine_info(), metas[0].get("engine", "?"),
            metas[0].get("model_name", "?"), passes, state, para, mode,
            None, cold_note, why, metas[0].get("ctx", effective_ctx(extra)),
            extra, tele).splitlines()
        if got[:-1] == want[:-1]:
            rp_ok += 1
        else:
            for a, b in zip(want, got):
                if a != b:
                    print("  {} mismatch:\n    want: {}\n    got:  {}".format(
                        name, a, b))
                    break
    # compare: the two committed llama.cpp blocks are a natural pair
    cp_ok, cp_all = 0, 4
    ha = open(os.path.join(here, "examples", "healthy-metal.txt")).read()
    fb = open(os.path.join(here, "examples", "cpu-fallback.txt")).read()
    pa = parse_block("someone posted this:\n" + ha + "\nhope it helps")
    pb = parse_block(fb)
    if pa and pb and pa["ctx"] == CTX \
            and pb["args"] == "--device none -ngl 0":
        cp_ok += 1
    two = render_compare(("A", "B"), pa, pb)
    if "SUSPECT: placement" in two and "0/33" in two:
        cp_ok += 1
    if "Nothing to compare" in render_compare(("A", "A"), pa,
                                              parse_block(ha)):
        cp_ok += 1
    # old format: strip the two fingerprint fields and the committed
    # block is byte for byte the pre-fingerprint output (they are the
    # only format change since); it must parse as unknown, not a guess
    old = re.sub(r"(?m)^(ctx \d+)", lambda m: " " * len(m.group(1)), fb)
    po = parse_block(re.sub(r"(?m)^(gpu\s{6}.*) \[.*\]$", r"\1", old))
    if po and po["ctx"] is None and po["args"] is None \
            and po["frac"] == 0.0 \
            and "unknown" in render_compare(("A", "B"), pa, po):
        cp_ok += 1
    # telemetry: synthetic timelines pushed through the real window
    # math and the real three source judge (no gpu needed, ci safe)
    te_ok, te_all = 0, 9
    gib = 1024 ** 3

    def synth_tele(idle_dev, work_dev, mem_base, mem_peak, src=None,
                   idle_w=0.02):
        # one 11.2 s pass after a 1.2 s baseline, ticked at 4 Hz; busy
        # samples land exactly in the tail aligned compute window
        t_end, wall, load_s, prompt_s, eval_s = 12.4, 11.2, 2.0, 1.3, 6.3
        dec1 = t_end - TELE_PAD_S
        pre0 = dec1 - eval_s - prompt_s
        t0, samples, t = t_end - wall, [], 0.0
        while t < t_end + 0.5:
            in_work = pre0 <= t <= dec1
            samples.append({
                "t": t,
                "dev": work_dev if in_work else idle_dev,
                "gpu_w": 10.6 if in_work and work_dev >= 50 else idle_w,
                "mem": mem_peak if t >= t0 + load_s else mem_base})
            t += 0.25
        return telemetry_summary(samples, [{
            "t_end": t_end, "wall_s": wall, "load_s": load_s,
            "prompt_s": prompt_s, "eval_s": eval_s}], src=src)

    fx = blank_pass()
    fx.update(offload_n=33, offload_total=33, prefill_toks=558.9,
              decode_toks=20.0, model_bytes=int(5.28 * gib))
    busy = synth_tele(0, 99, 600 * 1024 ** 2, int(7.0 * gib))
    flat = synth_tele(0, 0, 600 * 1024 ** 2, 700 * 1024 ** 2)
    # 1: gpu busy aligned with the compute window backs the full claim
    if telemetry_vote(busy, fx, "llama.cpp") == "agree" \
            and diagnose(fx, fx, "llama.cpp", busy)[0] == "HEALTHY" \
            and "mem +6.4 GiB, 10.6 W" in os_line(busy):
        te_ok += 1
    # 2: a flat line under a full offload claim is a two source fight;
    #    the WHY ladder stays out (the block itself is the exhibit)
    st, para = diagnose(fx, fx, "llama.cpp", flat)
    if st == "CONFLICTING EVIDENCE" and "stay flat" in para \
            and attribute_why(st, fx, "llama.cpp", []) is None:
        te_ok += 1
    # 3: a busy desktop disqualifies whole-gpu numbers: abstain, say so
    lifted = synth_tele(47, 99, 600 * 1024 ** 2, int(7.0 * gib))
    if telemetry_vote(lifted, fx, "llama.cpp") == "abstain" \
            and diagnose(fx, fx, "llama.cpp", lifted)[0] == "HEALTHY" \
            and "not judged" in os_line(lifted):
        te_ok += 1
    # 7: the shape that started this: a Mac desktop drawing at 30%
    #    utilization and 0.4 W before pass 1. Utilization alone called
    #    that busy and threw away the whole os vote, energy included.
    #    Watts say it never computed, so the run is judged, and the
    #    block carries the sentence that says why it was judged.
    desk = synth_tele(30, 99, 600 * 1024 ** 2, int(7.0 * gib),
                      src="ioreg", idle_w=0.4)
    note = pre_run_idle(desk)[1]
    # the paragraph is wrapped, so the block is flattened before the
    # sentence is looked for: that checks it survived wrapping whole
    # rather than that some fragment of it is in there somewhere
    blk = " ".join(render_verdict(
        machine_info(), "llama.cpp b1", "m", [fx, fx, fx], "HEALTHY",
        "Placement checks out.", "llama.cpp", None, False, None,
        CTX, [], desk).split())
    if telemetry_vote(desk, fx, "llama.cpp") == "agree" \
            and "idle 30%, work 99%" in os_line(desk, fx) \
            and "0.53 J/tok" in os_line(desk, fx) \
            and note and "30% at 0.4 W" in note and note in blk:
        te_ok += 1
    # 8: the negative control on the same meter. Same utilization, real
    #    power behind it, so something was computing and none of it can
    #    be pinned on this run. Still not idle, and the line now says
    #    which of the two signals disqualified it.
    hot = synth_tele(50, 99, 600 * 1024 ** 2, int(7.0 * gib),
                     src="ioreg", idle_w=12.0)
    if telemetry_vote(hot, fx, "llama.cpp") == "abstain" \
            and pre_run_idle(hot) == (False, None) \
            and "50% at 12.0 W before the run, not idle" in os_line(hot) \
            and telemetry_read(hot) is None:
        te_ok += 1
    # 9: no meter borrows another's watts. A discrete card has its own
    #    idle draw and nobody has measured one yet, so low watts there
    #    rescue nothing and the utilization answer stands.
    other = synth_tele(31, 99, 600 * 1024 ** 2, int(7.0 * gib),
                       src="nvml", idle_w=0.4)
    if pre_run_idle(other) == (False, None) \
            and telemetry_vote(other, fx, "llama.cpp") == "abstain" \
            and IDLE_W_GATE.get("nvml") is None \
            and pre_run_idle(dict(desk, src=None)) == (False, None):
        te_ok += 1
    # 4: the memory step vetoes the flat line contradiction
    stepped = synth_tele(0, 0, 600 * 1024 ** 2, int(6.4 * gib))
    if telemetry_vote(stepped, fx, "llama.cpp") == "abstain":
        te_ok += 1
    # 5: timing physics alone still catches a cpu shaped full claim
    slow = dict(fx, prefill_toks=28.3, decode_toks=12.0)
    st, para = diagnose(slow, slow, "llama.cpp")
    if st == "CONFLICTING EVIDENCE" and "CPU shaped" in para \
            and attribute_why(st, slow, "llama.cpp", []) is None:
        te_ok += 1
    # 6: whatever the reason for having no meter, the os line fits the
    # block. This one shipped broken twice unnoticed, so it is checked
    # rather than remembered: the reason text comes from whichever
    # backend failed, and backends get added.
    if all(len("os       " + os_line({"off": off, "ev": "engine+timing"}))
           <= WIDTH
           for off in ("no nvml", "no nvml/amdgpu", "disabled", "not macos",
                       "no ioreg", "no ioreg data", "no amdgpu data",
                       "a reason far longer than the block can ever hold")):
        te_ok += 1
    # verify: the two committed blocks pass, and blocks tampered by one
    # edit fail. Fixtures are built in memory from the real examples, so
    # no forged block ships in the repo; ha and fb are read above.
    ve_ok, ve_all = 0, 7
    if verify_block(parse_block(ha))[0] == "PASS":
        ve_ok += 1
    if verify_block(parse_block(fb))[0] == "PASS":
        ve_ok += 1
    # a block whose whole prefill lane abstains is one picchio printed
    # itself, so it has to read back. It answered "no verdict block
    # found" until the parser learned the third cell word, which reads
    # to a user as "your paste is wrong" about the tool's own output.
    # The committed cached-ollama block is the exhibit; prefill parks as
    # None, the other two lanes survive, and verify judges on those.
    ab = open(os.path.join(here, "examples",
                           "linux-5090-ollama.txt")).read()
    pab = parse_block(ab)
    if pab and pab["row"] == "warm mid" and pab["rates"][0] is None \
            and pab["rates"][1] and pab["rates"][2] \
            and verify_block(pab)[0] == "PASS":
        ve_ok += 1
    # flip the cpu-fallback block's placement line to claim the full gpu:
    # one edit, and three independent witnesses (the ratio, the os meter,
    # the headline) each catch the run's real cpu shape underneath
    forged = re.sub(r"gpu      NOT ENGAGED: 0/33 layers on GPU \[.*\]",
                    "gpu      ENGAGED: 33/33 layers on GPU (Metal: Apple M5)",
                    fb)
    fv, ff = verify_block(parse_block(forged))
    if fv == "FLAG" and any("cpu shaped" in x for x in ff) and len(ff) >= 3:
        ve_ok += 1
    # invert a lane so decode reads faster than prefill: pure physics,
    # impossible on one run, caught with no hardware knowledge at all
    inv = re.sub(r"(warm mid\s+)588\.0 tok/s(\s+)21\.1 tok/s",
                 r"\g<1>15.0 tok/s\g<2>21.1 tok/s", ha)
    iv, iff = verify_block(parse_block(inv))
    if iv == "FLAG" and any("outrun" in x for x in iff):
        ve_ok += 1
    # 5: a packed warm label leaves one separator before the first
    #    number. It must still outrank the cold row, or verify silently
    #    grades a five digit run on its cold pass and calls it a lie.
    tight = re.sub(r"(?m)^(\s{2}warm mid)\s{2,}", r"\1 ", ha)
    tp = parse_block(tight)
    if tp and tp["row"] == "warm mid" \
            and tp["rates"] == parse_block(ha)["rates"]:
        ve_ok += 1
    # 6: the live judge abstains on a flat utilization median when a
    #    model sized memory step proves the weights landed. Offline
    #    verification casts the same vote; a bursty gpu whose median
    #    reads 0 is not thereby a forgery.
    resident = re.sub(r"(?m)^(os\s+gpu .*work )\d+(%.*)$",
                      r"\g<1>0\g<2>", ha)
    rb = parse_block(resident)
    if rb and os_residency_witness(rb) \
            and verify_block(rb)[0] == "PASS":
        ve_ok += 1
    # watch: five required synthetic paths through the stable JSON contract
    # and the real machine-level judge (no gpu needed, ci safe)
    wa_ok, wa_all = 0, 5

    def synth_watch(dev_seq, mem, watt):
        return [{"t": i * 0.25, "dev": d, "mem": mem, "gpu_w": watt}
                for i, d in enumerate(dev_seq)]

    target = {"pid": 1, "name": "runner", "engine": None}
    watch_busy = watch_summary(synth_watch(
        [0, 98, 99, 97, 99, 98], int(6.5 * gib), 12.0))
    sb, pb = watch_verdict(watch_busy, "runner (pid 1)")
    jb = watch_json(target, watch_busy, sb, 0, 0, 2, "duration_elapsed")
    # 1 busy: JSON parses, verdict/exit agree, whole-gpu attribution stays
    if json.loads(json.dumps(jb))["exitCode"] == 0 \
            and sb == "GPU BUSY" and "machine level" in pb \
            and jb["attribution"] == "whole_gpu":
        wa_ok += 1
    watch_idle = watch_summary(synth_watch(
        [1, 2, 0, 3, 1, 2], 600 * 1024 ** 2, 0.03))
    si, pi = watch_verdict(watch_idle, "runner (pid 1)")
    # 2 idle: fallback exit 4 is in the artifact
    if si == "GPU IDLE" and "on the cpu" in pi \
            and watch_json(target, watch_idle, si, 4, 0, 2,
                           "duration_elapsed")["exitCode"] == 4:
        wa_ok += 1
    # 3 unavailable: null is preserved in raw+summary and warnings name it
    watch_missing = watch_summary(synth_watch([80, 90, 95], None, None))
    jm = watch_json(target, watch_missing, "GPU BUSY", 0, 0, 1,
                    "duration_elapsed")
    raw_missing = watch_sample_json(synth_watch([80], None, None)[0], 0.0)
    if jm["gpu"]["powerWatts"]["median"] is None \
            and raw_missing["gpuMemoryBytes"] is None \
            and any("power" in x for x in jm["warnings"]):
        wa_ok += 1
    # 4 target exit and 5 SIGINT both still close a complete summary
    if watch_json(target, watch_busy, sb, 0, 0, 1,
                  "target_exited")["stopReason"] == "target_exited":
        wa_ok += 1
    if watch_json(target, watch_busy, sb, 0, 0, 1,
                  "interrupted")["stopReason"] == "interrupted":
        wa_ok += 1
    # monitor: the per probe signature classifier and the session summary,
    # both pure, so ci needs no live server
    mo_ok, mo_all = 0, 8
    # a gpu shaped probe reads OK, a cpu shaped one FLAG (the same 5x/15x
    # lines the server block uses), a missing rate convicts nobody
    if monitor_classify(588.0, 21.1)[0] == "OK" \
            and monitor_classify(26.8, 12.2)[0] == "FLAG" \
            and monitor_classify(None, 21.0)[0] == "NODATA":
        mo_ok += 1
    # the unsure band is neither: a prefill slow but not cpu slow
    if monitor_classify(180.0, 20.0)[0] == "WATCH":
        mo_ok += 1
    # a session that flipped gpu->cpu->gpu counts two transitions, convicts
    # on the one cpu probe, and keeps that probe's ratio as the worst
    flap = monitor_summarize([("OK", 27.0), ("OK", 26.0), ("FLAG", 2.2),
                              ("OK", 25.0)])
    if flap["flag"] == 1 and flap["transitions"] == 2 \
            and abs(flap["worst_ratio"] - 2.2) < 1e-9:
        mo_ok += 1
    # an all healthy session names no fallback (the exit 0 shape)
    steady = monitor_summarize([("OK", 27.0), ("OK", 26.5), ("OK", 28.1)])
    if steady["flag"] == 0 and steady["transitions"] == 0 \
            and "ENGAGED throughout" in monitor_summary_line(steady):
        mo_ok += 1
    # baseline self-calibration: a many-core cpu fallback whose absolute
    # ratio still clears the fixed 15x line (a cpu run measured 15.1x on 48
    # EPYC threads) is caught relative to a 60x healthy baseline, the miss
    # the fixed floor alone would wave through
    if monitor_classify(1800.0, 30.0, baseline=60.0)[0] == "OK" \
            and monitor_classify(226.0, 15.0, baseline=60.0)[0] == "FLAG" \
            and monitor_classify(226.0, 15.0)[0] == "OK":
        mo_ok += 1
    # run-to-run noise against the baseline does not false flag; a real sag
    # into the dip band reads WATCH, not a false ENGAGED
    if monitor_classify(560.0, 24.0, baseline=27.0)[0] == "OK" \
            and monitor_classify(300.0, 20.0, baseline=27.0)[0] == "WATCH":
        mo_ok += 1
    # target detection: an http url is a server, a bare tag is ollama, a
    # file path is neither (monitor watches a running server, not a file)
    if monitor_target_mode("http://127.0.0.1:8080") == "server" \
            and monitor_target_mode("qwen3.5:9b") == "ollama" \
            and monitor_target_mode("/models/m.gguf") is None \
            and monitor_target_mode("m.gguf") is None:
        mo_ok += 1
    # the --json session artifact carries the verdict, baseline, counts and
    # the full probe timeline, shaped to paste into an issue
    mj = monitor_json("qwen3.5:9b", "ollama", "ollama 0.31.1", {"os": "x"},
                      27.0, [{"i": 1, "state": "OK", "ratio": 27.0},
                             {"i": 2, "state": "FLAG", "ratio": 2.3}],
                      monitor_summarize([("OK", 27.0), ("FLAG", 2.3)]))
    if mj["mode"] == "ollama" and mj["baseline_ratio"] == 27.0 \
            and mj["summary"]["cpu"] == 1 and len(mj["probes"]) == 2 \
            and mj["verdict"] == "SILENT CPU FALLBACK seen" \
            and mj["exit_code"] == 4:
        mo_ok += 1
    # ctx sweep: the slope sentence is exact on synthetic rows, and the
    # committed real sweep replays like a verdict block when present
    sw_ok, sw_all = 0, 2

    def row(ctx, depth, pf, dc, wc):
        return {"ctx": ctx, "depth": depth, "prefill": pf,
                "decode": dc, "wallclock": wc}

    decayed = [row(4096, 2800, 560.0, 21.0, 18.0),
               row(32768, 22000, 320.0, 15.0, 12.0)]
    if "decode fell 29%" in sweep_slope(decayed) \
            and "held within" in sweep_slope(
                [row(4096, 2800, 560.0, 21.0, 18.0),
                 row(32768, 22000, 500.0, 20.6, 17.0)]):
        sw_ok += 1
    swroot = os.path.join(here, "examples", "raw", "ctx-sweep")
    swtxt = os.path.join(here, "examples", "ctx-sweep.txt")
    if os.path.exists(os.path.join(swroot, "sweep.meta.json")) \
            and os.path.exists(swtxt):
        sm = json.load(open(os.path.join(swroot, "sweep.meta.json")))
        rows = []
        for ctx in sm["tiers"]:
            ps = []
            for i in range(1, sm["passes"] + 1):
                base = os.path.join(swroot, "ctx{}.pass{}".format(ctx, i))
                w = json.load(open(base + ".meta.json"))["wall_s"]
                if os.path.exists(base + ".stderr.txt"):
                    ps.append(parse_stderr(open(base + ".stderr.txt").read(), w))
                elif os.path.exists(base + ".response.json"):
                    ps.append(map_ollama(json.load(open(base + ".response.json")),
                                         w, None))
            if ps:
                rp = build_rep(ps)
                rows.append(row(ctx, rp.get("prompt_tokens"),
                                rp["prefill_toks"], rp["decode_toks"],
                                rp["wallclock_toks"]))
        got = render_sweep(machine_info(), sm["engine"],
                           sm["model_name"], rows).splitlines()
        want = open(swtxt).read().rstrip().splitlines()
        if got[:-1] == want[:-1]:  # footer names the replaying machine
            sw_ok += 1
    else:
        sw_all = 1  # no committed sweep yet: only the synthetic check runs
    # server endpoint judge: no engine claim exists over http, so the
    # two witnesses (os meter, speed signature) vote through the real
    # diagnose path; the synthetic telemetry timelines above are reused
    sv_ok, sv_all = 0, 5
    sfx = blank_pass()
    sfx.update(prefill_toks=560.0, decode_toks=20.0)  # 28x, gpu shaped
    st, para = diagnose(sfx, sfx, "server", busy)
    if st == "HEALTHY" and "os meter" in para and "gpu shaped" in para:
        sv_ok += 1
    cpu_fx = dict(sfx, prefill_toks=48.0, decode_toks=12.0)  # 4x, cpu
    st, para = diagnose(cpu_fx, cpu_fx, "server", flat)
    if st == "SILENT CPU FALLBACK" and "on the cpu" in para \
            and "server api" in attribute_why(st, cpu_fx, "server", []):
        sv_ok += 1
    st, para = diagnose(cpu_fx, cpu_fx, "server", busy)  # witnesses fight
    if st == "CONFLICTING EVIDENCE" and "Believe neither" in para:
        sv_ok += 1
    midr = dict(sfx, prefill_toks=200.0, decode_toks=20.0)  # 10x dead zone
    st, para = diagnose(midr, midr, "server", None)
    if st == "NO PLACEMENT EVIDENCE" and "placement is not" in para:
        sv_ok += 1
    # a warm decode that collapsed below the cold pass is contaminated:
    # placement stays HEALTHY, but the quotable number is withheld with a
    # rerun note instead of handing out a busy-machine figure
    st, para = diagnose(dict(sfx, decode_toks=20.0),
                        dict(sfx, decode_toks=10.0), "server", busy)
    if st == "HEALTHY" and "rerun idle" in para \
            and "Quote the warm median" not in para:
        sv_ok += 1
    # linux parser: the four graduated 4090 stderr shapes, each pinned
    # on the fields the diagnosis reads (all captured on b9430 CUDA and
    # cpu-only builds, driver 550.54.14)
    lx_ok, lx_all = 0, 4
    lxroot = os.path.join(here, "examples", "raw", "linux-4090")

    def lparse(fname):
        p = os.path.join(lxroot, fname)
        return parse_stderr(open(p).read(), None) if os.path.exists(p) \
            else None

    lx_h = lparse("cuda-healthy.stderr.txt")
    lx_m = lparse("misbuilt-cpu.stderr.txt")
    if lx_h and lx_h["offload_n"] == 33 and lx_h["offload_total"] == 33 \
            and lx_h["gpu_kind"] == "CUDA" \
            and lx_h["gpu_device"] == "GeForce RTX 4090" \
            and lx_h["free_mib"] == 23818:
        lx_ok += 1
    lx_z = lparse("cuda-ngl0.stderr.txt")
    if lx_z and lx_z["offload_n"] == 0 and lx_z["offload_total"] == 33 \
            and lx_z["gpu_kind"] == "CUDA":
        lx_ok += 1
    lx_p = lparse("cuda-partial.stderr.txt")
    if lx_p and lx_p["offload_n"] == 10 and lx_p["offload_total"] == 33:
        lx_ok += 1
    # the misbuilt build prints no offload line and no device line at
    # all; that absence is exactly what the silent-engine rule needs
    if lx_m and lx_m["offload_n"] is None and lx_m["gpu_kind"] is None \
            and lx_m["gpu_device"] is None and lx_m["threads"] == 48:
        lx_ok += 1
    # silent-engine: with no engine claim, an nvml flat line on an idle
    # machine convicts, and each of the five gates alone acquits
    se_ok, se_all = 0, 4
    se_fx = blank_pass()
    se_fx.update(prefill_toks=16.6, decode_toks=1.1,
                 model_bytes=int(5.28 * gib))
    se_flat = synth_tele(0, 0, 354 * 1024 ** 2, 354 * 1024 ** 2,
                         src="nvml")
    # 1: conviction, plus the memory step veto acquitting the same run
    st, para = diagnose(se_fx, se_fx, "llama.cpp", se_flat)
    se_step = synth_tele(0, 0, 354 * 1024 ** 2, int(6.0 * gib),
                         src="nvml")
    if st == "SILENT CPU FALLBACK" and "printed no gpu evidence" in para \
            and "stayed idle" in attribute_why(st, se_fx, "llama.cpp", []) \
            and diagnose(se_fx, se_fx, "llama.cpp",
                         se_step)[0] == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 2: a busy desktop abstains, no conviction on a lifted baseline
    se_busy = synth_tele(47, 47, 354 * 1024 ** 2, 354 * 1024 ** 2,
                         src="nvml")
    if diagnose(se_fx, se_fx, "llama.cpp",
                se_busy)[0] == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 3: no nvml, no upgrade: a cpu only machine keeps the old verdict,
    #    and so does the same flat line without the nvml source mark
    if diagnose(se_fx, se_fx, "llama.cpp",
                {"off": "no nvml"})[0] == "NO PLACEMENT EVIDENCE" \
            and diagnose(se_fx, se_fx, "llama.cpp",
                         synth_tele(0, 0, 1, 1))[0] \
            == "NO PLACEMENT EVIDENCE":
        se_ok += 1
    # 4: any engine claim keeps the old path: 0/33 with a flat curve is
    #    the ordinary fallback with the ladder WHY, not the silent one
    se_cl = dict(se_fx, offload_n=0, offload_total=33)
    st, para = diagnose(se_cl, se_cl, "llama.cpp", se_flat)
    if st == "SILENT CPU FALLBACK" and "printed no gpu evidence" not in para \
            and "engine log does not say" in attribute_why(
                st, se_cl, "llama.cpp", []):
        se_ok += 1
    # residency: the other half of the placement question. The three
    # measured series are the real ones from .ai/evidence-residency/;
    # RAMP and PLATEAU are synthetic and say so, because this machine
    # could not produce either (20.6 GiB of weights on 32 GiB of ram
    # means the kernel reclaims faster than a run can ramp).
    rs_ok, rs_all = 8, 8
    rs_file = int(20.61 * gib)

    def rs_rows(vals, dec, state="FLAG", vram=None, reported=None,
                ttft=24300.0):
        return [{"i": i + 1, "state": state, "rss": int(v * gib),
                 "decode": dec[i % len(dec)], "ttft_ms": ttft,
                 "vram": vram, "reported": reported}
                for i, v in enumerate(vals)]

    def rs_check(cond):
        return 0 if cond else 1

    # 1: shape is read off thirds, and level never enters it
    rs_ok -= rs_check(
        residency_shape([int(x * gib) for x in (1, 1, 1, 2, 2, 2, 3, 3, 3)])[0]
        == "RAMP"
        and residency_shape(
            [int(x * gib) for x in (1, 1, 1, 3, 3, 3, 3, 3, 3)])[0]
        == "PLATEAU"
        and residency_shape([int(20.3 * gib)] * 12)[0] == "FLAT"
        and residency_shape([int(3 * gib)] * 8)[0] == "UNDECIDED")
    # 2: measured, ollama fully loaded on metal. The trap: rss cannot see
    #    weights that live in the device pool, and a level test would call
    #    an ordinary run a streaming runtime
    dev = rs_rows([3.3] * 12, [30.0], state="OK", vram=int(20.8 * gib),
                  reported=int(20.8 * gib), ttft=1300.0)
    st, held, para = residency_verdict(dev, rs_file, "ollama")
    rs_ok -= rs_check(
        st == "WEIGHTS ON DEVICE" and held == int(20.8 * gib)
        and "30 tok/s" in para
        and residency_verdict(dev, rs_file, "server")[0] == "WEIGHTS ON DEVICE")
    # 3: measured, cpu mmap on a machine under memory pressure. No
    #    direction, so no working set, however low the level sits
    osc = rs_rows([11.5, 5.3, 9.6, 10.2, 7.5, 11.2, 9.7, 11.4, 14.9, 9.2,
                   7.0, 9.8, 7.1], [6.6, 16.8])
    st, held, para = residency_verdict(osc, rs_file, "server")
    rs_ok -= rs_check(st == "NO SETTLED WORKING SET" and "reclaim" in para)
    # 4: measured, the seven round no-warmup run. This one is the
    #    regression: an earlier cut judged on decode wobble, and a run
    #    that crawls uniformly has no wobble, so 4 tok/s with fifty
    #    second first tokens was told it "really did need less than the
    #    file". Too few rounds to read a shape is the honest answer
    few = rs_rows([5.0, 4.1, 3.5, 5.3, 3.6, 4.1, 4.5], [4.0], ttft=50600.0)
    st, held, para = residency_verdict(few, rs_file, "server")
    rs_ok -= rs_check(st == "NO SETTLED WORKING SET" and "too few" in para
                      and "4 tok/s" in para and "50.6 s" in para)
    # 5: synthetic. A bounded cache is the one shape that earns a
    #    quotable figure, and the figure is the plateau not the median
    plat = rs_rows([1.0, 1.2, 1.1, 6.0, 6.1, 6.0, 6.1, 6.0, 6.1], [20.0])
    st, held, para = residency_verdict(plat, rs_file, "server")
    rs_ok -= rs_check(st == "BOUNDED WORKING SET"
                      and abs(held - 6.1 * gib) < 0.2 * gib
                      and "stopped climbing" in para)
    # 6: synthetic. Still climbing means nothing here is quotable, even
    #    though the level would look excellent screenshotted early
    ramp = rs_rows([2.0, 2.4, 2.9, 3.4, 3.9, 4.4, 4.9, 5.4, 5.9], [20.0])
    st, held, para = residency_verdict(ramp, rs_file, "server")
    rs_ok -= rs_check(st == "NO SETTLED WORKING SET"
                      and "still climbing" in para)
    # 7: fully resident is the model itself
    cap = rs_rows([20.4] * 12, [32.0], ttft=900.0)
    st, held, para = residency_verdict(cap, rs_file, "server")
    rs_ok -= rs_check(st == "CAPACITY" and "is the model" in para)
    # 8: probes vary in content and not in length, or one mixture of
    #    experts answers every round and the curve never moves; and
    #    nothing serving the file means no pid, not a guessed one
    plen = {len(residency_prompt(i).split()) for i in range(12)}
    rs_ok -= rs_check(
        len(plen) == 1 and plen.pop() == len(BENCH_PROMPT.split())
        and len({residency_prompt(i) for i in range(10)}) == 10
        and residency_verdict(few[:RES_MIN_PROBES - 1], rs_file,
                              "server") is None
        and residency_verdict(osc, None, "server")[0]
        == "NO RESIDENCY EVIDENCE"
        and engine_pid_for(os.path.join(here, "no-such.gguf")) is None
        and engine_pid_for(None) is None)

    # locale: llama.cpp prints its numbers through the machine's locale,
    # so on a comma decimal box every timing regex used to miss and the
    # whole lane table came back n/a (issue #1). Same log, two decimal
    # conventions, one parse.
    lc_ok, lc_all = 0, 4
    lc_dot = ("llama_model_loader: - model params     = 34.89 B\n"
              "print_info: file size   = 28.53 GiB (6.56 BPW)\n"
              "llama_perf_context_print:        load time =  5321.49 ms\n"
              "llama_perf_context_print: prompt eval time =  1234.56 ms "
              "/   730 tokens\n"
              "llama_perf_context_print:        eval time = 98765.43 ms "
              "/   128 runs\n"
              "llm_load_tensors: offloaded 41/41 layers to GPU\n")
    lc_de = lc_dot.replace(".", ",").replace("41/41", "41/41")
    lc_a, lc_b = parse_stderr(lc_de, 133.6), parse_stderr(lc_dot, 133.6)
    lc_keys = ("load_ms", "prompt_ms", "eval_ms", "prompt_tokens",
               "eval_tokens", "model_params", "model_size", "model_bytes",
               "offload_n", "prefill_toks", "decode_toks")
    if all(lc_a[k] == lc_b[k] for k in lc_keys) and lc_a["load_ms"] == 5321.49:
        lc_ok += 1
    # the rightmost separator is the decimal point either way round
    if (_num("28,53"), _num("28.53"), _num("1.234,56"), _num("1,234.56"),
            _num("34"), _num("")) == (28.53, 28.53, 1234.56, 1234.56,
                                      34.0, None):
        lc_ok += 1
    if size_bytes("28,53 GiB") == size_bytes("28.53 GiB") \
            and size_bytes("28,53 GiB") and _dotted("28,53 GiB") \
            == "28.53 GiB" and _dotted("28.53 GiB") == "28.53 GiB":
        lc_ok += 1
    # the source side: the engine runs with the decimal point pinned,
    # and LC_ALL has to go because it outranks LC_NUMERIC
    lc_env = dict(os.environ)
    try:
        os.environ["LC_ALL"] = "de_DE.UTF-8"
        os.environ["LANG"] = "de_DE.UTF-8"
        ev = engine_env()
        if ev.get("LC_NUMERIC") == "C" and "LC_ALL" not in ev \
                and ev.get("LANG") == "de_DE.UTF-8":
            lc_ok += 1
    finally:
        os.environ.clear()
        os.environ.update(lc_env)

    # the timing gate: a full offload claim whose run produced no
    # numbers at all is not HEALTHY. Both cross checks below it read
    # rates, so with no rates they used to pass by absence.
    tg_ok, tg_all = 0, 4
    tg_none = dict(blank_pass(), offload_n=41, offload_total=41,
                   wall_s=133.6)
    st, para = diagnose(tg_none, tg_none, "llama.cpp")
    if st == "NO TIMING EVIDENCE" and EXIT_CODES[st] == 7 \
            and "unmeasured, not" in para:
        tg_ok += 1
    tg_real = dict(blank_pass(), offload_n=41, offload_total=41,
                   wall_s=20.0, load_ms=1700.0, prompt_ms=1200.0,
                   eval_ms=6000.0, prompt_tokens=730, eval_tokens=128)
    finish_rates(tg_real)
    if diagnose(tg_real, tg_real, "llama.cpp")[0] == "HEALTHY":
        tg_ok += 1
    # an engine that confessed still outranks the missing numbers: the
    # gate only catches a claim nothing measured, never a confession
    tg_cpu = dict(blank_pass(), offload_n=0, offload_total=33, wall_s=99.0)
    tg_part = dict(blank_pass(), offload_n=10, offload_total=33, wall_s=99.0)
    if diagnose(tg_cpu, tg_cpu, "llama.cpp")[0] == "SILENT CPU FALLBACK" \
            and diagnose(tg_part, tg_part,
                         "llama.cpp")[0] == "PARTIAL OFFLOAD":
        tg_ok += 1
    # a replay of an older run carries no wall clock, and an absent run
    # is not a failed one
    tg_old = dict(blank_pass(), offload_n=41, offload_total=41)
    if diagnose(tg_old, tg_old, "llama.cpp")[0] == "HEALTHY":
        tg_ok += 1

    # the amd meter: sysfs only, no rocm-smi, and a three card box
    # collapses to one reading per tick
    am_ok, am_all = 0, 5

    def am_tree(root, cards):
        for name, fields in cards.items():
            dev = os.path.join(root, name, "device")
            os.makedirs(dev)
            for rel, val in fields.items():
                path = os.path.join(dev, rel)
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                with open(path, "w") as fh:
                    fh.write("{}\n".format(val))
        return gpu_meters._AMDGPU(root=root)

    pw = os.path.join("hwmon", "hwmon3", "power1_average")
    with tempfile.TemporaryDirectory() as am_root:
        am = am_tree(am_root, {
            "card0": {"gpu_busy_percent": 97, "mem_info_vram_used": 9 * gib,
                      pw: 310000000},
            "card1": {"gpu_busy_percent": 12, "mem_info_vram_used": 9 * gib,
                      pw: 95000000},
            "card2": {"gpu_busy_percent": 41, "mem_info_vram_used": 9 * gib},
            # a connector, not a gpu, and a card whose firmware is silent
            "card0-DP-1": {"gpu_busy_percent": 99,
                           "mem_info_vram_used": 999 * gib},
            "card3": {"mem_info_vram_used": 9 * gib},
        })
        if len(am.cards) == 3:
            am_ok += 1
        s = am.sample()
        # util is the max (two cards finishing early must not read idle),
        # memory and watts are the sum (the weights and the draw split)
        if s and s["dev"] == 97 and s["mem"] == 27 * gib \
                and abs(s["gpu_w"] - 405.0) < 0.01:
            am_ok += 1
        # a field nothing answers is absent, never a zero
        for c in ("card0", "card1", "card2"):
            os.remove(os.path.join(am_root, c, "device",
                                   "mem_info_vram_used"))
        s = am.sample()
        if s and s["mem"] is None and s["dev"] == 97 \
                and am.device_name() is None and am.throttled() is False:
            am_ok += 1
        # one unreadable card drops out; the others still report
        with open(os.path.join(am_root, "card0", "device",
                               "gpu_busy_percent"), "w") as fh:
            fh.write("n/a\n")
        s = am.sample()
        if s and s["dev"] == 41 and abs(s["gpu_w"] - 95.0) < 0.01:
            am_ok += 1
        # nothing readable is no sample, and the meter never opens on a
        # tree with no amd card in it
        for c in ("card1", "card2"):
            with open(os.path.join(am_root, c, "device",
                                   "gpu_busy_percent"), "w") as fh:
                fh.write("n/a\n")
        try:
            gpu_meters._AMDGPU(root=os.path.join(am_root, "card3"))
            opened = True
        except OSError:
            opened = False
        if am.sample() is None and not opened:
            am_ok += 1

    # real curve regression: the two 4090 telemetry captures replay
    # through the real window math; the misbuilt one must convict and
    # the healthy one must not
    rc_ok, rc_all = 0, 2

    def load_curve(name):
        rows = []
        path = os.path.join(lxroot, name + ".telemetry.jsonl")
        if not os.path.exists(path):
            return rows
        for line in open(path):
            d = json.loads(line)
            if "util_gpu" in d:
                rows.append({"t": d["t"], "dev": d["util_gpu"],
                             "mem": d["mem_used_mib"] * 1024 ** 2,
                             "gpu_w": d.get("power_w")})
        return rows

    def curve_marks(samples, meta_name, rep):
        meta = json.load(open(os.path.join(lxroot, meta_name)))
        return [{"t_end": samples[-1]["t"], "wall_s": meta["wall_s"],
                 "load_s": (rep["load_ms"] or 0) / 1000.0,
                 "prompt_s": (rep["prompt_ms"] or 0) / 1000.0,
                 "eval_s": (rep["eval_ms"] or 0) / 1000.0}]

    rc_m = load_curve("misbuilt-cpu")
    if rc_m and lx_m:
        mtele = telemetry_summary(
            rc_m, curve_marks(rc_m, "misbuilt-cpu.meta.json", lx_m),
            src="nvml")
        if telemetry_read(mtele) == "flat" \
                and diagnose(lx_m, lx_m, "llama.cpp",
                             mtele)[0] == "SILENT CPU FALLBACK":
            rc_ok += 1
    rc_h = load_curve("cuda-healthy")
    if rc_h and lx_h:
        htele = telemetry_summary(
            rc_h, curve_marks(rc_h, "cuda-healthy.meta.json", lx_h),
            src="nvml")
        if diagnose(lx_h, lx_h, "llama.cpp", htele)[0] == "HEALTHY" \
                and telemetry_vote(htele, lx_h,
                                   "llama.cpp") in ("agree", "abstain"):
            rc_ok += 1
    # plan: a synthetic gguf header replays through the real reader,
    # and the kv formula must land on the engine's own committed
    # allocation figures; the speed gate refuses everything but a
    # cached dense measurement
    pl_ok, pl_all = 0, 6

    def synth_gguf(arch, kvs):
        out = [b"GGUF", struct.pack("<I", 3), struct.pack("<Q", 0),
               struct.pack("<Q", len(kvs) + 1)]

        def emit(key, t, packed):
            out.append(struct.pack("<Q", len(key)) + key.encode())
            out.append(struct.pack("<I", t) + packed)

        emit("general.architecture", 8,
             struct.pack("<Q", len(arch)) + arch.encode())
        for k, v in kvs.items():
            emit(arch + "." + k, 4, struct.pack("<I", v))
        return io.BytesIO(b"".join(out))

    m9 = gguf_meta_stream(synth_gguf("qwen35", {
        "block_count": 32, "attention.head_count": 16,
        "attention.head_count_kv": 4, "attention.key_length": 256,
        "attention.value_length": 256, "full_attention_interval": 4}))
    kv9, note9 = kv_account(m9)
    # 1: reader plus formula reproduce the engine's own 128.00 MiB
    #    llama_kv_cache line (examples/raw/healthy-metal, ctx 4096)
    if m9["qwen35.block_count"] == 32 and kv9 == 128 * 1024 ** 2 \
            and "8 of 32" in note9:
        pl_ok += 1
    m35 = gguf_meta_stream(synth_gguf("qwen35moe", {
        "block_count": 40, "attention.head_count": 16,
        "attention.head_count_kv": 2, "attention.key_length": 256,
        "attention.value_length": 256, "full_attention_interval": 4,
        "expert_count": 256}))
    kv35, _n = kv_account(m35)
    # 2: the moe kv comes from the attention geometry alone: 80.00 MiB,
    #    matching the engine's allocation for the local 35B
    if kv35 == 80 * 1024 ** 2 and plan_is_moe(m35) and not plan_is_moe(m9):
        pl_ok += 1
    # 3: head_dim falls back to embedding over heads when the header
    #    has no key/value length (the classic dense layout)
    kvf, _n = kv_account(gguf_meta_stream(synth_gguf("llama", {
        "block_count": 32, "attention.head_count": 32,
        "attention.head_count_kv": 8, "embedding_length": 4096})))
    if kvf == 4096 * 32 * 8 * 256 * 2:
        pl_ok += 1
    # 4: the fit bands sit where the calibration put them (the 35B ran
    #    healthy at 85% of budget, so 80% fits, 100% tight, 112% no)
    if plan_state(20 * gib, 25 * gib) == "fits" \
            and plan_state(25 * gib, 25 * gib) == "tight" \
            and plan_state(28 * gib, 25 * gib) == "no":
        pl_ok += 1
    # 5: no cached run, no speed: the refusal is explicit, not a guess
    bw, note = plan_speed_source(None)
    if bw is None and "not calibrated" in note:
        pl_ok += 1
    # 6: a cached dense run prices a dense target and refuses a moe
    #    target; a cached moe run refuses to calibrate at all
    bw, note = plan_speed_source({"model_bytes": 5 * gib, "moe": False,
                                  "model_name": "m",
                                  "rates": {"decode": 20.0}})
    mbw, mnote = plan_speed_source({"model_bytes": 5 * gib, "moe": True,
                                    "model_name": "m",
                                    "rates": {"decode": 20.0}})
    if bw == 100 * gib and plan_est_decode(bw, 10 * gib, False) == 10.0 \
            and plan_est_decode(bw, 10 * gib, True) is None \
            and mbw is None and "mixture of experts" in mnote:
        pl_ok += 1
    # id: a synthetic gguf with a real tensor table replays through the
    # same walk, account and expert arithmetic used live (the big real
    # files stay out of ci; they are the manual acceptance step). The
    # engine side of the cross check reads the committed real stderr.
    id_ok, id_all = 0, 12

    def synth_id_img(specs, kvs, strings=()):
        # a minimal legal gguf v3 image: kv section, tensor table,
        # data section sized and aligned exactly like the real files
        arch = "synthmoe"
        pairs = [("general.architecture", 8,
                  struct.pack("<Q", len(arch)) + arch.encode()),
                 ("general.file_type", 4, struct.pack("<I", 15))]
        pairs += [(k, 8, struct.pack("<Q", len(v)) + v.encode())
                  for k, v in strings]
        pairs += [(arch + "." + k, 4, struct.pack("<I", v))
                  for k, v in kvs]
        kvb = b"".join(struct.pack("<Q", len(k)) + k.encode()
                       + struct.pack("<I", t) + p for k, t, p in pairs)
        off, rows = 0, []
        for name, dims, tt in specs:
            _tn, tsize, blck = GGML_TENSOR_TYPES[tt]
            n = 1
            for d in dims:
                n *= d
            rows.append(struct.pack("<Q", len(name)) + name.encode()
                        + struct.pack("<I", len(dims))
                        + struct.pack("<{}Q".format(len(dims)), *dims)
                        + struct.pack("<I", tt) + struct.pack("<Q", off))
            off += (n // blck * tsize + 31) // 32 * 32
        img = (b"GGUF" + struct.pack("<I", 3)
               + struct.pack("<Q", len(specs))
               + struct.pack("<Q", len(pairs)) + kvb + b"".join(rows))
        return img + b"\0" * (-len(img) % 32) + b"\0" * off

    img = synth_id_img([("blk.0.attn_q.weight", (256, 4), 12),
                        ("blk.0.ffn_down_exps.weight", (256, 2, 4), 13),
                        ("output_norm.weight", (256,), 0)],
                       [("expert_count", 4), ("expert_used_count", 2)])
    fh = io.BytesIO(img)
    idm = gguf_meta_stream(fh)
    idt, id_hdr_end = gguf_tensor_table(fh, idm.get("__tensor_count", 0))
    id_data = len(img) - (id_hdr_end + 31) // 32 * 32
    idh, ide, idb = id_account(idt, id_data)
    # 1: the walk reads the claim and the table, and the two pricing
    #    methods close. Priced by hand from the machine's own libggml
    #    triples: 1024 q4_K elements are 576 bytes (144 per 256
    #    block), 2048 q5_K are 1408, 256 f32 are 1024.
    if LLAMA_FTYPES.get(idm.get("general.file_type")) == "Q4_K_M" \
            and idh["q4_K"] == [1, 1024, 576] \
            and idh["q5_K"] == [1, 2048, 1408] \
            and idh["f32"] == [1, 256, 1024] and (ide, idb) == (3328, 3008):
        id_ok += 1
    # 2: a lying offset and a type outside the pinned triples both
    #    refuse to price instead of printing a wrong number
    ok2 = 0
    try:
        id_account([idt[0], (idt[1][0], idt[1][1], idt[1][2],
                             idt[1][3] + 64), idt[2]], id_data)
    except ValueError:
        ok2 += 1
    try:
        id_account([("x", (32,), 4, 0)])
    except ValueError:
        ok2 += 1
    if ok2 == 2:
        id_ok += 1
    # 3: the engine's own loader census crosses the walk: the committed
    #    healthy fixture reports the same five type counts the real
    #    file's table measured (verified against the file the day this
    #    landed), and its kv marker parses to f16/f16
    hp = parse_stderr(open(os.path.join(
        rawroot, "healthy-metal", "pass1.stderr.txt")).read(), 10.0)
    if hp["tensor_types"] == {"f32": 177, "q8_0": 48, "q4_K": 132,
                              "q5_K": 48, "q6_K": 22} \
            and hp["kv_types"] == ["f16", "f16"]:
        id_ok += 1
    # 4: the non-f16 sample measured here (-ctk q8_0 -ctv q8_0,
    #    committed raw) pins the K (q8_0) line shape
    qp = parse_stderr(open(os.path.join(
        rawroot, "kv-q8", "ctk-q8.stderr.txt")).read(), 1.0)
    if qp["kv_types"] == ["q8_0", "q8_0"]:
        id_ok += 1
    # 5: the expert bank is the slowest dimension matching
    #    expert_count: 2048 of 3328 elements park in banks, 2 of 4
    #    experts wake, so one token reads 2304; a dense header (no
    #    expert_count) renders no axis at all
    if id_experts(idm, idt, ide) == (2, 4, 2304) \
            and id_experts({"general.architecture": "d"}, idt, 9) is None:
        id_ok += 1
    # 6: the api mirror prices through the same account (no offsets to
    #    audit over http) and an unknown type string refuses
    ok6 = False
    try:
        oh, oe, ob = id_account(ollama_tensor_table(
            [{"name": "w", "type": "Q4_K", "shape": [256, 4]}]))
        ok6 = oh["q4_K"] == [1, 1024, 576] and (oe, ob) == (1024, 576)
        ollama_tensor_table([{"name": "w", "type": "Q9_Z",
                              "shape": [1]}])
        ok6 = False
    except ValueError:
        pass
    if ok6:
        id_ok += 1
    # 7: Ollama's runner line has the same concrete K/V markers even
    #    though /api/generate and /api/ps do not expose dtype fields
    if parse_kv_types(
            "llama_kv_cache: size = 128.00 MiB (4096 cells), "
            "K (q8_0): 64.00 MiB, V (f16): 64.00 MiB") \
            == ["q8_0", "f16"]:
        id_ok += 1
    # 8: only the cold Ollama pass loads a runner and writes that line;
    #    aggregation must carry its evidence over the warm last pass
    kp1, kp2 = blank_pass(), blank_pass()
    kp1["kv_types"] = ["f16", "f16"]
    kp1["kv_source"] = "Ollama runner log"
    kr = build_rep([kp1, kp2])
    if kr["kv_types"] == ["f16", "f16"] \
            and kr["kv_source"] == "Ollama runner log":
        id_ok += 1
    # 9: identity cards select an exact model+engine partition and never
    #    borrow the global last run from another model
    ck = measurement_key("ollama", "qwen3.5:9b")
    cr = {"model_name": "other:latest", "kv_types": ["q4_0", "q4_0"],
          "measurement_key": measurement_key("ollama", "other:latest"),
          "measurements": {ck: {"model_name": "qwen3.5:9b",
                                "kv_types": ["f16", "f16"]}}}
    if cache_for_measurement(cr, ck, "qwen3.5:9b")["kv_types"] \
            == ["f16", "f16"] and cache_for_measurement(
                cr, measurement_key("ollama", "missing:latest"),
                "missing:latest") is None:
        id_ok += 1
    # 10: the identity line is sha256 over every byte plus the exact
    #     count, so two people holding the same file print the same
    #     string. Written to a real file, then checked against hashlib
    #     over the same bytes.
    idimg = synth_id_img(
        [("blk.0.attn_q.weight", (256, 4), 12)], [],
        [("general.quantized_by", "SomeQuantizer"),
         ("general.repo_url", "https://example.invalid/who")])
    iddir = tempfile.mkdtemp(prefix="picchio-id-")
    idp = os.path.join(iddir, "synth.gguf")
    with open(idp, "wb") as f:
        f.write(idimg)
    want_id = "sha256 {}, {:,} bytes".format(
        hashlib.sha256(idimg).hexdigest()[:12], len(idimg))
    if id_file_note(idp) == want_id:
        id_ok += 1
    # 11: the same bytes twice give the same line; one byte flipped in
    #     the middle gives a different one at an identical file size.
    #     The middle is the whole point. A head-and-tail fingerprint
    #     would call these two files the same file, and a same-size
    #     different-content pair is not hypothetical: the ollama blob
    #     for qwen3.5:9b and the unsloth gguf in examples/quantizers
    #     share a byte count and do not share a digest.
    mid = len(idimg) // 2
    flip = idimg[:mid] + bytes([idimg[mid] ^ 0xFF]) + idimg[mid + 1:]
    idp2 = os.path.join(iddir, "flipped.gguf")
    with open(idp2, "wb") as f:
        f.write(flip)
    if id_file_note(idp) == want_id and len(flip) == len(idimg) \
            and id_file_note(idp2) != want_id:
        id_ok += 1
    # 12: origin keys are quoted as written and labeled unverified; a
    #     file carrying none says so rather than naming a likely author
    idsrc = gguf_source_note(gguf_meta_stream(io.BytesIO(idimg)))
    idbare = gguf_source_note(gguf_meta_stream(io.BytesIO(img)))
    if "quantized_by SomeQuantizer" in idsrc \
            and "repo_url https://example.invalid/who" in idsrc \
            and "verified against nothing" in idsrc \
            and idbare.startswith("not recorded") \
            and "no origin keys" in idbare:
        id_ok += 1
    shutil.rmtree(iddir, ignore_errors=True)
    # settings: the disclosure line. Every value on it was already in a
    # log picchio was parsing anyway, so the whole group is about
    # repeating them without inventing any, and about saying which
    # surface stayed quiet when one does.
    st_ok, st_all = 0, 7
    # 1: llama.cpp's own log, read field by field against what the
    #    committed stderr literally prints (sampler seed: 7 and
    #    "top_k = 40, top_p = 0.950, min_p = 0.050, ..., temp = 0.800")
    if hp["sampling"] == {"seed": 7, "top_k": 40.0, "top_p": 0.95,
                          "min_p": 0.05, "temp": 0.8} \
            and settings_line(hp, "llama.cpp") \
            == "temp 0.8, top-k 40, top-p 0.95, min-p 0.05, seed 7":
        st_ok += 1
    # 1b: counts print as counts and continuous knobs keep a decimal.
    #     Not cosmetic: Qwen3.6-35B-A3B ships general.sampling.temp = 1.0
    #     and general.sampling.top_k = 20 in its own header, llama.cpp
    #     honors both, and the same picchio command therefore sampled
    #     that model at temp 1.0 where the 9B ran llama.cpp's 0.8
    #     default. A "temp 1" in that block would read like a count.
    if settings_line({"sampling": {"temp": 1.0, "top_k": 20, "top_p": 0.95,
                                   "min_p": 0.05, "seed": 7}},
                     "llama.cpp") \
            == "temp 1.0, top-k 20, top-p 0.95, min-p 0.05, seed 7":
        st_ok += 1
    # 2: the list separator is not a decimal point. "top_p = 0.950,"
    #    ends on a comma, and a value group allowed to end on one hands
    #    _num a thousands separator to undo, which returns 950. This
    #    shipped for exactly one test run before this check existed.
    if parse_stderr("top_k = 40, top_p = 0.950, min_p = 0.050, "
                    "typical_p = 1.000, temp = 0.800", 1.0)["sampling"] \
            == {"top_k": 40.0, "top_p": 0.95, "min_p": 0.05,
                "temp": 0.8}:
        st_ok += 1
    # 3: the server echoes the same settings under its own names, as
    #    floats that survived a json round trip (0.800000011920929).
    #    One line renders both engines and neither shows sixteen digits.
    sv = map_server(json.load(open(os.path.join(
        rawroot, "server-endpoint", "pass1.response.json"))), 9.0)
    if settings_line(sv, "server") \
            == "temp 0.8, top-k 40, top-p 0.95, min-p 0.05, seed 7":
        st_ok += 1
    # 4: ollama's generate api returns no sampling at all. The line says
    #    not recorded and names the surface that was read; it never
    #    fills in llama.cpp's 0.8 default on ollama's behalf, which
    #    would be the one number in the block a reader could not trust.
    ol = map_ollama(json.load(open(os.path.join(
        rawroot, "ollama-qwen35", "pass1.response.json"))), 10.0, None)
    olline = settings_line(ol, "ollama")
    if ol["sampling"] is None and olline.startswith("not recorded") \
            and "ollama api" in olline and "temp" not in olline:
        st_ok += 1
    # 5: whichever engine stayed quiet, the line still fits the block.
    #    Same discipline as the os line's reasons: the text comes from
    #    a table that grows, so the width is checked, not remembered.
    if all(len("settings " + settings_line({}, mode)) <= WIDTH
           for mode in list(NO_SAMPLING) + ["something new"]):
        st_ok += 1
    # 6: guard echoes the command it is judging, quoted well enough to
    #    paste back into a shell. Everything else guard prints is about
    #    these arguments and cannot be read without them.
    gcmd = ["llama-completion", "-m", "/tmp/models/Qwen3.5-9B-Q4_K_M.gguf",
            "-p", "Say hi.", "-ngl", "0"]
    gecho = "picchio guard: command: {}".format(
        " ".join(shlex.quote(a) for a in gcmd))
    if "'Say hi.'" in gecho and "-ngl 0" in gecho \
            and gecho.count("llama-completion") == 1:
        st_ok += 1
    # energy: joules per generated token, priced off the decode window
    # only and recomputable by hand from the committed curve
    en_ok, en_all = 0, 4
    entele = json.load(open(os.path.join(
        rawroot, "healthy-metal", "telemetry.json")))
    ensum = telemetry_summary(entele["samples"], entele["marks"])
    # 1: an independent recompute of the same figure straight off the
    #    raw samples: median watts inside each decode window, over the
    #    warm median decode rate the lane table prints. Same arithmetic
    #    anyone reading the block can redo with a calculator.
    enwin = []
    for mk in entele["marks"]:
        d1 = mk["t_end"] - TELE_PAD_S
        d0 = d1 - mk["eval_s"]
        enwin += [s["gpu_w"] for s in entele["samples"]
                  if d0 <= s["t"] <= d1 and s.get("gpu_w") is not None]
    enrep = {"decode_toks": 21.1}
    if abs(energy_per_token(ensum, enrep)
           - statistics.median(enwin) / 21.1) < 1e-9 \
            and "0.52 J/tok" in os_line(ensum, enrep):
        en_ok += 1
    # 2: the decode window is not the compute window. Prefill drives the
    #    gpu harder than decode does, so pricing a generated token off
    #    the combined median would bill decode for prefill's bursts.
    if ensum["dec_w"] is not None and ensum["work_w"] is not None \
            and ensum["dec_w"] < ensum["work_w"] \
            and ensum["dec_n"] < ensum["work_n"]:
        en_ok += 1
    # 3: sampled, but the power channel gave nothing. macOS reads watts
    #    out of a private framework that can vanish on an os update, and
    #    a field that disappears with its meter looks exactly like a
    #    field nobody printed. It says n/a instead.
    now = telemetry_summary(
        [dict(s, gpu_w=None) for s in entele["samples"]], entele["marks"])
    if energy_per_token(now, enrep) is None \
            and "n/a J/tok" in os_line(now, enrep) \
            and "work 99%" in os_line(now, enrep):
        en_ok += 1
    # 4: too few decode samples to have a median, and no rate to divide
    #    by, both abstain. A power median off two ticks is a number with
    #    no error bar pretending to have one.
    if energy_per_token(dict(ensum, dec_n=3), enrep) is None \
            and energy_per_token(ensum, {"decode_toks": None}) is None \
            and energy_per_token({"off": "disabled"}, enrep) is None:
        en_ok += 1
    # share: the three postable shapes. All of it is reformatting, so
    # the group is about the reformatting never inventing or losing a
    # value the block did not already carry.
    sh_ok, sh_all = 0, 5
    shb = parse_block(ha)
    shf = share_facts(shb)
    # 1: same source, same value. Every field the one line form prints
    #    is read straight off the block, so each one has to appear in
    #    it verbatim; a share line that disagreed with its own block
    #    would be worse than no share line.
    shline = share_line(shf)
    if all(str(x) in shline for x in
           (shb["model"], shb["quant"], shb["engine"], shb["ctx"],
            shb["settings"], shb["protocol"],
            "{:.1f}".format(shb["rates"][0]),
            "{:.1f}".format(shb["rates"][1]))) \
            and SHARE_URL in shline and shline.count(" | ") == 11:
        sh_ok += 1
    # 2: the file side is absent without a model to walk, and absent
    #    prints as n/a. Dropping those fields would make a shorter line
    #    that reads as though nobody had asked for them.
    if share_missing(shf) == ["bpw", "sha256", "kv"] \
            and "bpw n/a" in shline and "sha256 n/a" in shline \
            and "KV n/a" in shline:
        sh_ok += 1
    # 3: the row is a whole table, not a naked row, so it renders
    #    wherever it is pasted. Three lines, equal column counts, and a
    #    separator made only of markdown alignment cells.
    rows = share_row(shf).splitlines()
    widths = {len(r.split("|")) for r in rows}
    if len(rows) == 3 and len(widths) == 1 \
            and widths.pop() == len(SHARE_COLUMNS) + 2 \
            and all(set(c.strip()) <= set("-:") and c.strip()
                    for c in rows[1].split("|")[1:-1]) \
            and rows[1].count("---:") == 2 and SHARE_URL in rows[0]:
        sh_ok += 1
    # 4: the skeleton carries the block through unchanged and leaves
    #    the opinion to the person who ran it
    post = share_post(shf, ha)
    if all(ln in post for ln in ha.rstrip().splitlines()) \
            and "your take goes here" in post \
            and "```text" in post and SHARE_URL in post:
        sh_ok += 1
    # 5: a walked file supplies exactly the three fields the block
    #    cannot carry, and the quant it walked wins over the one
    #    guessed from a filename
    shimg = synth_id_img([("blk.0.attn_q.weight", (256, 4), 12)], [])
    shdir = tempfile.mkdtemp(prefix="picchio-share-")
    shp = os.path.join(shdir, "Some-Model-Q8_0.gguf")
    with open(shp, "wb") as f:
        f.write(shimg)
    shf2 = share_facts(shb, shp)
    if shf2["quant"] == "Q4_K_M" and shf2["bpw"] == 4.5 \
            and shf2["sha256"] == hashlib.sha256(shimg).hexdigest() \
            and "4.50 bpw" in share_line(shf2):
        sh_ok += 1
    shutil.rmtree(shdir, ignore_errors=True)
    # vet: reading someone else's post. Three samples taken from the
    # thread this came from, at the three specification levels that
    # thread kept producing.
    vt_ok, vt_all = 0, 5
    vague = "Q4 gives me 50 t/s"
    bench = ("llama-bench -m gpt-oss-20b-mxfp4.gguf -ngl 99 -fa 1 "
             "-ctk q8_0 -ctv q8_0 -b 2048 -t 8 | gpt-oss 20B MXFP4 MoE "
             "| pp512 | 823.93 | tg128 | 42.06 | t/s "
             "| RTX 4060 Laptop GPU - 8GB VRAM")
    asking = ('Or KV cache quant. "This model sucks past 32k '
              'context!" What is your kv cache at? "Q4"')
    # 1: the vague one states a number and nothing that makes it mean
    #    anything. Q4 is a family, not a file, so it is not a recipe.
    vs = vet_scan(vague)
    if vs == {"rate": "50"} and vet_quant_note(vague, vs) == "Q4" \
            and vet_rate_lane(vague) is None:
        vt_ok += 1
    # 2: the specified one states most of it, and the lane labels are
    #    what make its two figures readable at all
    vb = vet_scan(bench)
    if vb["model"] == "gpt-oss-20b" and vb["quant"] == "mxfp4" \
            and vb["engine"] == "llama-bench" and vb["kv"] == "q8_0" \
            and vb["placement"] == "99" and vb["machine"] == "RTX 4060" \
            and vet_rate_lane(bench) == "both prefill and decode" \
            and "ctx" not in vb:
        vt_ok += 1
    # 3: raised but unreadable is its own answer. The bench table names
    #    a t/s column with the figures three pipes away, and the third
    #    sample argues about a kv quant without ever stating it.
    va = vet_scan(asking)
    if vb["rate"] == "unparsed" and va["kv"] == "unparsed" \
            and va["ctx"] == "32k" and "engine" not in va:
        vt_ok += 1
    # 4: mechanisms are named when the text raises them, not whenever a
    #    field is missing. The vague sample raises exactly one; the
    #    third raises the kv and depth pair it is actually about.
    def fired(t):
        return {n["key"] for n in VET_NOTES
                if re.search(n["trigger"], t, re.I)}
    if fired(vague) == {"quant-spread"} \
            and fired(asking) == {"kv-dtype", "ctx-depth",
                                  "quant-spread"} \
            and "moe-residency" in fired(bench):
        vt_ok += 1
    # 5: every mechanism cites a file that is in this tree. A note
    #    whose evidence moved or was never committed is a tool
    #    lecturing from memory, which is the thing it is arguing
    #    against.
    if all(os.path.exists(os.path.join(here, src))
           for n in VET_NOTES for src in n["sources"]) \
            and all(n["sources"] for n in VET_NOTES):
        vt_ok += 1
    # cache: a prefill number that is not prefill. Every fixture here is
    # a real run; the pathological one came off an RTX 5090 box.
    ch_ok, ch_all = 9, 9

    def _leg(name):
        out = []
        for i in (1, 2, 3):
            base = os.path.join(rawroot, name, "pass{}".format(i))
            meta = json.load(open(base + ".meta.json"))
            if os.path.exists(base + ".stderr.txt"):
                p = parse_stderr(open(base + ".stderr.txt").read(),
                                 meta["wall_s"])
            else:
                p = map_ollama(json.load(open(base + ".response.json")),
                               meta["wall_s"], meta.get("ps"))
            # the artifact's own nonce, exactly as the replay loop reads
            # it: legs recorded before nonces existed carry None and are
            # judged as the runs that produced them were
            p["nonce"] = meta.get("prompt_nonce")
            out.append(p)
        return out
    # 1: ollama 0.32.15 served passes 2 and 3 out of its prefix cache,
    #    reporting all 770 prompt tokens in 36 ms. The lane abstains and
    #    the reason names the pass that disagrees.
    doubt = prefill_trust(_leg("linux-5090-ollama"), "ollama")
    if not (doubt and doubt[0] == "all" and "cold pass" in doubt[1]):
        ch_ok -= 1
    # 2: llama.cpp on Vulkan compiles shaders inside the first pass, so
    #    its cold prefill is 111x off its own warm passes. Same rule,
    #    same abstention: picchio cannot tell which number to trust.
    if (prefill_trust(_leg("linux-5090-vulkan")) or ("", ""))[0] != "all":
        ch_ok -= 1
    # 3: healthy runs are nowhere near the gate and keep their numbers.
    #    CUDA spread 1.0016x, Metal 1.005x, ollama on Metal 1.05x.
    if any(prefill_trust(_leg(n)) for n in
           ("linux-5090-cuda", "healthy-metal", "ollama-qwen35")):
        ch_ok -= 1
    # 8: the gate reads how far out a pass is, not which pass it is. The
    #    5090 CUDA leg has a cold pass like every other run, and its cold
    #    prefill lands 0.16% off the warm ones, so nothing abstains. A
    #    gate that fired on cold as such would eat this one too.
    cu = [p["prefill_toks"] for p in _leg("linux-5090-cuda")]
    if not (prefill_trust(_leg("linux-5090-cuda"), "llama.cpp", "CUDA") is None
            and max(cu) / min(cu) < 1.01):
        ch_ok -= 1
    # 9: the same asymmetry on a real nonce run rather than a spliced
    #    one, end to end: llama.cpp on Vulkan, every pass behind its own
    #    nonce, cold abstains and is named shader compilation while the
    #    warm passes stand. The naming needs the backend, and this build
    #    prints no ggml_vulkan banner at all, so the backend has to come
    #    off the device llama.cpp says it is using.
    vkn = _leg("linux-5090-vulkan-nonce")
    vkr = build_rep(vkn)
    vt2 = prefill_trust(vkn, "llama.cpp", vkr.get("gpu_kind"))
    if not (vkr.get("gpu_kind") == "Vulkan"
            and "5090" in (vkr.get("gpu_device") or "")
            and vt2 and vt2[0] == "cold"
            and "shader compilation" in vt2[1]):
        ch_ok -= 1
    # 6: the same numbers, this time with the run's own nonce evidence
    #    on every pass. The cache explanation is ruled out by that
    #    evidence, so only the cold cell goes and the warm passes,
    #    which agree with each other to a tenth of a percent, survive.
    #    Without this asymmetry shader compilation would eat every
    #    Vulkan warm prefill forever.
    vk = _leg("linux-5090-vulkan")
    for i, q in enumerate(vk):
        q["nonce"] = prompt_nonce("0f3a9c81", i)
    vt = prefill_trust(vk, "llama.cpp", "Vulkan")
    if not (vt and vt[0] == "cold" and "shader compilation" in vt[1]):
        ch_ok -= 1
    # 7: nonce evidence is per pass and has to be distinct. One shared
    #    prefix, or one missing, and nothing was ruled out.
    if nonce_witnessed(vk) is not True \
            or nonce_witnessed([dict(q, nonce=None) for q in vk]) \
            or nonce_witnessed([dict(q, nonce="same") for q in vk]):
        ch_ok -= 1
    # 4: the nonce differs per pass and keeps one shape, so the passes
    #    stay comparable while no two of them share a prefix
    ns = [bench_prompt("0f3a9c81", i) for i in range(3)]
    if len({n[:22] for n in ns}) != 3 \
            or len({len(n) for n in ns}) != 1 \
            or not all(n.endswith(BENCH_PROMPT) for n in ns):
        ch_ok -= 1
    # 5: the server lane was already immune and the committed evidence
    #    says so rather than the code claiming it: picchio sends
    #    cache_prompt false and every committed pass came back cache_n 0
    if not all(json.load(open(os.path.join(
            rawroot, "server-endpoint",
            "pass{}.response.json".format(i))))["timings"]["cache_n"] == 0
            for i in (1, 2, 3)):
        ch_ok -= 1
    # argv split: the `--` passthrough is cut by hand before argparse,
    # so its semantics cannot vary with the interpreter (3.9.6 and
    # 3.12.3 rejected an option followed by `--` as unrecognized
    # arguments, measured; 3.12.13 and 3.13+ accept it). The command
    # shapes users actually type:
    av_ok, av_all = 0, 4
    # 1: options before the separator stay with picchio, engine args
    #    after it arrive verbatim (the exact shape old argparse rejects)
    if split_engine_args(["m.gguf", "--keep-logs", "d", "--", "-ngl", "0"]) \
            == (["m.gguf", "--keep-logs", "d"], ["-ngl", "0"]):
        av_ok += 1
    # 2: no separator, trailing junk stays put for the unexpected
    #    extra arguments error, never silently swallowed as engine args
    if split_engine_args(["m.gguf", "foo", "bar"]) \
            == (["m.gguf", "foo", "bar"], None):
        av_ok += 1
    # 3: no separator at all: nothing moves, None says none was typed
    if split_engine_args(["m.gguf"]) == (["m.gguf"], None):
        av_ok += 1
    # 4: only the first separator splits; any later one belongs to the
    #    engine command line untouched
    if split_engine_args(["m.gguf", "--", "-a", "--", "-b"]) \
            == (["m.gguf"], ["-a", "--", "-b"]):
        av_ok += 1
    # onboarding: the zero-argument entry decision is pure given what the
    # scan found, whether a terminal is attached, and what gets typed. The
    # four paths plus the two edges, none of them touching a tty or a gpu
    gd_ok, gd_all = 0, 9
    two = [("qwen3.5:9b", "ollama", "qwen3.5:9b", "5.3 GiB"),
           ("llama-3-8b.gguf", "gguf", "/models/llama-3-8b.gguf",
            "4.6 GiB")]
    one = [two[0]]

    def scripted(lines):
        it = iter(lines)
        return lambda prompt: next(it, None)

    def sink():
        out = []
        return out, out.append

    # 1: exactly one model on a terminal runs with no question asked
    log, emit = sink()
    if resolve_direction(one, True, scripted([]), emit) \
            == ("run", "qwen3.5:9b") \
            and any("Selected: qwen3.5:9b" in x for x in log):
        gd_ok += 1
    # 2: a real fork, the user types the menu number, that model runs
    log, emit = sink()
    if resolve_direction(two, True, scripted(["2"]), emit) \
            == ("run", "/models/llama-3-8b.gguf") \
            and "2 models found." in log:
        gd_ok += 1
    # 3: not a terminal (pipe/redirect) falls back to pasteable commands
    if resolve_direction(two, False, scripted([]), lambda s: None) \
            == ("print", None):
        gd_ok += 1
    # 4: the scan missed it, a typed path overrides the menu and runs
    if resolve_direction(two, True, scripted(["/tmp/my.gguf"]),
                         lambda s: None) == ("run", "/tmp/my.gguf"):
        gd_ok += 1
    # 5: nothing found but a terminal is on, the one prompt takes a tag
    log, emit = sink()
    if resolve_direction([], True, scripted(["some-tag:latest"]), emit) \
            == ("run", "some-tag:latest") and "No models found." in log:
        gd_ok += 1
    # 6: an out-of-range number re-asks, it never runs model zero
    log, emit = sink()
    if resolve_direction(two, True, scripted(["9", "1"]), emit) \
            == ("run", "qwen3.5:9b") \
            and any("No model 9" in x for x in log):
        gd_ok += 1
    # 7: a name longer than the column truncates in the menu row only;
    # the size column survives and the untouched full path still runs
    log, emit = sink()
    longlab = "L" * 47 + "-Q4_K_M.gguf"
    if resolve_direction(two + [(longlab, "gguf", "/m/" + longlab,
                                 "8.0 GiB")], True, scripted(["3"]),
                         emit) == ("run", "/m/" + longlab) \
            and any("..." in x and "8.0 GiB" in x for x in log):
        gd_ok += 1
    # 8: models found past the display cap are surfaced, not dropped in
    # silence; the header counts the true total and a trailer names how
    # many are hidden, and a typed number still reaches a shown one
    log, emit = sink()
    if resolve_direction(two, True, scripted(["1"]), emit, 5) \
            == ("run", "qwen3.5:9b") \
            and "7 models found." in log \
            and any("and 5 more" in x for x in log):
        gd_ok += 1
    # 9: commands printed by non-interactive discovery name the entry that
    # actually ran. A downloaded ./picchio must never instruct the reader
    # to invoke a source file that does not exist beside it.
    old_argv0 = sys.argv[0]
    try:
        sys.argv[0] = "./picchio"
        zip_call = invocation()
        sys.argv[0] = "picchio.py"
        source_call = invocation()
    finally:
        sys.argv[0] = old_argv0
    if zip_call == "./picchio" and source_call == "python3 picchio.py":
        gd_ok += 1
    vp_ok, vp_all = 0, 3
    if parse_engine_version("version: 9430 (d48a56ef)") == "b9430":
        vp_ok += 1
    # the tarball sentinel that once rendered as a fake build b0
    if parse_engine_version("version: 0 (unknown)") == "(version unknown)":
        vp_ok += 1
    if parse_engine_version("") == "(version unknown)":
        vp_ok += 1
    from picchio_core.selftest import run_selftests
    core_ok, core_all, core_failures = run_selftests(
        [sys.executable, os.path.abspath(sys.argv[0])])
    print("parser fixtures {}/{}, verdict replay {}/{}, compare {}/{}, "
          "telemetry {}/{}, verify {}/{}, watch {}/{}, monitor {}/{}, "
          "sweep {}/{}, server {}/{}, linux {}/{}, silent-engine {}/{}, "
          "locale {}/{}, timing-gate {}/{}, amdgpu {}/{}, "
          "residency {}/{}, curves {}/{}, "
          "plan {}/{}, id {}/{}, settings {}/{}, energy {}/{}, "
          "share {}/{}, vet {}/{}, cache {}/{}, argv {}/{}, "
          "version {}/{}, "
          "onboarding {}/{}, queue/parity {}/{}".format(
              fx_ok, fx_all, rp_ok, rp_all, cp_ok, cp_all, te_ok, te_all,
              ve_ok, ve_all, wa_ok, wa_all, mo_ok, mo_all, sw_ok, sw_all,
              sv_ok, sv_all, lx_ok, lx_all, se_ok, se_all,
              lc_ok, lc_all, tg_ok, tg_all, am_ok, am_all,
              rs_ok, rs_all, rc_ok, rc_all,
              pl_ok, pl_all, id_ok, id_all, st_ok, st_all, en_ok, en_all,
              sh_ok, sh_all, vt_ok, vt_all, ch_ok, ch_all,
              av_ok, av_all, vp_ok, vp_all,
              gd_ok, gd_all, core_ok, core_all))
    if core_failures:
        print("queue/parity failures: " + ", ".join(core_failures))
    sys.exit(0 if fx_ok == fx_all and rp_ok == rp_all and rp_all
             and cp_ok == cp_all and te_ok == te_all
             and ve_ok == ve_all and wa_ok == wa_all and mo_ok == mo_all
             and sw_ok == sw_all and sv_ok == sv_all
             and lx_ok == lx_all and se_ok == se_all
             and lc_ok == lc_all and tg_ok == tg_all and am_ok == am_all
             and rs_ok == rs_all and rc_ok == rc_all
             and pl_ok == pl_all and id_ok == id_all and av_ok == av_all
             and st_ok == st_all and en_ok == en_all
             and sh_ok == sh_all and vt_ok == vt_all
             and ch_ok == ch_all
             and vp_ok == vp_all and gd_ok == gd_all
             and core_ok == core_all else 1)


# -------------------------------------------------------------------- main

def split_engine_args(argv):
    """Everything after the first bare `--` goes to the engine
    verbatim, and argparse never sees the separator. Splitting by hand
    is what keeps the passthrough identical on every Python: whether
    argparse itself honors a `--` that follows an option flipped
    between CPython versions (measured here: 3.9.6 and 3.12.3 reject
    "MODEL --keep-logs D -- -ngl 0" as unrecognized arguments, 3.12.13
    and 3.13+ accept it). Returns (argv_before, engine_args), with
    engine_args None when no separator was typed at all."""
    if "--" not in argv:
        return argv, None
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def measurement_key(mode, model):
    """Stable cache partition for one model on one engine surface."""
    if mode == "ollama":
        return "ollama@{}:{}".format(OLLAMA_HOST, model)
    if mode == "server":
        return "llama-server:{}".format(model.rstrip("/"))
    return "llama.cpp:{}".format(os.path.realpath(model))


def cache_for_measurement(cache, key, model=None):
    """Select evidence for this model/engine, with a safe legacy fallback.

    Old cache files held only the global last run. They remain usable only
    when their model name plainly matches; a run for another model must
    never supply an identity card's kv dtype.
    """
    if not isinstance(cache, dict):
        return None
    records = cache.get("measurements")
    if isinstance(records, dict) and isinstance(records.get(key), dict):
        return records[key]
    if cache.get("measurement_key") == key:
        return cache
    if not cache.get("measurement_key") and model:
        names = {str(model), os.path.basename(str(model))}
        if cache.get("model_name") in names:
            return cache
    return None


def save_cache(payload, key=None):
    try:
        payload = dict(payload)
        old = load_cache() or {}
        records = old.get("measurements")
        records = dict(records) if isinstance(records, dict) else {}
        if key:
            previous = records.get(key)
            # A runner can remain resident or its service logs can be
            # unreadable. Keep the last observed dtype, with its own stamp,
            # instead of erasing valid model-specific evidence.
            if not payload.get("kv_types") and isinstance(previous, dict) \
                    and previous.get("kv_types"):
                for field in ("kv_types", "kv_source", "kv_stamp"):
                    payload[field] = previous.get(field)
            payload["measurement_key"] = key
            records.pop(key, None)
            records[key] = dict(payload)
            while len(records) > 32:
                records.pop(next(iter(records)))
            payload["measurements"] = records
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp.{}".format(os.getpid())
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    if sys.argv[1:2] == ["run"]:
        from picchio_core.cli import run_cli
        run_cli(sys.argv[2:], {
            "version": VERSION, "protocol": PROTOCOL,
            "telemetry_start": telemetry_start,
        })
        return
    if sys.argv[1:2] == ["capabilities"]:
        from picchio_core.cli import capabilities_cli
        capabilities_cli(sys.argv[2:], VERSION, PROTOCOL)
        return
    # The explicit verb is the stable AI entry. Legacy positional model
    # invocation stays valid. "ollama" means the one model already resident,
    # not a model tag literally named ollama.
    if sys.argv[1:2] == ["diagnose"]:
        rest = sys.argv[2:]
        if rest[:1] == ["ollama"]:
            loaded, why = ollama_loaded()
            if not loaded:
                sys.exit("picchio diagnose: {}.".format(why))
            rest = [loaded] + rest[1:]
        sys.argv = [sys.argv[0]] + rest
    # guard wraps an arbitrary user command, so its arguments must not
    # pass through the measurement mode parser: dispatch on the word
    if sys.argv[1:2] == ["guard"]:
        guard_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["compare"]:
        compare_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["verify"]:
        verify_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["watch"]:
        watch_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["monitor"]:
        monitor_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["plan"]:
        plan_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["id"]:
        id_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["share"]:
        share_cli(sys.argv[2:])
        return
    if sys.argv[1:2] == ["vet"]:
        vet_cli(sys.argv[2:], WIDTH)
        return
    from picchio_core.cli import command_help_epilog
    ap = argparse.ArgumentParser(
        prog="picchio",
        description="Catch local LLM CPU fallback and see the GPU placement, "
                    "prefill, decode, memory and power behind your run.",
        epilog=command_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("model", nargs="?",
                    help="path to a .gguf file, an ollama model tag, or "
                         "the url of a running llama-server "
                         "(http://host:port)")
    ap.add_argument("--version", action="version",
                    version="picchio {}".format(VERSION))
    ap.add_argument("--bin", help="llama.cpp binary (default: find "
                                  "llama-completion or llama-cli on PATH)")
    ap.add_argument("--passes", type=int, default=3, metavar="N",
                    help=argparse.SUPPRESS)
    ap.add_argument("--explain", type=float, metavar="TOKS",
                    help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true",
                    help="write only JSON to stdout; human verdict goes to "
                         "stderr")
    ap.add_argument("--share", choices=("line", "row", "post"),
                    help="after measuring, write a postable line, Markdown "
                         "row or post to stdout; the full result stays on "
                         "stderr")
    ap.add_argument("--keep-logs", metavar="DIR",
                    help=argparse.SUPPRESS)
    ap.add_argument("--no-telemetry", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--ctx-sweep", nargs="?", const="4096,16384,32768",
                    metavar="LIST", dest="ctx_sweep",
                    help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("extra", nargs="*", default=[],
                    help="args after -- go straight to the llama.cpp engine "
                         "(e.g. -- --device none -ngl 0)")
    argv, engine_args = split_engine_args(sys.argv[1:])
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return
    if args.json and args.share:
        sys.exit("picchio: --json and --share both own stdout; choose one.")
    if args.ctx_sweep is not None and args.share:
        sys.exit("picchio: --share formats an mp1 verdict, while --ctx-sweep "
                 "prints a multi-context table; choose one.")

    if args.model is None and args.explain is not None:
        cached = load_cache()
        if not cached:
            sys.exit("picchio: no previous run cached; run with a model "
                     "first.")
        verdict, para = classify_number(args.explain, cached["rates"])
        print(colorize("\n".join(
            ["YOUR NUMBER: {:.1f} tok/s -> {}".format(args.explain, verdict)]
            + wrap_para(para)
            + ["(rates: {}, {}, {})".format(
                cached.get("model_name", "?"), cached.get("machine", "?"),
                str(cached.get("stamp", "?"))[:10])])))
        return

    if args.model is None:
        # no direction from argv: the one place a model may be asked for.
        # a terminal on both ends means a person is watching; a pipe or a
        # redirect on either end stays composable and is never asked
        cands, dropped = scan_models()
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        action, chosen = resolve_direction(
            cands, interactive, _ask_line,
            lambda line: print(menu_paint(line)), dropped)
        if action == "run":
            args.model = chosen
        else:
            if action == "print":
                print_discovery(cands, dropped)
            elif not interactive:
                print(hint_no_models())
            sys.exit(0)
    if args.passes < 2:
        sys.exit("picchio: --passes must be at least 2 (one cold, one warm).")
    if args.extra:
        # argparse never sees a `--` anymore, so whatever landed in the
        # extra positional is stray junk, not engine args
        sys.exit("picchio: unexpected extra arguments: {}\n"
                 "(a pasted trailing comment does this; engine args need "
                 "a bare -- first)".format(" ".join(args.extra)))
    args.extra = engine_args or []

    mach = machine_info()
    logdir = args.keep_logs
    lp = (lambda name: os.path.join(logdir, name)) if logdir else \
        (lambda name: None)

    mode, binpath, engine_str, model_name = resolve_engine(args.model,
                                                           args.bin)
    if mode != "llama.cpp" and args.extra:
        sys.exit("picchio: passthrough args after -- only work in "
                 "llama.cpp mode.")

    if args.ctx_sweep is not None:
        if mode == "server":
            sys.exit("picchio: --ctx-sweep sets the context size per "
                     "tier, and a server endpoint fixes it server side. "
                     "Run the sweep on the .gguf file instead.")
        # a separate diagnostic, not an mp1 verdict: it changes the prompt
        # per tier, so it prints its own block and never touches the cache
        rows = ctx_sweep(args.model, mode, binpath, engine_str, model_name,
                         parse_tiers(args.ctx_sweep), max(2, args.passes), lp)
        print(colorize(render_sweep(mach, engine_str, model_name, rows)))
        sys.exit(0)

    passes = []
    # one id per invocation; with the pass number it rebuilds the exact
    # prompt every pass sent, which is the whole audit trail the nonce
    # needs. Not a measurement input, so it never reaches the block.
    run_id = "{:08x}".format(struct.unpack("<I", os.urandom(4))[0])
    if mode == "ollama" and ollama_ps_entry(args.model):
        sys.stderr.write("picchio: unloading model for a colder pass 1 ...\n")
        ollama_unload(args.model)
    if mode == "server" and not url_is_local(binpath):
        # ioreg meters this machine; a remote server's gpu is not on it,
        # so the os witness recuses itself instead of testifying about
        # the wrong computer
        sampler = {"off": "remote endpoint", "ev": "timing"}
    else:
        sampler = telemetry_start(args.no_telemetry)
        if mode == "server" and isinstance(sampler, dict):
            sampler["ev"] = "timing"
    if isinstance(sampler, GpuSampler):
        time.sleep(1.2)  # a few ticks of idle baseline before pass 1
    block_ctx = server_ctx(binpath) if mode == "server" \
        else effective_ctx(args.extra)
    for i in range(args.passes):
        if i > 0:
            note = " (warm)"
        elif mode == "server":
            note = " (warm; the server is already loaded)"
        else:
            note = " (includes any cold load)"
        sys.stderr.write("picchio: pass {}{} ...\n".format(i + 1, note))
        nonce = prompt_nonce(run_id, i)
        prompt = bench_prompt(run_id, i)
        if mode == "llama.cpp":
            p = run_llama_pass(binpath, args.model, args.extra,
                               lp("pass{}.stderr.txt".format(i + 1)),
                               prompt=prompt)
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "extra_args": args.extra,
                    "prompt_nonce": nonce}
        elif mode == "server":
            p = run_server_pass(
                binpath, lp("pass{}.response.json".format(i + 1)),
                prompt=prompt)
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "mode": "server",
                    "ctx": block_ctx, "prompt_nonce": nonce}
        else:
            p, ps = run_ollama_pass(
                args.model, lp("pass{}.response.json".format(i + 1)),
                prompt=prompt)
            meta = {"wall_s": p["wall_s"], "engine": engine_str,
                    "model_name": model_name, "ps": ps,
                    "prompt_nonce": nonce}
        p["nonce"] = nonce
        if isinstance(sampler, GpuSampler):
            sampler.mark_pass(p)
        keep_log(lp("pass{}.meta.json".format(i + 1)),
                 json.dumps(meta, indent=1))
        passes.append(p)

    tele = sampler.stop() if isinstance(sampler, GpuSampler) else sampler
    if isinstance(sampler, GpuSampler):
        keep_log(lp("telemetry.json"), json.dumps(
            {"summary": tele, "marks": sampler.marks,
             "samples": sampler.samples}, indent=1))

    cold_note = None
    l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
    if l1 is not None and l2 is not None and l1 < 2 * l2 + 500:
        cold_note = True

    rep = build_rep(passes)
    state, para = diagnose(passes[0], rep, mode, tele)
    why = attribute_why(state, rep, mode, args.extra)

    explain_part = None
    rates = {
        "prefill": rep["prefill_toks"],
        "decode": rep["decode_toks"],
        "wallclock": rep["wallclock_toks"],
    }
    if args.explain is not None:
        v, ep = classify_number(args.explain, rates)
        explain_part = ("{:.1f} tok/s -> {}".format(args.explain, v), ep)

    block = render_verdict(mach, engine_str, model_name, passes, state,
                           para, mode, explain_part, cold_note, why,
                           block_ctx, args.extra, tele)
    exit_code = EXIT_CODES.get(state, 0)
    if args.json or args.share:
        sys.stderr.write(colorize(block, sys.stderr) + "\n")
    else:
        print(colorize(block))

    if mode == "server" and url_is_local(binpath) \
            and not rep.get("model_bytes"):
        # a loopback server's weights are a local file: its size is the
        # one calibration figure the http api cannot give plan
        try:
            rep["model_bytes"] = os.path.getsize(
                server_props(binpath).get("model_path") or "")
        except OSError:
            pass
    stamp = time.strftime("%Y-%m-%d %H:%M")
    save_cache({
        "stamp": stamp,
        "model_name": model_name,
        "machine": "{}, {} GB".format(mach["chip"], mach["ram_gb"] or "?"),
        "protocol": PROTOCOL,
        "rates": rates,
        "state": state,
        # what plan's speed estimate calibrates from: decode x bytes is
        # this machine's effective bandwidth, but only on a dense model
        "model_bytes": rep.get("model_bytes"),
        "moe": (bool(rep["n_expert"]) if rep.get("n_expert") is not None
                else None),
        # The runtime kv dtype this run actually used: llama.cpp stderr
        # or, on a local Ollama host, bytes newly appended to the runner
        # log during this request. The id card never assumes a default.
        "kv_types": rep.get("kv_types"),
        "kv_source": rep.get("kv_source"),
        "kv_stamp": stamp if rep.get("kv_types") else None,
    }, measurement_key(mode, args.model))

    if args.json:
        print(json.dumps({"schema": "picchio.diagnose.v1",
                          "machine": mach, "engine": engine_str,
                          "model": model_name, "mode": mode,
                          "protocol": PROTOCOL, "passes": passes,
                          "warm_median": rates, "state": state,
                          "why": why, "telemetry": tele,
                          "exitCode": exit_code,
                          "evidenceDirectory": os.path.abspath(logdir)
                          if logdir else None}, indent=1))
    elif args.share:
        facts = share_facts(parse_block(block), args.model, sys.stderr.write)
        print(render_share(facts, block, args.share))
        warn_share_missing(facts, args.model)

    sys.exit(exit_code)


def entrypoint():
    try:
        main()
    except SystemExit as e:
        # the exit-code table defines 2 as "could not run"; a bare
        # sys.exit("message") anywhere would otherwise leave python's
        # default 1, which the table never defines, so a caller cannot
        # tell a run failure from anything else. Integer codes pass
        # through untouched (0/2/3/4/5, argparse's own 2 on a usage
        # error, a guarded child's passed-through code); only a string
        # message, which always means we stopped before a verdict, is
        # remapped to 2 after printing it.
        if isinstance(e.code, str):
            sys.stderr.write(e.code.rstrip("\n") + "\n")
            raise SystemExit(2)
        raise


if __name__ == "__main__":
    entrypoint()
