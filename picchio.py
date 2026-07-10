#!/usr/bin/env python3
# picchio: knocks on your local llama.cpp setup and listens for hollow spots.
#
# What it does, in one run:
#   1. runs the same fixed prompt through your model twice (pass 1 and pass 2)
#   2. parses the engine's own timing and layer placement output
#   3. reports prefill, decode and wallclock tok/s as three separate numbers
#   4. tells you whether the GPU actually did the work, or quietly did not
#   5. shows where the seconds of the first pass went (load, prefill, decode)
#   6. prints a verdict block you can paste into an issue or a comment
#
# Usage:
#   python3 picchio.py /path/to/model.gguf
#   python3 picchio.py /path/to/model.gguf --explain 36
#   python3 picchio.py --explain 36              (classifies against last run)
#   python3 picchio.py model.gguf -- --device none -ngl 0   (args after --
#                                       are passed straight to the engine)
#
# Needs: python3 (any recent one), llama.cpp installed somewhere on PATH
# (llama-completion, or llama-cli on older builds). Nothing else. No pip.
#
# Exit codes: 0 ok/healthy, 2 could not run, 3 partial offload,
#             4 silent cpu fallback.

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time

VERSION = "0.1.0"
WIDTH = 66
N_PREDICT = 128
CTX = 4096
CACHE_PATH = os.path.expanduser("~/.cache/picchio/last.json")

# A fixed prompt of roughly 700 tokens. Prefill throughput measured on a
# handful of tokens is dominated by per-call overhead and swings wildly;
# it only settles into a meaningful rate with a few hundred tokens of
# context. The text itself does not matter, only its length and that it
# is identical on every machine.
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
    else:
        info["os"] = sysname
    if not info["chip"]:
        info["chip"] = platform.machine() or "unknown cpu"
    return info


# ------------------------------------------------------------------ engine

def find_binary(explicit):
    if explicit:
        if shutil.which(explicit) or os.path.isfile(explicit):
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


def engine_version(binpath):
    out = _cmd_out([binpath, "--version"])
    m = re.search(r"version:\s*(\S+)\s*\(([0-9a-f]+)\)", out)
    if m:
        return "build {} ({})".format(m.group(1), m.group(2))
    return os.path.basename(binpath)


def run_pass(binpath, model, extra_args):
    base = [
        binpath,
        "-m", model,
        "-p", BENCH_PROMPT,
        "-n", str(N_PREDICT),
        "-c", str(CTX),
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
        try:
            r = subprocess.run(
                args + extra_args,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            sys.exit("picchio: engine run exceeded 30 minutes, giving up.")
        wall_s = time.monotonic() - t0
        if r.returncode == 0:
            return parse_stderr(r.stderr, wall_s)
        last = r
    tail = "\n".join(last.stderr.strip().splitlines()[-6:])
    sys.exit(
        "picchio: engine exited with code {}.\nLast lines:\n{}".format(
            last.returncode, tail
        )
    )


def parse_stderr(text, wall_s):
    d = {
        "wall_s": wall_s,
        "load_ms": None,
        "prompt_ms": None, "prompt_tokens": None,
        "eval_ms": None, "eval_tokens": None,
        "offload_n": None, "offload_total": None,
        "gpu_device": None, "gpu_kind": None,
        "model_params": None, "model_size": None,
        "threads": None, "cores": None,
    }
    re_load = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
    re_pair = re.compile(r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)")
    re_off = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
    re_metal = re.compile(r"ggml_metal_init: found device:\s*(.+)")
    re_cuda = re.compile(r"Device\s+\d+:\s*([^,]+),")
    re_params = re.compile(r"model params\s*=\s*([\d.]+\s*\S?)")
    re_size = re.compile(r"file size\s*=\s*([\d.]+\s*\S+)")
    re_threads = re.compile(r"n_threads\s*=\s*(\d+).*?/\s*(\d+)")

    for line in text.splitlines():
        if "prompt eval time" in line:
            m = re_pair.search(line)
            if m:
                d["prompt_ms"] = float(m.group(1))
                d["prompt_tokens"] = int(m.group(2))
        elif "eval time" in line:
            m = re_pair.search(line)
            if m:
                d["eval_ms"] = float(m.group(1))
                d["eval_tokens"] = int(m.group(2))
        elif "load time" in line:
            m = re_load.search(line)
            if m:
                d["load_ms"] = float(m.group(1))
        m = re_off.search(line)
        if m:
            d["offload_n"] = int(m.group(1))
            d["offload_total"] = int(m.group(2))
        m = re_metal.search(line)
        if m:
            d["gpu_device"] = m.group(1).strip()
            d["gpu_kind"] = "Metal"
        if "ggml_cuda_init" in line or "CUDA devices" in line:
            d["gpu_kind"] = d["gpu_kind"] or "CUDA"
        m = re_cuda.search(line)
        if m and d["gpu_kind"] == "CUDA" and not d["gpu_device"]:
            d["gpu_device"] = m.group(1).strip()
        if "ggml_vulkan" in line.lower() and not d["gpu_kind"]:
            d["gpu_kind"] = "Vulkan"
        m = re_params.search(line)
        if m:
            d["model_params"] = m.group(1).strip()
        m = re_size.search(line)
        if m:
            d["model_size"] = m.group(1).strip()
        if "system_info" in line:
            m = re_threads.search(line)
            if m:
                d["threads"] = int(m.group(1))
                d["cores"] = int(m.group(2))

    if d["prompt_ms"] and d["prompt_tokens"]:
        d["prefill_toks"] = d["prompt_tokens"] / (d["prompt_ms"] / 1000.0)
    else:
        d["prefill_toks"] = None
    if d["eval_ms"] and d["eval_tokens"]:
        d["decode_toks"] = d["eval_tokens"] / (d["eval_ms"] / 1000.0)
    else:
        d["decode_toks"] = None
    if d["eval_tokens"] and wall_s > 0:
        d["wallclock_toks"] = d["eval_tokens"] / wall_s
    else:
        d["wallclock_toks"] = None
    return d


# --------------------------------------------------------------- diagnosis

def diagnose(p1, p2):
    """Returns (state, paragraph). State drives the exit code."""
    n, total = p2["offload_n"], p2["offload_total"]
    gpu = p2["gpu_kind"]
    decode = p2["decode_toks"] or p1["decode_toks"]
    prefill = p2["prefill_toks"] or p1["prefill_toks"]

    if n is None:
        return "NO PLACEMENT EVIDENCE", (
            "This engine build did not report layer placement, so picchio "
            "cannot prove where the model ran. The rates above are still "
            "measured. If prefill is not far above decode, suspect the CPU "
            "and rerun with a newer llama.cpp build."
        )

    if n == 0:
        if gpu:
            placement = ("0 of {} layers offloaded (a {} device was "
                         "initialized, then left idle)".format(total, gpu))
        else:
            placement = ("0 of {} layers offloaded, no GPU device "
                         "initialized".format(total))
        para = ("The engine loaded, answered, and never used the GPU: "
                + placement + ".")
        if decode:
            para += (" Decode looks almost normal ({:.1f} tok/s), which is "
                     "why nobody notices.".format(decode))
        if prefill:
            para += (" Prefill gives it away: at {:.0f} tok/s, a 2500 token "
                     "prompt sits {:.0f} s before the first word appears. "
                     "Check -ngl and your build flags.".format(
                         prefill, 2500.0 / prefill))
        return "SILENT CPU FALLBACK", para

    if total and n < total:
        return "PARTIAL OFFLOAD", (
            "{} of {} layers made it to the GPU, the rest run on CPU. "
            "Usually a memory fit decision by the engine. Expect prefill "
            "and decode both below what this machine can do with a model "
            "or quant that fits entirely.".format(n, total)
        )

    para = "The GPU did the work."
    if decode:
        para += (" Quote the decode number ({:.1f} tok/s) when you compare "
                 "setups.".format(decode))
    if prefill and decode and prefill > 3 * decode:
        para += (" {:.0f} tok/s is real too, but that is prefill: prompt "
                 "reading speed, not generation speed.".format(prefill))
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


def bar_line(label, secs, frac):
    barw = 28
    fill = max(0, min(barw, int(round(frac * barw))))
    return "  {:<13}{:>6.1f} s  {}{}  {:>3.0f}%".format(
        label, secs, "#" * fill, "." * (barw - fill), frac * 100
    )


def wrap_para(text):
    return textwrap.wrap(text, width=WIDTH - 2,
                         initial_indent="  ", subsequent_indent="  ")


def render_verdict(mach, engine_str, model_name, p1, p2, state, para,
                   explain_part=None, cold_note=None):
    out = []
    stamp = time.strftime("%Y-%m-%d %H:%M")
    head = "picchio v{} ".format(VERSION)
    dots = "." * (WIDTH - len(head) - len(stamp) - 1)
    out.append(head + dots + " " + stamp)
    out.append("machine   {}, {} GB ram, {}".format(
        mach["chip"], mach["ram_gb"] or "?", mach["os"]))
    eng = "llama.cpp {}".format(engine_str)
    if p2.get("threads"):
        eng += ", {} of {} cpu threads".format(p2["threads"], p2["cores"])
    out.append("engine    " + eng)
    bits = [model_name]
    if p2.get("model_params"):
        bits.append("{} params".format(p2["model_params"]))
    if p2.get("model_size"):
        bits.append("{} on disk".format(p2["model_size"]))
    out.append("model     " + ", ".join(bits))

    n, total = p2["offload_n"], p2["offload_total"]
    if n is None:
        gline = "NO EVIDENCE (engine did not report layer placement)"
    elif n == 0:
        gline = "NOT ENGAGED: 0/{} layers on GPU".format(total)
    elif n < total:
        gline = "PARTIAL: {}/{} layers on GPU".format(n, total)
    else:
        gline = "ENGAGED: {}/{} layers on GPU".format(n, total)
    if p2["gpu_kind"] and p2["gpu_device"]:
        gline += " ({}: {})".format(p2["gpu_kind"], p2["gpu_device"])
    elif p2["gpu_kind"]:
        gline += " ({})".format(p2["gpu_kind"])
    out.append("gpu       " + gline)
    out.append("")
    out.append("            {:>14}  {:>14}  {:>14}".format(
        "prefill", "decode", "wallclock"))
    for name, p in (("pass 1", p1), ("pass 2", p2)):
        out.append("  {:<10}{:>14}  {:>14}  {:>14}".format(
            name, fmt_rate(p["prefill_toks"]), fmt_rate(p["decode_toks"]),
            fmt_rate(p["wallclock_toks"])))
    out.append("")

    wall = p1["wall_s"]
    load_s = (p1["load_ms"] or 0) / 1000.0
    prefill_s = (p1["prompt_ms"] or 0) / 1000.0
    decode_s = (p1["eval_ms"] or 0) / 1000.0
    other_s = max(0.0, wall - load_s - prefill_s - decode_s)
    out.append("where pass 1 went ({:.1f} s wall)".format(wall))
    if wall > 0:
        out.append(bar_line("load weights", load_s, load_s / wall))
        out.append(bar_line("prefill", prefill_s, prefill_s / wall))
        out.append(bar_line("decode", decode_s, decode_s / wall))
        out.append(bar_line("engine misc", other_s, other_s / wall))
    if cold_note:
        out.append("  note: " + cold_note)
    out.append("")
    out.append("VERDICT: " + state)
    out.extend(wrap_para(para))
    if explain_part:
        out.append("")
        out.append("YOUR NUMBER: " + explain_part[0])
        out.extend(wrap_para(explain_part[1]))
    out.append("=" * WIDTH)
    return "\n".join(out)


# -------------------------------------------------------------------- main

def save_cache(payload):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(payload, f, indent=1)
    except OSError:
        pass


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(
        prog="picchio",
        description="Knocks on your llama.cpp setup and listens for hollow "
                    "spots: are your tok/s numbers what you think they are, "
                    "and did the GPU actually do the work?",
    )
    ap.add_argument("model", nargs="?", help="path to a .gguf model file")
    ap.add_argument("--bin", help="engine binary (default: find "
                                  "llama-completion or llama-cli on PATH)")
    ap.add_argument("--explain", type=float, metavar="TOKS",
                    help="classify a tok/s number you saw somewhere against "
                         "this machine's measured rates")
    ap.add_argument("--json", action="store_true",
                    help="print raw measurements as JSON after the verdict")
    ap.add_argument("extra", nargs="*", default=[],
                    help="args after -- go straight to the engine "
                         "(e.g. -- --device none -ngl 0)")
    args = ap.parse_args()

    if args.model is None and args.explain is not None:
        cached = load_cache()
        if not cached:
            sys.exit("picchio: no previous run cached; run with a model "
                     "path first.")
        verdict, para = classify_number(args.explain, cached["rates"])
        print("YOUR NUMBER: {:.1f} tok/s -> {}".format(args.explain, verdict))
        print("\n".join(wrap_para(para)))
        print("(rates from {} on {})".format(
            cached.get("model_name", "?"), cached.get("stamp", "?")))
        return

    if args.model is None:
        ap.print_help()
        sys.exit(2)
    if not os.path.isfile(args.model):
        sys.exit("picchio: model file not found: {}".format(args.model))

    binpath = find_binary(args.bin)
    engine_str = engine_version(binpath)
    mach = machine_info()
    model_name = os.path.basename(args.model)

    sys.stderr.write("picchio: pass 1 (includes any cold load) ...\n")
    p1 = run_pass(binpath, args.model, args.extra)
    sys.stderr.write("picchio: pass 2 (warm) ...\n")
    p2 = run_pass(binpath, args.model, args.extra)

    cold_note = None
    l1, l2 = p1["load_ms"], p2["load_ms"]
    if l1 and l2 and l1 < 2 * l2 + 500:
        cold_note = "pass 1 was not a true cold start, weights were cached"

    state, para = diagnose(p1, p2)

    explain_part = None
    rates = {
        "prefill": p2["prefill_toks"] or p1["prefill_toks"],
        "decode": p2["decode_toks"] or p1["decode_toks"],
        "wallclock": p2["wallclock_toks"] or p1["wallclock_toks"],
    }
    if args.explain is not None:
        v, ep = classify_number(args.explain, rates)
        explain_part = ("{:.1f} tok/s -> {}".format(args.explain, v), ep)

    block = render_verdict(mach, engine_str, model_name, p1, p2, state,
                           para, explain_part, cold_note)
    print(block)

    save_cache({
        "stamp": time.strftime("%Y-%m-%d %H:%M"),
        "model_name": model_name,
        "rates": rates,
        "state": state,
    })

    if args.json:
        print(json.dumps({"machine": mach, "engine": engine_str,
                          "model": model_name, "pass1": p1, "pass2": p2,
                          "state": state}, indent=1))

    codes = {"HEALTHY": 0, "NO PLACEMENT EVIDENCE": 0,
             "PARTIAL OFFLOAD": 3, "SILENT CPU FALLBACK": 4}
    sys.exit(codes.get(state, 0))


if __name__ == "__main__":
    main()
