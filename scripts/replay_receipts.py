#!/usr/bin/env python3
"""Recalculate public M5, Linux 27B and T4 receipts from captured inputs."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import picchio  # noqa: E402


class ReplayError(Exception):
    pass


def require(ok, message):
    if not ok:
        raise ReplayError(message)


def source_line(path, needle):
    lines = path.read_text(encoding="utf-8").splitlines()
    found = [i for i, line in enumerate(lines, 1) if needle in line]
    require(found, "{}: missing {!r}".format(path, needle))
    return "{}:{}".format(path.relative_to(ROOT), found[-1])


def replay(name, raw_name=None):
    raw_dir = ROOT / "examples" / "raw" / (raw_name or name)
    receipt_path = ROOT / "examples" / (name + ".txt")
    require(receipt_path.is_file(), "missing {}".format(receipt_path))
    passes, metas, model_paths = [], [], []
    for number in range(1, 4):
        stem = raw_dir / "pass{}".format(number)
        meta_path = Path(str(stem) + ".meta.json")
        stderr_path = Path(str(stem) + ".stderr.txt")
        require(meta_path.is_file() and stderr_path.is_file(),
                "missing pass {} inputs in {}".format(number, raw_dir))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stderr = stderr_path.read_text(encoding="utf-8")
        parsed = picchio.parse_stderr(stderr, meta["wall_s"])
        paths = re.findall(r"loaded meta data .*? from (.+?) \(version GGUF",
                           stderr)
        require(paths and len(set(paths)) == 1,
                "{}: model path is missing or inconsistent".format(
                    stderr_path))
        model_paths.append(paths[0])
        for field in ("offload_n", "offload_total", "prompt_tokens",
                      "prompt_ms", "eval_tokens", "eval_ms", "load_ms",
                      "prefill_toks", "decode_toks", "wallclock_toks"):
            require(parsed[field] is not None,
                    "{}: missing {}".format(stderr_path, field))
        parsed["nonce"] = meta.get("prompt_nonce")
        passes.append(parsed)
        metas.append(meta)
    require(len({(m["model_name"], m["engine"],
                  tuple(m.get("extra_args", []))) for m in metas}) == 1,
            "{}: pass metadata disagree".format(raw_dir))
    require(len({(p["offload_n"], p["offload_total"])
                 for p in passes}) == 1,
            "{}: placement differs across passes".format(raw_dir))
    require(len(set(model_paths)) == 1,
            "{}: model path differs across passes".format(raw_dir))
    tele_path = raw_dir / "telemetry.json"
    require(tele_path.is_file(), "missing {}".format(tele_path))
    raw = json.loads(tele_path.read_text(encoding="utf-8"))
    require(raw.get("samples") and len(raw.get("marks", [])) == 3,
            "{}: expected samples and three pass marks".format(tele_path))
    saved = raw.get("summary", {})
    tele = picchio.telemetry_summary(
        raw["samples"], raw["marks"], saved.get("throttled", False),
        saved.get("src"))
    for field in ("idle_med", "work_med", "mem_step", "work_w", "dec_w"):
        require(tele[field] is not None,
                "{}: cannot derive {}".format(tele_path, field))
    rep = picchio.build_rep(passes)
    state, para = picchio.diagnose(passes[0], rep, "llama.cpp", tele)
    extra = metas[0].get("extra_args", [])
    why = picchio.attribute_why(state, rep, "llama.cpp", extra)
    l1, l2 = passes[0]["load_ms"], passes[1]["load_ms"]
    cold_note = l1 < 2 * l2 + 500
    got = picchio.render_verdict(
        picchio.machine_info(), metas[0]["engine"],
        metas[0]["model_name"], passes, state, para, "llama.cpp",
        None, cold_note, why,
        metas[0].get("ctx", picchio.effective_ctx(extra)), extra, tele)
    want = receipt_path.read_text(encoding="utf-8").rstrip()
    got_lines, want_lines = got.splitlines(), want.splitlines()
    # machine_info() describes the reader's computer. The recorded footer
    # belongs to the capture machine and is checked across the paired runs.
    for number, (actual, published) in enumerate(
            zip(got_lines[:-1], want_lines[:-1]), 1):
        require(actual == published,
                "{}:{} differs\n  published: {}\n  replayed:  {}".format(
                    receipt_path.relative_to(ROOT), number,
                    published, actual))
    require(len(got_lines) == len(want_lines),
            "{}: line count differs".format(receipt_path))
    block = picchio.parse_block(want)
    require(block and block["verdict"] == state,
            "{}: verdict cannot be read back".format(receipt_path))
    require(block["model"] == metas[0]["model_name"]
            and block["engine"] == metas[0]["engine"],
            "{}: model or engine footer disagrees with metadata".format(
                receipt_path))
    return {"name": name, "block": block, "passes": passes,
            "metas": metas, "tele": tele, "state": state,
            "model_path": model_paths[0],
            "marks": raw["marks"],
            "sources": [str((raw_dir / "pass2.meta.json").relative_to(ROOT)),
                        source_line(raw_dir / "pass2.stderr.txt",
                                    "offloaded "),
                        source_line(raw_dir / "pass2.stderr.txt",
                                    "prompt eval time ="),
                        source_line(raw_dir / "pass2.stderr.txt",
                                    "eval time ="),
                        str(tele_path.relative_to(ROOT)) + " (marks, samples)"],
            "receipt": str(receipt_path.relative_to(ROOT))}


def check_pair(gpu, cpu):
    a, b = gpu["block"], cpu["block"]
    require((a["chip"], a["ram"], a["os"], a["model"], a["engine"],
             a["ctx"], a["quant"], a["settings"]) ==
            (b["chip"], b["ram"], b["os"], b["model"], b["engine"],
             b["ctx"], b["quant"], b["settings"]),
            "M5 receipts disagree on machine, model or settings")
    require(a["chip"] == "Apple M5" and a["ram"] == "32"
            and a["ctx"] == 4096,
            "M5 comparison identity does not match the public claim")
    require(gpu["metas"][0].get("extra_args", []) == []
            and cpu["metas"][0]["extra_args"] ==
            ["--device", "none", "-ngl", "0"],
            "M5 placement intervention differs from recorded flags")
    require(gpu["model_path"] == cpu["model_path"]
            and gpu["passes"][0]["gpu_device"] == "Apple M5",
            "M5 logs disagree on model path or Metal device")
    for number, (gpu_pass, cpu_pass) in enumerate(
            zip(gpu["passes"], cpu["passes"]), 1):
        for field in ("prompt_tokens", "eval_tokens", "threads", "cores",
                      "model_params", "model_size", "sampling"):
            require(gpu_pass[field] == cpu_pass[field],
                    "M5 pass {} differs on {}".format(number, field))
    require(gpu["passes"][0]["offload_n"] == 33
            and cpu["passes"][0]["offload_n"] == 0
            and gpu["passes"][0]["offload_total"] == 33
            and cpu["passes"][0]["offload_total"] == 33,
            "M5 layer contrast is not 33/33 versus 0/33")


def check_kaggle_pair(gpu, cpu):
    raw_dir = ROOT / "examples" / "raw" / "kaggle-t4-9b"
    fingerprints = json.loads(
        (raw_dir / "fingerprints.json").read_text(encoding="utf-8"))
    model_sha = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
    require(fingerprints["model"] == "Qwen3.5-9B-Q4_K_M.gguf"
            and fingerprints["model_bytes"] == 5680522464
            and fingerprints["model_sha256"] ==
            fingerprints["expected_model_sha256"] == model_sha,
            "examples/raw/kaggle-t4-9b/fingerprints.json: "
            "captured model identity, size or SHA-256 differs")
    require(fingerprints["picchio_zipapp_sha256"] ==
            "87632c11bce5afb97d012e8920e0f7b28bedfced635c5c3bd0ec744b5336da94"
            and fingerprints["engine_archive_sha256"] ==
            "c10e4c58b2ff953e69564fb5e37fddc5a105ec00e8e278e1ebeab3de374e6787",
            "examples/raw/kaggle-t4-9b/fingerprints.json: "
            "captured tool or engine archive SHA-256 differs")
    nonce = json.loads(
        (raw_dir / "nonce-control.json").read_text(encoding="utf-8"))
    site_path = raw_dir / "sitecustomize.py"
    require(hashlib.sha256(site_path.read_bytes()).hexdigest() ==
            nonce["sitecustomize_sha256"] and nonce["run_id"] == "44332211",
            "examples/raw/kaggle-t4-9b/sitecustomize.py: "
            "controlled prompt source or run id differs")
    query = (raw_dir / "nvidia-query.stdout.txt").read_text(
        encoding="utf-8").splitlines()
    require(len(query) == 2 and all(", Tesla T4, GPU-" in line
                                    and ", 580.159.04, 15360 MiB, 7.5" in line
                                    for line in query),
            "examples/raw/kaggle-t4-9b/nvidia-query.stdout.txt: "
            "captured two-card T4 machine differs")
    require("22.04.5 LTS" in (raw_dir / "os-release.stdout.txt").read_text(
        encoding="utf-8"),
            "examples/raw/kaggle-t4-9b/os-release.stdout.txt: OS differs")
    require("build 11228" in (raw_dir / "llama-version.stderr.txt").read_text(
        encoding="utf-8"),
            "examples/raw/kaggle-t4-9b/llama-version.stderr.txt: "
            "engine build differs")
    require("CUDA0: Tesla T4" in (raw_dir / "llama-devices.stdout.txt").read_text(
        encoding="utf-8"),
            "examples/raw/kaggle-t4-9b/llama-devices.stdout.txt: "
            "engine did not list a T4")
    done = json.loads((raw_dir / "capture-end.json").read_text(
        encoding="utf-8"))
    require(done["success"] is True and (raw_dir / "SUCCESS.json").is_file(),
            "examples/raw/kaggle-t4-9b/capture-end.json: capture did not finish")

    a, b = gpu["block"], cpu["block"]
    for field in ("chip", "ram", "os", "model", "engine", "ctx",
                  "quant", "settings", "threads", "protocol"):
        require(a[field] == b[field],
                "Kaggle pair differs on receipt {}".format(field))
    require(a["chip"] == "Intel(R) Xeon(R) CPU @ 2.00GHz + Tesla T4 x2"
            and a["os"].startswith("Linux ")
            and a["model"] == "Qwen3.5-9B Q4_K_M"
            and a["engine"] == "llama.cpp b0.5.0-dev"
            and a["ctx"] == 4096,
            "Kaggle receipt identity differs from recorded machine/model")
    require(gpu["model_path"] == cpu["model_path"]
            and gpu["model_path"].endswith("/Qwen3.5-9B-Q4_K_M.gguf"),
            "examples/raw/kaggle-t4-9b/*/pass*.stderr.txt: "
            "model paths do not match")
    for leg, run, args, offload in (
            ("gpu", gpu, ["-ngl", "99"], 33),
            ("cpu", cpu, ["--device", "none", "-ngl", "0"], 0)):
        command = json.loads((raw_dir / (leg + ".command.json")).read_text(
            encoding="utf-8"))
        require(command["returncode"] == (0 if leg == "gpu" else 4)
                and command["argv"][-len(args):] == args
                and command["argv"][0:2] ==
                ["env", "PYTHONPATH=/tmp/picchio-fresh-linux/fixed-nonce"],
                "examples/raw/kaggle-t4-9b/{}.command.json: "
                "capture command or exit differs".format(leg))
        for number, (p, meta, mark) in enumerate(
                zip(run["passes"], run["metas"], run["marks"]), 1):
            stem = "examples/raw/kaggle-t4-9b/{}/pass{}".format(
                leg, number)
            require(p["offload_n"] == offload
                    and p["offload_total"] == 33
                    and p["gpu_device"] == "Tesla T4",
                    stem + ".stderr.txt: CUDA layer placement differs")
            require(meta["ctx"] == 4096
                    and meta["prompt_nonce"] ==
                    "Run 44332211 pass {:02d}. ".format(number)
                    and p["prompt_tokens"] == 776
                    and p["eval_tokens"] == 127,
                    stem + ": prompt, context or token counts differ")
            stderr = (ROOT / (stem + ".stderr.txt")).read_text(
                encoding="utf-8")
            require(re.search(r"I llama_context: n_ctx\s+=\s+4096\b", stderr),
                    stem + ".stderr.txt: engine context differs")
            for mark_field, expected in (
                    ("wall_s", meta["wall_s"]),
                    ("load_s", p["load_ms"] / 1000),
                    ("prompt_s", p["prompt_ms"] / 1000),
                    ("eval_s", p["eval_ms"] / 1000)):
                require(abs(mark[mark_field] - expected) < 0.00001,
                        "examples/raw/kaggle-t4-9b/{}/telemetry.json: "
                        "pass {} {} disagrees with meta/stderr".format(
                            leg, number, mark_field))
    gpu_cmd = json.loads((raw_dir / "gpu.command.json").read_text(
        encoding="utf-8"))
    cpu_cmd = json.loads((raw_dir / "cpu.command.json").read_text(
        encoding="utf-8"))
    prefix_end = gpu_cmd["argv"].index("--keep-logs")
    require(gpu_cmd["argv"][:prefix_end] ==
            cpu_cmd["argv"][:cpu_cmd["argv"].index("--keep-logs")]
            and gpu_cmd["end_utc"] <= cpu_cmd["start_utc"]
            and gpu_cmd["argv"][4] == gpu["model_path"],
            "examples/raw/kaggle-t4-9b/{gpu,cpu}.command.json: "
            "legs were not sequential on one model and engine")
    for number, (p, q) in enumerate(zip(gpu["passes"], cpu["passes"]), 1):
        for field in ("prompt_tokens", "eval_tokens", "threads", "cores",
                      "model_params", "model_size", "sampling", "nonce"):
            require(p[field] == q[field],
                    "Kaggle paired pass {} differs on {}".format(
                        number, field))
    require(gpu["tele"]["src"] == cpu["tele"]["src"] == "nvml"
            and gpu["tele"]["work_med"] >= 30
            and cpu["tele"]["work_med"] <= 5
            and gpu["tele"]["mem_step"] >= 5 * 1024 ** 3
            and cpu["tele"]["mem_step"] < 100 * 1024 ** 2,
            "examples/raw/kaggle-t4-9b/*/telemetry.json: "
            "whole-GPU activity or memory contrast is missing")


def check_linux_27b(run):
    block, passes, metas, marks = (run["block"], run["passes"],
                                   run["metas"], run["marks"])
    require(block["chip"].endswith("GeForce RTX 5090")
            and block["os"].startswith("Linux ")
            and block["model"] == "Qwen3.8-27B Q4_K_M"
            and block["engine"] == "llama.cpp b0.2.0-dev"
            and block["ctx"] == 4096,
            "{}: published Linux 27B identity or context differs".format(
                run["receipt"]))
    require(run["model_path"] == "Qwen3.8-27B-UD-Q4_K_M.gguf"
            and all(p["offload_n"] == p["offload_total"] == 66
                    and p["gpu_device"] == "GeForce RTX 5090"
                    for p in passes),
            "examples/raw/linux-5090-27b/pass*.stderr.txt: CUDA model or "
            "layer placement differs")
    for number, (parsed, meta, mark) in enumerate(
            zip(passes, metas, marks), 1):
        stderr_path = ROOT / "examples" / "raw" / "linux-5090-27b" / \
            "pass{}.stderr.txt".format(number)
        stderr = stderr_path.read_text(encoding="utf-8")
        require(re.search(r"I llama_context: n_ctx\s+=\s+4096\b", stderr),
                "{}: context differs from 4096".format(stderr_path))
        for field in ("prompt_tokens", "eval_tokens", "threads", "cores",
                      "model_params", "model_size", "sampling"):
            require(parsed[field] == passes[0][field],
                    "examples/raw/linux-5090-27b/pass{}.stderr.txt: "
                    "{} differs across passes".format(number, field))
        for mark_field, expected in (("wall_s", meta["wall_s"]),
                                     ("load_s", parsed["load_ms"] / 1000),
                                     ("prompt_s", parsed["prompt_ms"] / 1000),
                                     ("eval_s", parsed["eval_ms"] / 1000)):
            require(abs(mark[mark_field] - expected) < 0.00001,
                    "examples/raw/linux-5090-27b/telemetry.json: pass {} "
                    "{} disagrees with meta/stderr".format(
                        number, mark_field))
    require(block["threads"] == "8/16"
            and block["settings"] ==
            "temp 1.0, top-k 20, top-p 0.95, min-p 0.05, seed 7"
            and block["rates"][1] == 81.5
            and run["tele"]["work_med"] == 92,
            "{}: 27B settings, decode or GPU work differ".format(
                run["receipt"]))


def check_27b():
    path = ROOT / "examples" / "windows-4070s-27b.txt"
    text = path.read_text(encoding="utf-8")
    block = picchio.parse_block(text)
    require(block and block["verdict"] == "PARTIAL OFFLOAD",
            "{}: expected a partial-offload receipt".format(path))
    placement = re.search(r"(\d+)/(\d+) layers on GPU", block["place"])
    remaining = re.search(r"PARTIAL OFFLOAD\. (\d+) layers sat on CPU\.",
                          text)
    require(placement and remaining
            and int(placement[2]) - int(placement[1]) == int(remaining[1]),
            "{}: GPU and CPU layers do not sum to total".format(path))
    require(block["rates"][1] == 5.7,
            "{}: warm decode differs from 5.7 tok/s".format(path))
    return path.relative_to(ROOT), int(placement[2]), int(placement[1]), \
        int(remaining[1]), block["rates"][1]


def main():
    argparse.ArgumentParser(
        description="Recalculate bundled M5, Linux 27B and T4 receipts without GPU or model"
    ).parse_args()
    try:
        gpu = replay("healthy-metal")
        cpu = replay("cpu-fallback")
        check_pair(gpu, cpu)
        for run in (gpu, cpu):
            block, tele = run["block"], run["tele"]
            print("PASS {}: {} ({})".format(
                run["receipt"], run["state"], block["place"]))
            print("  warm prefill / decode / wall: {} / {} / {} tok/s".format(
                *["{:.1f}".format(n) for n in block["rates"]]))
            print("  GPU idle / work: {:.0f}% / {:.0f}%; memory +{:.1f} GiB; "
                  "decode {:.1f} W".format(
                      tele["idle_med"], tele["work_med"],
                      tele["mem_step"] / 1024 ** 3, tele["dec_w"]))
            print("  sources: " + "; ".join(run["sources"]))
        print("PASS recorded pair: Apple M5, 32 GB, same Qwen3.5-9B "
              "model path, llama.cpp b9430, ctx 4096, sampling and "
              "token counts; placement 33/33 vs 0/33 via "
              "--device none -ngl 0")
        print("  boundary: shared machine and file bytes have no "
              "independent fingerprint; OS GPU samples cover the whole "
              "GPU, and the runs differ in placement flags")
        t4_gpu = replay("kaggle-t4-9b-gpu", "kaggle-t4-9b/gpu")
        t4_cpu = replay("kaggle-t4-9b-cpu", "kaggle-t4-9b/cpu")
        check_kaggle_pair(t4_gpu, t4_cpu)
        for run in (t4_gpu, t4_cpu):
            block = run["block"]
            print("PASS {}: {} ({}); warm prefill / decode / wall "
                  "{:.1f} / {:.1f} / {:.1f} tok/s; NVML work {:.0f}%".format(
                      run["receipt"], run["state"], block["place"],
                      *block["rates"], run["tele"]["work_med"]))
            print("  sources: " + "; ".join(run["sources"]))
        print("PASS fresh Linux pair: one author-run Kaggle VM with two "
              "Tesla T4 cards, same Qwen3.5-9B file and engine, ctx 4096, "
              "sampling, and exact per-pass prompt (776 tokens); "
              "placement 33/33 vs 0/33 via --device none -ngl 0")
        print("  captured model SHA-256: "
              "examples/raw/kaggle-t4-9b/fingerprints.json; "
              "prompt control: examples/raw/kaggle-t4-9b/sitecustomize.py")
        print("  boundary: self-run, not an independent user; NVML work is "
              "the busiest card and memory/power sum both whole cards; "
              "the forced-CPU GPU watts are idle draw, not model energy")
        linux = replay("linux-5090-27b")
        check_linux_27b(linux)
        print("PASS {}: {} ({}); warm prefill / decode / wall "
              "{:.1f} / {:.1f} / {:.1f} tok/s; GPU work {:.0f}%".format(
                  linux["receipt"], linux["state"],
                  linux["block"]["place"], *linux["block"]["rates"],
                  linux["tele"]["work_med"]))
        print("  NVML idle / work: {:.0f}% / {:.0f}%; memory +{:.1f} GiB; "
              "work {:.0f} W; decode {:.2f} J/token".format(
                  linux["tele"]["idle_med"],
                  linux["tele"]["work_med"],
                  linux["tele"]["mem_step"] / 1024 ** 3,
                  linux["tele"]["work_w"],
                  linux["tele"]["dec_w"] / linux["block"]["rates"][1]))
        linux_dir = ROOT / "examples" / "raw" / "linux-5090-27b"
        print("  sources: " + "; ".join(
            str((linux_dir / "pass{}.meta.json".format(n)).relative_to(ROOT))
            for n in (1, 2, 3)))
        print("  placement: " + "; ".join(
            source_line(linux_dir / "pass{}.stderr.txt".format(n),
                        "offloaded ") for n in (1, 2, 3)))
        print("  timing: " + "; ".join(
            source_line(linux_dir / "pass{}.stderr.txt".format(n), needle)
            for n in (2, 3)
            for needle in ("prompt eval time =", "eval time =")))
        print("  GPU samples: " + linux["sources"][-1])
        print("  boundary: one recorded Linux CUDA machine; OS GPU samples "
              "cover the whole GPU, not just the model process")
        path, total, on_gpu, on_cpu, decode = check_27b()
        print("PASS {}: {} - {} = {} CPU layers; warm decode "
              "{:.1f} tok/s (receipt arithmetic only; no public "
              "4070 SUPER 27B raw logs to replay)".format(
                  path, total, on_gpu, on_cpu, decode))
        return 0
    except (ReplayError, OSError, KeyError, ValueError, TypeError) as exc:
        print("FAIL receipt replay: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
