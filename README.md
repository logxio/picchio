<div align="center">

<img src="assets/picchio-mark-a.svg" width="96" alt="pixel woodpecker on a trunk">

<h1>Inference Placement Receipts for local LLMs</h1>

<p>
<a href="https://github.com/logxio/picchio/actions/workflows/selftest.yml"><img src="https://github.com/logxio/picchio/actions/workflows/selftest.yml/badge.svg" alt="selftest"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="license: MIT"></a>
<img src="https://img.shields.io/badge/python-3.9%2B%2C%20stdlib%20only-3776ab" alt="python 3.9+, stdlib only">
</p>

<p><a href="#run-it">Run it</a> · <a href="https://logxio.github.io/picchio/">Browse results</a> · <a href="#send-me-your-machine">Add your machine</a></p>

<p>An inference placement receipt joins the engine's layer placement, the system's GPU readings and the run's separate speed measurements. It shows where a local model actually ran when a single tok/s number cannot. A 27B model still answered at 5.7 tok/s, but its <a href="examples/windows-4070s-27b.txt">recorded receipt</a> showed 28 of 66 layers on the CPU and a memory-fit cause. <a href="examples/contest-demo-v3.mp4">Watch the demo</a>.</p>

<img src="assets/picchio-demo.svg" width="680" alt="Recorded Qwen3.8-27B run: 38 of 66 layers on GPU, 28 on CPU, 5.7 tok/s warm decode">

</div>

A separate [Linux CUDA 27B receipt](examples/linux-5090-27b.txt) records all
66 layers on an RTX 5090, 92% GPU work and 81.5 tok/s warm decode. Recalculate
it from [three captured passes and NVML samples](examples/raw/linux-5090-27b/),
alongside the Apple M5 GPU and forced-CPU runs, without a GPU or model download:

```sh
python3 scripts/replay_receipts.py
```

The command checks each published receipt against its [captured inputs](examples/raw/).
A mismatch points to the affected file or receipt line and exits nonzero.

## Run it

```sh
curl -fsSL https://raw.githubusercontent.com/logxio/picchio/main/public/picchio.pyz -o picchio
chmod +x picchio
./picchio
```

On Windows, in PowerShell:

```powershell
curl.exe -fsSL https://raw.githubusercontent.com/logxio/picchio/main/public/picchio.pyz -o picchio.pyz
python picchio.pyz
```

With no arguments, Picchio finds Ollama tags, local GGUF files and models in
the Hugging Face and LM Studio caches. Pick one and it runs three passes.

You can also point it straight at a model or a running server:

```sh
./picchio model.gguf
./picchio qwen3.5:9b
./picchio http://127.0.0.1:8080
```

It runs on macOS, Linux and Windows with Python 3.9+. The download is one
file and uses only the standard library.

## Next

- Collect more Linux Radeon runs to check GPU activity and multi-card results against real engine output, building on the [RX 7900 XTX report](https://github.com/logxio/picchio/issues/1).
- Verify the complete Ollama path on Windows hardware, from finding a model to reporting where it ran.

Public work spans July through September 2026, with [v1.0.0](https://github.com/logxio/picchio/releases/tag/v1.0.0) released in August. In [Issue #1](https://github.com/logxio/picchio/issues/1), a Radeon user reported missing speed readings and an incorrect cold-run breakdown; the maintainer replied with a fix and requested a new verdict. The user has not posted that rerun, so the corrected result on that machine remains unverified.

[Contribute a run or a fix](CONTRIBUTING.md).

## Use the result

| I want to… | Run |
|---|---|
| catch CPU fallback and measure the run | `./picchio MODEL` |
| know whether a model fits here before I download it | `./picchio plan https://huggingface.co/.../model-Q4_K_M.gguf` · `./picchio plan qwen3.5:9b` |
| check that fit at my own context and KV type | `./picchio plan MODEL --ctx 262144 --kv q8_0` |
| guard something already running and get told the moment it leaves the GPU: your own command, a loaded model, a server | `./picchio guard -- COMMAND` · `./picchio guard ollama` · `./picchio guard http://127.0.0.1:8080` |
| compare two runs and show the first changed setting | `./picchio compare before.txt after.txt` |
| turn one run into a complete Ollama or llama.cpp Issue report | `./picchio MODEL --share bug-report` |

Add `--json` when you want machine-readable output.
Add `--ctx 262144` when the problem only appears at a larger context.

The fit check answers before the download, from the file header and the
registry manifest, and it answers in one word and one move. In the
browser: [will it fit?](https://logxio.github.io/picchio/fit.html)

```
FITS. The whole model fits in this machine's memory.
  5.3 GiB to download
```

## What it gives you

- the exact number of model layers on the GPU
- separate prefill, decode and wall-clock speeds
- GPU activity, memory, power and energy per generated token
- the first setting that changed when you compare two runs
- a clear `HEALTHY`, `CPU FALLBACK` or `PARTIAL OFFLOAD` result
- a paste-ready GitHub Issue with your machine, model, GPU placement and measured rates

## Compare machines

I measured these with Picchio:

| machine | model and engine | placement | prefill | decode | wall-clock |
|---|---|---|---:|---:|---:|
| Apple M5 | Qwen3.5-9B, llama.cpp Metal | 33/33 | 588.0 | 21.1 | 15.5 |
| Apple M5 | same file, forced CPU | 0/33 | 26.8 | 12.2 | 3.0 |
| RTX 4090 | same file, llama.cpp CUDA | 33/33 | 6763.3 | 138.0 | 25.2 |
| RTX 5090 | same file, llama.cpp CUDA | 33/33 | 9135.9 | 226.4 | 57.3 |
| RTX 5090 | same file, llama.cpp Vulkan | 33/33 | 6206.3 | 198.4 | 51.0 |
| RTX 5090 | qwen3.5:9b, Ollama | 100% GPU | 8614.7 | 193.5 | 153.9 |
| RTX 4070 SUPER | same file, llama.cpp CUDA | 33/33 | 4187.8 | 78.2 | 28.9 |
| your machine | | | | | |

[Open every result](https://logxio.github.io/picchio/) or compare the outputs
in [examples/](examples/).

## What Picchio reads

- llama.cpp: per-layer placement, applied sampling settings and timing
- Ollama: CPU/GPU weight split and timing
- macOS: Apple GPU activity, memory, power and energy per token
- NVIDIA on Linux and Windows: GPU activity, memory, power and energy per token through NVML
- AMD on Linux: GPU activity and memory through amdgpu sysfs

Point Picchio at a GGUF path, an Ollama tag or a running llama-server URL.
The result tells you where the model ran and which number is safe to compare.

### One run, two sources of evidence

The [llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) shows `backends`, `n_gpu_layers`, and repeated-test `avg_ts` / `stddev_ts` output. Its `n_gpu_layers` is a test setting, not a measurement of GPU activity. The [llama-cli documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md) includes `--show-timings`. Picchio reads the engine's output and samples the OS during that same run:

| Input and source | Output in the public Apple M5 / Qwen3.5-9B receipts | What it can establish |
|---|---|---|
| llama.cpp stderr from each pass | `offloaded 33/33` with prompt/eval timings in the [GPU run](examples/raw/healthy-metal/); `offloaded 0/33` in the [forced CPU run](examples/raw/cpu-fallback/) | Actual layer placement and engine timing as reported by llama.cpp; not independent proof that the GPU was busy. |
| Picchio's time-aligned macOS GPU samples from those same passes | [GPU run](examples/healthy-metal.txt): 99% work, +6.0 GiB, 11.0 W, 0.52 J/token; [forced CPU run](examples/cpu-fallback.txt): 5% work, +0.3 GiB, 0.1 W, 0.01 J/token | OS-side activity, memory and energy alongside the engine report; these are whole-GPU readings, not per-process attribution or total-system energy. |
| Picchio's combined result | `HEALTHY` versus `SILENT CPU FALLBACK`, with `--device none -ngl 0` identified in the CPU run | A diagnosis for each recorded run. The different placement and rates come from the GPU settings, not from Picchio accelerating inference. |

The two M5 receipts report one Apple M5, the same model path, engine build, context and sampling settings; the forced-CPU run changes `--device none -ngl 0`. The raw files contain no independent machine or model-byte fingerprint, and the OS samples measure the whole GPU. No `llama-bench` throughput number is mixed into this comparison. The [Linux 27B receipt](examples/linux-5090-27b.txt) replays from its [three recorded CUDA passes and NVML samples](examples/raw/linux-5090-27b/); the OS samples measure the whole GPU, and the raw files have no independent machine or model-byte fingerprint. The [4070 SUPER 27B receipt](examples/windows-4070s-27b.txt) can be checked for its own arithmetic (`66 − 38 = 28` CPU layers); its raw layer and timing logs are not public, so the replay command does not independently reproduce its measurements. Speeds across the two 27B machines are not a controlled comparison or evidence that Picchio accelerates inference.

## Send me your machine

```sh
./picchio MODEL --share row > result.md 2> picchio.txt
```

[Add your result](https://github.com/logxio/picchio/issues/new?template=verdict-report.md).
If Picchio calls your run wrong, [send me that one first](https://github.com/logxio/picchio/issues/new?template=misdiagnosis-report.md).

## License

[MIT](LICENSE)
