<div align="center">

<img src="assets/picchio-mark-a.svg" width="96" alt="pixel woodpecker on a trunk">

<h1>catch local LLM CPU fallback before you trust tok/s</h1>

<p>
<a href="https://github.com/logxio/picchio/actions/workflows/selftest.yml"><img src="https://github.com/logxio/picchio/actions/workflows/selftest.yml/badge.svg" alt="selftest"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="license: MIT"></a>
<img src="https://img.shields.io/badge/python-3.9%2B%2C%20stdlib%20only-3776ab" alt="python 3.9+, stdlib only">
</p>

<p><a href="#run-it">Run it</a> · <a href="https://logxio.github.io/picchio/">Browse real results</a> · <a href="#send-me-your-machine">Add your machine</a></p>

<img src="assets/picchio-demo.svg" width="680" alt="Picchio compares the same Qwen model on GPU and CPU and finds a 22x prefill drop">

</div>

A local LLM can return HTTP 200 and generate text while running **0 of 33
layers on the GPU**. I built Picchio to catch that in one run.

On the same Apple M5, with the same Qwen3.5-9B file:

| | GPU | silent CPU fallback |
|---|---:|---:|
| layers on GPU | 33/33 | 0/33 |
| prefill | 588.0 tok/s | 26.8 tok/s |
| decode | 21.1 tok/s | 12.2 tok/s |
| HTTP response | 200 | 200 |

Picchio reads the engine's placement report beside the operating system's
GPU meter, then shows prefill, decode, wall-clock speed, memory and power in
one result.

## Run it

```sh
curl -fsSL https://raw.githubusercontent.com/logxio/picchio/main/public/picchio.pyz -o picchio
chmod +x picchio
./picchio
```

With no arguments, Picchio finds Ollama tags, local GGUF files and models in
the Hugging Face and LM Studio caches. Pick one and it runs three passes.

You can also point it straight at a model or a running server:

```sh
./picchio model.gguf
./picchio qwen3.5:9b
./picchio http://127.0.0.1:8080
```

It runs on macOS and Linux with Python 3.9+. The download is one file and
uses only the standard library.

## Use the result

| I want to… | Run |
|---|---|
| catch CPU fallback and measure the run | `./picchio MODEL` |
| warn when a command leaves layers on the CPU | `./picchio guard -- COMMAND` |
| watch a loaded Ollama model or GPU process | `./picchio watch ollama --for 8` |
| catch a server that drops out of its normal lane | `./picchio monitor MODEL` |
| compare two runs and show the first changed setting | `./picchio compare before.txt after.txt` |
| paste a compact result into an issue or post | `./picchio MODEL --share row` |

Add `--json` when you want machine-readable output.

## What it gives you

- the exact number of model layers on the GPU
- separate prefill, decode and wall-clock speeds
- GPU activity, memory, power and energy per generated token
- the first setting that changed when you compare two runs
- a clear `HEALTHY`, `CPU FALLBACK` or `PARTIAL OFFLOAD` result

## Compare real machines

I measured these with Picchio:

| machine | model and engine | placement | prefill | decode | wall-clock |
|---|---|---|---:|---:|---:|
| Apple M5 | Qwen3.5-9B, llama.cpp Metal | 33/33 | 588.0 | 21.1 | 15.5 |
| Apple M5 | same file, forced CPU | 0/33 | 26.8 | 12.2 | 3.0 |
| RTX 4090 | same file, llama.cpp CUDA | 33/33 | 6763.3 | 138.0 | 25.2 |
| RTX 5090 | same file, llama.cpp CUDA | 33/33 | 9135.9 | 226.4 | 57.3 |
| RTX 5090 | same file, llama.cpp Vulkan | 33/33 | 6206.3 | 198.4 | 51.0 |
| RTX 5090 | qwen3.5:9b, Ollama | 100% GPU | 8614.7 | 193.5 | 153.9 |
| your machine | | | | | |

[Open every result](https://logxio.github.io/picchio/) or compare the outputs
in [examples/](examples/).

## What Picchio reads

- llama.cpp: per-layer placement, applied sampling settings and timing
- Ollama: CPU/GPU weight split and timing
- macOS: Apple GPU activity, memory, power and energy per token
- NVIDIA Linux: GPU activity, memory, power and energy per token through NVML
- AMD Linux: GPU activity and memory through amdgpu sysfs

Point Picchio at a GGUF path, an Ollama tag or a running llama-server URL.
The result tells you where the model ran and which number is safe to compare.

## Send me your machine

```sh
./picchio MODEL --share row > result.md 2> picchio.txt
```

[Add your result](https://github.com/logxio/picchio/issues/new?template=verdict-report.md).
If Picchio calls your run wrong, [send me that one first](https://github.com/logxio/picchio/issues/new?template=misdiagnosis-report.md).

## License

[MIT](LICENSE)
