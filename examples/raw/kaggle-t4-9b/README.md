# Author-run Linux T4 pair

This capture ran on one fresh Kaggle Linux VM on 2026-09-28. It is the
author's own run, not an independent user reproduction. The VM reported
Ubuntu 22.04.5, an Intel Xeon with 4 logical CPUs, 31 GiB of RAM, two
Tesla T4 cards with 15 GiB each, and NVIDIA driver 580.159.04. The
[GPU receipt](../../kaggle-t4-9b-gpu.txt) and
[forced-CPU receipt](../../kaggle-t4-9b-cpu.txt) came from successive
commands in that one kernel.

Both legs used the same 5,680,522,464-byte
`Qwen3.5-9B-Q4_K_M.gguf` from Unsloth's Apache-2.0
`Qwen3.5-9B-GGUF`, fixed at revision
`9f870da1e1c96da710c13926d36c6946bb7ebb38`. The capture
computed SHA-256
`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`;
see [fingerprints.json](fingerprints.json). The engine was the official
llama.cpp b11228 Linux CUDA 12.8 archive, with its archive hash in the
same file. Ubuntu 24.04's `glibc` and `libstdc++` packages were
extracted into a temporary directory to run that binary on the Kaggle
Ubuntu 22.04 image; [runtime.json](runtime.json) records their exact
packages and hashes. Picchio v1.0.0 mp1 was used without changing its
engine or measurement code.

The [GPU command](gpu.command.json) and
[CPU command](cpu.command.json) changed only the placement flags:
`-ngl 99` for GPU,
`--device none -ngl 0` for CPU. Both ran at context 4096 and
reported seed 7, temperature 0.8, top-k 40, top-p 0.95 and min-p 0.05.
Picchio ordinarily gives each invocation a random prompt prefix.
The saved [sitecustomize.py](sitecustomize.py) pinned only its
four-byte run id for this paired capture. Thus corresponding passes
sent the exact same 776-token prompt, while passes 01, 02 and 03 kept
different prefixes to prevent prefix-cache reuse. The three
`pass*.meta.json` files in each leg show those prefixes. The initial
16-token GPU/CPU smoke runs used one exact prompt, but their rates
are not used in the receipts.

Each leg contains three untouched llama.cpp stderr files, three
Picchio meta files, and the full Picchio NVML sample timeline with
pass marks. The engine reported 33/33 layers on GPU in the GPU leg
and 0/33 in the forced-CPU leg. Picchio's warm median prefill/decode
was 833.7/40.1 tok/s and 9.5/2.6 tok/s respectively. The NVML
compute-window work medians were 53% and 0%.

Run `python3 scripts/replay_receipts.py` from the repository root to
recalculate both receipts from those files and check their pairing,
model and tool fingerprints, exact per-pass prompt, timings and
telemetry marks. It needs neither a GPU nor a model download. It can
check the captured model SHA-256 against the pinned value, but it
cannot rehash model bytes that are not published here.

NVML measures the whole two-card fleet: Picchio takes the busiest
card's utilization and sums both cards' memory and power. It cannot
attribute activity to this process alone. The 20.2 W in the CPU
receipt is GPU idle draw, and its J/token figure is not model energy.
These speeds describe this VM and these placement flags; comparing
them with another machine does not show that Picchio accelerates
inference.
