# Verify local AI runs on your laptop

Verify four captured Apple M5 runs and run fresh boundary and SQLite checks on your computer. Python 3.11 or newer and the files in this repository are enough to start.

## Quick Start

```sh
PYTHONPATH=workbench/runner python3 -m humanscale run --scenario all --output /tmp/picchio_verify
python3 workbench/receipts/replay_workbench_receipts.py
```

The first command verifies every captured artifact hash, source snapshot, decision and database row. It runs fresh OS boundary probes and a frozen test and SQLite exercise on the host. Workspace tests turn green while the ledger remains at 13. The frozen tests catch the mismatch. A real repair then produces 42 and passes both checks. The command writes `/tmp/picchio_verify/receipt.json` and exits 0. Repeating the command is supported. An omitted model or a model path that does not exist selects this same instant verification route.

The host boundary adapter uses Seatbelt on macOS and Landlock with seccomp on supported Linux kernels. The receipt records the actual adapter and each probe result. On other systems the captured boundary evidence and live SQLite checks remain available, and the host boundary adapter is marked unavailable. Captured inference measurements keep their Apple M5 identity; the instant route performs fresh deterministic host checks.

## Run fresh model inference

On macOS, install llama.cpp and supply your own GGUF file below 3 GiB:

```sh
PYTHONPATH=workbench/runner python3 -m humanscale run \
  --model /absolute/path/to/your/model.gguf \
  --scenario all --output /tmp/picchio_inference
```

The existing model route enforces the Seatbelt policy across the agent, server and tools. Use a new output directory for each inference run. Installing with `pip install ./workbench/runner` provides the standalone `humanscale` command for model execution. Both routes use the standard library. Verification requires the captured receipts included in the clone.

## What One Run Proves

1. **Hardware and Context Reality**: Records engine layer placement (`29/29` GPU layers on Apple Silicon Metal at `673.5 tok/s` prefill and `86.1 tok/s` decode vs. `0/29` layers on CPU at `81.8 tok/s` prefill and `35.3 tok/s` decode), resident memory envelope (`1.19 GiB` peak RSS within a `4 GiB` budget), and boundary context markers (`First`, `Middle`, `Tail` nonces) at the inference boundary.
2. **Process Tree Boundary Under Native GPU**: Wraps the agent, inference server, and all tool child processes inside an inherited OS Seatbelt policy. Writes are restricted to the authorized task workspace, and network sockets are restricted to the single local loopback inference port. Five active out of bounds probes (`outside_read`, `outside_write`, `protected_test_read`, `protected_test_write`, and `external_socket`) are blocked at the kernel level (`EPERM`, `0` bytes leaked) while Metal GPU inference runs at full speed.
3. **Independent End State and Frozen Check Verification**: Freezes the original test suite outside the agent's writable directory before execution starts. When the agent finishes and claims success, the observer replays both the workspace tests and the frozen original tests in parallel and inspects the final SQLite ledger (`ledger.sqlite`) through an independent read only connection, catching deleted assertions, injected test skips, and incomplete database updates (`13` vs. target `42`).

## Receipt Schema and Interoperability

Each run writes a deterministic `receipt.json` and a path sanitized `receipt.public.json` for zero install browser replay:

* **LocalScore Compatibility**: Exports `avg_prompt_tps`, `avg_gen_tps`, and `avg_ttft_ms` alongside raw `llama.cpp` server timings.
* **Inspect AI Compatibility**: Exports `inspect.samples.jsonl` and `inspect_adapter.py` so evaluation pipelines can load and rescore recorded runs directly without rerunning model inference:

```sh
PYTHONPATH=workbench/runner inspect eval workbench/runner/inspect_adapter.py@recorded_receipts --model mockllm/model
```

## Wrapping Existing Local Agents and MCP Tools

Use `humanscale guard` to place any local agent, `llamafile`, or MCP tool command inside the same kernel file and network boundary:

```sh
humanscale guard --workspace /path/to/task --model /path/to/model.gguf \
  --port 18080 --output /path/to/external_audit -- python3 /path/to/task/agent.py
```

## License

The HumanScale Workbench runner, browser replay interface, and verification modules are released under the MIT License. Model weights (`Qwen2.5-Coder-1.5B-Instruct-GGUF`) retain their upstream Apache 2.0 license, and `llama.cpp` retains its MIT license.
