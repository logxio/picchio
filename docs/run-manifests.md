# Picchio run manifests

`picchio run MANIFEST` is the AI-facing long-run entry. It always writes
progress to stderr, exactly one final JSON object to stdout, and durable
evidence beside the manifest. Repeating the same command resumes from the
last atomically committed case.

## Queue

```json
{
  "schema": "picchio.queue.manifest.v1",
  "adapter": {
    "type": "http",
    "url": "http://127.0.0.1:8080/v1/chat/completions",
    "timeoutSeconds": 600,
    "requestTemplate": {"model": "local-model", "stream": false}
  },
  "runtime": {
    "pid": 12345,
    "stderrPath": "/absolute/path/server.stderr.txt"
  },
  "safety": {
    "minFreePercent": 5,
    "minReclaimableBytes": 2147483648,
    "maxSwapoutDeltaBytes": 1073741824
  },
  "cases": [
    {
      "id": "reason-01",
      "category": "reasoning",
      "maxOutputTokens": 256,
      "request": {"messages": [{"role": "user", "content": "..."}]},
      "rubric": {"contains": ["CODEWORD"]}
    }
  ]
}
```

`runtime.pid` adopts an already running server. Alternatively,
`runtime.launchCommand` is a string array and `runtime.ready.url` is polled
until ready. Picchio records the exact executable path, PID, start time and
parent PID. It never signals the runtime. A launched runtime is therefore
left running for the caller to own.

An HTTP adapter sends the case request as JSON. A command adapter receives
one `picchio.adapter.request.v1` object on stdin and must return one JSON
object on stdout:

Artifacts preserve the manifest verbatim. Do not put credentials in
`headers` or `env`; inherit command environment variables, or map an HTTP
header to an environment-variable name with `headerEnv`.

```json
{
  "ok": true,
  "text": "model answer",
  "usage": {"promptTokens": 100, "completionTokens": 40},
  "timings": {
    "routeAdmissionMs": 2,
    "runtimeLoadMs": 0,
    "serverTtftMs": 110,
    "serverPrefillTokensPerSecond": 500,
    "serverDecodeTokensPerSecond": 22,
    "adapterFirstChunkMs": 118,
    "serverTotalMs": 1900
  }
}
```

Quality is never inferred from `ok`. Cases may use `contains`, `notContains`,
`regex`, `jsonFields`, `responseFields`, or a command `judge`. Every receipt
keeps `runtimeStatus` and `qualityStatus` separate.

## Generic agent traces

A command adapter may add one optional `agentTrace` object to its response.
The contract is application-neutral: component names and round purposes come
from Picchio's fixed vocabulary, so human output cannot inherit product names
or arbitrary internal labels.

```json
{
  "agentTrace": {
    "schema": "picchio.agent-trace.v1",
    "components": [
      {
        "id": "user",
        "kind": "user_input",
        "characters": 45,
        "bytes": 45,
        "tokens": 12,
        "countSource": "measured",
        "tokenSource": "tokenized"
      },
      {
        "id": "system",
        "kind": "system_instructions",
        "characters": 61404,
        "bytes": 61404,
        "countSource": "measured"
      },
      {
        "id": "document",
        "kind": "document_context",
        "characters": 46801,
        "bytes": 46801,
        "countSource": "measured",
        "includedIn": "system"
      },
      {
        "id": "history",
        "kind": "conversation_history",
        "characters": 0,
        "bytes": 0,
        "items": 0,
        "countSource": "measured"
      }
    ],
    "rounds": [
      {
        "index": 0,
        "purpose": "planning",
        "usage": {"promptTokens": 26336, "completionTokens": 31},
        "timings": {
          "serverTtftMs": 93200,
          "serverPrefillTokensPerSecond": 282.6,
          "serverDecodeTokensPerSecond": 25.23
        },
        "wireEvidence": {
          "kind": "actual_engine_request_body",
          "provenance": "application_engine_boundary_capture",
          "bodyBase64": "...",
          "sha256": "..."
        }
      }
    ],
    "context": {
      "capacityTokens": 32768,
      "compactAtTokens": 16384,
      "reportedCompactTriggered": false
    },
    "firstVisibleTokenMs": 93200,
    "visibleDecodeTokensPerSecond": 25.23
  }
}
```

Supported component kinds are `user_input`, `system_instructions`,
`document_context`, `conversation_history`, `retrieval_evidence`,
`tool_schemas`, `tool_results`, `attachments`, `chat_template`, and `other`.
`includedIn` marks a subset: a document context included in system instructions
is displayed beneath that total and is never meant to be added to it.
Adapter component IDs exist only to express this relationship; Picchio replaces
them with neutral internal IDs and never renders adapter labels. `countSource`
and `tokenSource` accept `measured`, `tokenized`, `derived`, `estimated`,
`reported`, or `unknown`.

Supported round purposes are `initial`, `planning`, `tool`, `final`, `repair`,
`verification`, and `other`. Every round may carry its own exact engine-boundary
wire body, provider usage, TTFT, prefill, decode, and total server time.

Picchio derives these values itself:

- `currentPromptTokens`: the last round's prompt, used for current context
  pressure;
- `peakPromptTokens`: the largest single request;
- `totalPromptTokensProcessed`: the sum across rounds, used for total work;
- `compactDecision`: `correct`, `false_positive`, `missed`, `not_reported`,
  or `not_evaluated`.

Current context and cumulative work are deliberately different fields. Picchio
never substitutes their sum for the last request. Each traced case writes
`agent-trace.json`, a neutral `agent-trace.md` table, and one verified binary
request body per round. Missing usage or wire evidence makes the trace partial;
it never turns a successful inference into a runtime failure.

The binary request bodies can contain complete prompts, document context,
retrieval evidence, attachments, and tool results. They stay in the local
artifact directory. Treat that directory as private and redact it before
sharing; the Markdown table contains counts and neutral categories only.

## Bare/product parity

```json
{
  "schema": "picchio.parity.manifest.v1",
  "executionOrder": "interleaved",
  "fingerprint": {
    "modelSha256": "...",
    "runtimeBuild": "b9430",
    "context": 32768,
    "kv": "f16",
    "parallel": 1,
    "sampling": {"temperature": 0},
    "maxOutputTokens": 256,
    "projector": false
  },
  "relativeRegressionPercent": 10,
  "absoluteLatencyRegressionMs": 10,
  "lanes": {
    "baseline": {
      "adapter": {"type": "command", "command": ["./bare-adapter"]},
      "fingerprint": {"modelSha256": "...", "runtimeBuild": "b9430", "context": 32768, "kv": "f16", "parallel": 1, "sampling": {"temperature": 0}, "maxOutputTokens": 256, "projector": false}
    },
    "product": {
      "adapter": {"type": "command", "command": ["./product-adapter"]},
      "fingerprint": {"modelSha256": "...", "runtimeBuild": "b9430", "context": 32768, "kv": "f16", "parallel": 1, "sampling": {"temperature": 0}, "maxOutputTokens": 256, "projector": false}
    }
  },
  "calibrationCases": [],
  "cases": [
    {
      "id": "same-prompt-01",
      "maxOutputTokens": 256,
      "request": {"prompt": "..."},
      "rubric": {"contains": ["CODEWORD"]}
    }
  ]
}
```

The top-level `fingerprint` is the requested contract, not a copy of whatever
the runtime happened to report. Each lane fingerprint records the declared
pre-run configuration. Every adapter response must then report actual values:

```json
{
  "received": {
    "inputSha256": "...",
    "promptTokens": 100,
    "modelSha256": "...",
    "runtimeBuild": "b9430",
    "context": 32768,
    "kv": "f16",
    "parallel": 1,
    "sampling": {"temperature": 0},
    "maxOutputTokens": 256,
    "launchArgs": ["llama-server", "--ctx-size", "32768"],
    "runtimePid": 12345,
    "runtimeStartTime": "...",
    "launchSequence": 1,
    "projector": false,
    "cacheN": 0,
    "toolRounds": 0
  },
  "wireEvidence": {
    "kind": "actual_engine_request_body",
    "provenance": "product_engine_boundary_capture",
    "bodyBase64": "eyJtZXNzYWdlcyI6W119",
    "sha256": "..."
  }
}
```

`baseline` uses `direct_engine_send`: the bytes are exactly what its adapter
sent to the engine. `product` uses `product_engine_boundary_capture`: the
bytes were captured where the product called the engine. Reconstructed bodies
and hashes are not causal evidence. Picchio decodes the bytes, writes
`engine-wire-body.bin`, recomputes SHA-256 and compares the two lanes.

The causal gate also checks received prompt tokens against response usage,
actual model SHA/parallel/sampling against the requested fingerprint, exact
wire-body equality, cache state, one-request tool state, and the same runtime
PID/start time/launch sequence/process identity across lanes. Missing or
unequal evidence yields `DIRECTIONAL`, omits causal attribution and exits 5.

`interleaved` is the default. Consecutive fixtures run baseline/product then
product/baseline, producing an A-B-B-A lane pattern without doubling each
fixture. `abba` runs every fixture baseline/product/product/baseline as two
paired replicates. The chosen order, per-phase execution index and replicate
are committed in every receipt and checked before causal attribution.

Latency attribution must cross both thresholds. The default 10% relative and
10 ms absolute gates prevent a few milliseconds over a near-zero baseline
from being labeled user-visible product overhead.

## Artifact

Without `--artifact`, `suite.json` writes `suite.picchio-run/`:

```text
manifest.json
checkpoint.json
events.jsonl
summary.json
cases/*/attempt-*/
rollups/hour-*.json
runtime/*
```

Inputs, raw adapter output, model text, memory/swap snapshots, whole-GPU
samples, quality checks and per-case receipts are immutable. A receipt is
written before its checkpoint pointer. After `SIGKILL`, the next run
reconciles committed receipts, skips them, and opens a new attempt for the
unfinished case.
