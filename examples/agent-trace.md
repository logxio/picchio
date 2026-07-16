# Agent trace

## Request composition

| Component | Characters | Bytes | Tokens | Evidence | Included in |
|---|---:|---:|---:|---|---|
| User input | 45 | 45 | 12 | measured; tokens tokenized | — |
| System instructions | 61,404 | 61,404 | — | measured | — |
| Document context | 46,801 | 46,801 | — | measured | System instructions |
| Conversation history | 0 | 0 | — | measured | — |

Rows with `Included in` are subsets and must not be added to their parent.

## Inference rounds

| Round | Purpose | Prompt tokens | Completion tokens | TTFT | Prefill tok/s | Decode tok/s | Wire bytes |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | planning | 6,000 | 4 | 1 s | 500 | 20 | 52 |
| 1 | tool | 7,000 | 3 | 1.1 s | 490 | 21 | 48 |
| 2 | final | 8,000 | 5 | 1.2 s | 480 | 22 | 49 |

## Context accounting

| Metric | Value |
|---|---:|
| Current prompt tokens | 8,000 |
| Peak prompt tokens | 8,000 |
| Total prompt tokens processed | 21,000 |
| Context capacity | 32,768 |
| Compact threshold | 16,384 |
| Current context utilization | 24.41% |
| Reported compact | yes |
| Compact should trigger | no |
| Compact decision | FALSE POSITIVE |
| First visible token | 93.2 s |
| Visible decode | 25.23 tok/s |
