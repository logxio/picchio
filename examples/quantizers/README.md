# Four quantizers, one label

Same base model (Qwen/Qwen3.5-9B), same Q4_K_M label, four
published GGUFs. Each card here is `picchio id` output.

| quantizer | file | bytes | sha256 (first 12) |
|---|---|---|---|
| unsloth | Qwen3.5-9B-Q4_K_M.gguf | 5,680,522,464 | 03b74727a860 |
| bartowski | Qwen_Qwen3.5-9B-Q4_K_M.gguf | 6,169,341,984 | d784ce9eda1a |
| lmstudio-community | Qwen3.5-9B-Q4_K_M.gguf | 5,627,044,256 | cd76ec205963 |
| mradermacher | Qwen3.5-9B.Q4_K_M.gguf | 5,627,045,120 | 9fa52e37c829 |

Over the 427 tensors all four files share, effective bits per
weight: lmstudio-community 5.02, mradermacher 5.02, unsloth 5.07,
bartowski 5.27. The bartowski card reads 5.36 whole-file because
that file bundles a 243M-parameter MTP head at q8_0; unsloth ships
the same head as a separate repo.

The unsloth file was walked in full and its sha256 matches the
repo's lfs oid. The other three were walked from the first 16 MiB
(the tensor table lives in the header) plus the file size; all
four sizes were verified byte-exact against the CDN's
Content-Range. Rerun any of it: `python3 picchio.py id <file>`.

Retrieved 2026-07-14, revision main.
